# -*- coding: utf-8 -*-
"""H3 双卡固定槽流水：ComfyUI 节点。

节点：
- H3DualGPUPipeline —— 接在模型补丁链末端（LoRA/Sage/SolAttn 之后），
  启用固定槽流水 + 层49副卡 + 步末页回收
- H3TESecondaryGPU —— Qwen3-VL TE 的加载设备指到副卡
"""

import logging
import os
import time

import torch

import comfy.model_management
import comfy.patcher_extension
import comfy.utils
from comfy.quant_ops import QuantizedTensor

from .fixed_slot import SlotPipeline, STREAM_LINERARS

LOG = logging.getLogger("H3FixedSlot")

WRAP_KEY = "h3_fixed_slot_pipeline"
_STATE_KEY = "h3_fixed_slot_executor"
N_MAIN_BLOCKS = 49   # 层 0-48 主卡；层 49 副卡（输出头 final_layer 留主卡）
# TE 用完即卸载：H3TESecondaryGPU 保存 patcher，wrapper 首次采样调用时
# 卸载到内存（TE 编码已在 ReferenceToVideo 节点完成，采样阶段不再需要）
_TE_PATCHER = None


# ======================================================================
# 双卡流水执行器（DIFFUSION_MODEL wrapper + 逐块 patch_replace）
# ======================================================================

class DualGPUExecutor:
    """进程内执行状态。wrapper 每次 forward 从 transformer_options 取回。"""

    def __init__(self, model_patcher, secondary_device, source_path,
                 mode="dual_gpu", hash_file=""):
        self.model_patcher = model_patcher
        self.diffusion_model = model_patcher.get_model_object("diffusion_model")
        self.secondary_device = secondary_device
        self.source_path = source_path
        self.mode = mode                      # dual_gpu / baseline_capture
        self.hash_file = hash_file
        self.pipeline = None                  # baseline 模式不建槽
        if mode == "dual_gpu":
            # 双卡模式：立即建流水管理器（单例按权重路径）
            self.pipeline = SlotPipeline.get(source_path, self.main_device())
        self.layer_hashes = {}                # f"step{s}_layer{l}" -> float
        self.hash_enabled = False             # 显式开关（空字典是假值，不能做条件）
        self._prepared = False
        self.step_count = 0

    def main_device(self):
        return comfy.model_management.get_torch_device()

    # ------------------------------------------------------------------
    def prepare(self):
        """首次 forward 前的一次性准备。"""
        if self._prepared:
            return
        dm = self.diffusion_model
        dev = self.main_device()
        t0 = time.perf_counter()

        # TE 用完即卸载：采样开始前 TE 编码已完成（ReferenceToVideo 节点），
        # 卸载到内存释放副卡显存；之后如需再用 comfy 按需自动加载
        global _TE_PATCHER
        if _TE_PATCHER is not None:
            try:
                comfy.model_management.unload_model_and_clones(_TE_PATCHER)
                LOG.info("[H3FixedSlot] TE 已卸载到内存（编码完成，释放副卡显存）")
            except Exception as e:
                LOG.warning("[H3FixedSlot] TE 卸载失败: %s", e)
            finally:
                _TE_PATCHER = None

        import os as _os
        stage = _os.environ.get("H3_PREP_STAGE", "all")   # 二分调试：bake/unload/resident/slots

        # 1) LoRA 烘焙（生成/命中缓存文件；常驻件烘回权重对象）
        if stage in ("all", "bake"):
            self.pipeline.source.bake(dm, self.model_patcher, dev)
            self.pipeline.source.bake_resident(dm, self.model_patcher, dev)
            # 缓存后重建 plans（形状/dtype/尺寸与原文件不同），同步 pipeline
            self.pipeline.rebuild()
            # bf16 模式：块内 STREAM Linear 清 layout_type（f16 槽张量绝不走 int8 kernel）
            # int8 模式：保留 layout_type → _use_quantized=True 走 int8 kernel
            if not getattr(self.pipeline.source, "int8_mode", False):
                for li in range(self.pipeline.n_layers):
                    for lname in STREAM_LINERARS:
                        lin = comfy.utils.get_attr(dm.blocks[li], lname)
                        lin.layout_type = None

        # 2) 槽接管前先把 legacy 驻留副本打回 CPU——
        #    不用 dm.to(off)：_quantized_apply 会把 QuantizedTensor 重包装成
        #    Parameter，改变 sage 内核的 dispatch 路径（invalid argument）。
        #    改为逐 qdata 手工搬 CPU，绕开 Parameter 重注册。
        LOG.info("[H3FixedSlot] 卸载前主卡 allocated=%.2fG",
                 torch.cuda.memory_allocated(dev) / 2**30)
        for name, mod in dm.named_modules() if stage in ("all", "unload") else []:
            w = getattr(mod, "weight", None)
            if w is None:
                continue
            wd = w.data if isinstance(w, torch.nn.Parameter) else w
            if isinstance(wd, QuantizedTensor) and wd._qdata.device.type == "cuda":
                mod.weight = torch.nn.Parameter(
                    wd._copy_with(
                        qdata=wd._qdata.to("cpu"),
                        params=wd._params.to_device(torch.device("cpu"))),
                    requires_grad=False)
            elif isinstance(wd, torch.Tensor) and wd.device.type == "cuda" \
                    and not isinstance(wd, QuantizedTensor):
                mod.weight = torch.nn.Parameter(wd.to("cpu"),
                                                requires_grad=False)
        torch.cuda.synchronize()
        LOG.info("[H3FixedSlot] 卸载后主卡 allocated=%.2fG（目标<1.5G）",
                 torch.cuda.memory_allocated(dev) / 2**30)

        # 3) 常驻件 → 主卡：refiner / embed 投影 / rope / 时间表 / final
        _resident = stage in ("all", "resident")
        # 警告：含 QuantizedTensor 的模块绝不能 .to()——_quantized_apply 会重包装
        # Parameter 改变 dispatch（invalid argument）。改用安全手工搬运。
        def _safe_to(module, device):
            """安全搬运：小张量参数/buffer 直接 .data.to；QuantizedTensor 走 _copy_with。"""
            import dataclasses
            for pname, p in list(module.named_parameters(recurse=False)):
                pd = p.data
                if isinstance(pd, QuantizedTensor):
                    if pd._qdata.device != device:
                        module._parameters[pname] = torch.nn.Parameter(
                            pd._copy_with(
                                qdata=pd._qdata.to(device),
                                params=pd._params.to_device(device)),
                            requires_grad=False)
                elif pd.device != device:
                    module._parameters[pname] = torch.nn.Parameter(
                        pd.to(device), requires_grad=False)
            for bname, b in list(module.named_buffers(recurse=False)):
                if b.device != device:
                    module._buffers[bname] = b.to(device)

        if _resident:
            # 常驻件永久留 CPU（与基线 lowvram 形态一致）：comfy cast_bias_weight
            # 每步自动搬运（vbar/legacy 路径）；曾怀疑 _safe_to 搬运破坏 context
            # （454 炸点），且小件搬运收益微小（~200MB/步 H2D @12GB/s ≈ 17ms）。
            # 基线等价设计，不再提供 SKIP 开关。
            if hasattr(dm, "adaln_t_table"):
                dm.adaln_t_table.data = dm.adaln_t_table.data.to("cpu")
            # inv_freq 永久留 CPU（与基线一致）：lowvram 原生路径每步 cast_to
            # 从 CPU buffer 转换；搬上主卡曾被怀疑破坏 context（454 炸点），
            # 且无性能收益（128KB H2D/步可忽略）——基线等价设计。
            dm.rope.inv_freq.data = dm.rope.inv_freq.data.to("cpu")
            _safe_to(dm.final_layer, dev)
        LOG.info("[H3FixedSlot] 探针A 常驻件后 allocated=%.2fG",
                 torch.cuda.memory_allocated(dev) / 2**30)

        # 3) 块小张量永久留 CPU（与基线 lowvram 形态一致）：
        #    norm/adaln/q_norm/k_norm + per-row scale 全由 comfy cast_bias_weight
        #    每步自动搬运——曾怀疑 _safe_to/_copy_with 逐块搬运破坏 context
        #    （454/F2 炸点），且 50 层小件搬运收益微小（~170MB/步 H2D ≈ 15ms）。
        #    f16 模式下 scale 搬运是 int8 路径遗留物，直接删除。
        sec = self.secondary_device

        # 4) 显存槽（末层槽在副卡）+ 预取线程
        LOG.info("[H3FixedSlot] 探针B 块小张量后 allocated=%.2fG",
                 torch.cuda.memory_allocated(dev) / 2**30)
        if stage in ("all", "slots"):
            self.pipeline.build_vram_slots(dm, sec)
            # 探针C：建槽后立即做一次最小 H2D——若此处炸，context 在建槽时被破坏
            _c = torch.zeros(1024, device=dev)
            _c.copy_(torch.zeros(1024, pin_memory=True), non_blocking=True)
            torch.cuda.synchronize(dev)
            LOG.info("[H3FixedSlot] 探针C 建槽后 H2D OK")
            self.pipeline.start()
            # 探针D：预取线程启动后同样探针——若此处炸，预取线程是凶手
            _d = torch.zeros(1024, device=dev)
            _d.copy_(torch.zeros(1024, pin_memory=True), non_blocking=True)
            torch.cuda.synchronize(dev)
            LOG.info("[H3FixedSlot] 探针D 预取启动后 H2D OK")
            # 探针E：跨卡拷贝（TE 在副卡 → cond 在 cuda:1 → 主卡 .to() 的真实路径）
            _e_src = torch.zeros(1024, device=sec)
            _e_dst = _e_src.to(dev)
            torch.cuda.synchronize(dev)
            LOG.info("[H3FixedSlot] 探针E 跨卡 cuda:%s→cuda:%s OK",
                     sec.index if sec.type == "cuda" else sec, dev.index)

        # 5) 阻断 LoRA 运行时叠加：
        #    - model_patcher.patches（legacy load 会再设 LowVramPatch）
        #    - Linear 上已存在的 weight_function/bias_function
        #    注意：patches={} 只防 patches 再次生效，删不掉已挂在 Linear 实例上的
        #    weight_function 属性——常驻件/adaln 等所有 Linear 都要清，
        #    否则运行时搬运函数对已在 GPU 的权重再搬运 → CUDA invalid argument
        self.model_patcher.patches = {}
        self.model_patcher.backup = {}
        _n_cleared = 0
        for _name, _mod in dm.named_modules():
            if isinstance(_mod, torch.nn.Module) and hasattr(_mod, "weight_function"):
                if getattr(_mod, "weight_function", None):
                    _mod.weight_function = []
                    _n_cleared += 1
                if getattr(_mod, "bias_function", None):
                    _mod.bias_function = []
                for pk in ("weight", "bias"):
                    if hasattr(_mod, f"{pk}_lowvram_function"):
                        delattr(_mod, f"{pk}_lowvram_function")
        LOG.info("[H3FixedSlot] 清理运行时搬运函数: %d 个 Linear", _n_cleared)

        self._prepared = True
        LOG.info("[H3FixedSlot] prepare 完成 %.1fs（块0-48主卡槽 / 块%d副卡槽）",
                 time.perf_counter() - t0, N_MAIN_BLOCKS)


def _hash_capture(executor, li, out, step=None):
    """逐层求和（含步号，跨步可对比同一层）。"""
    key = f"s{step}_l{li}" if step is not None else f"l{li}"
    executor.layer_hashes[key] = out["img"].detach().float().sum().item()


def _make_block_patch(executor, li):
    """生成第 li 块的 patch_replace：注入槽权重 → 调原块。"""

    def block_patch(args, replacement_context):
        pipeline = executor.pipeline
        host = pipeline.wait_layer(li)
        turn = pipeline.load_slot(li, host)
        pipeline.install(executor.diffusion_model, li, turn)
        # 探针I：install 后立即核对 4 矩阵设备（mat2@cpu 排查）
        _blk = executor.diffusion_model.blocks[li]
        _q = _blk.attn.qkv_proj.weight
        LOG.info("[H3FixedSlot] 探针I 块%d 槽=%s qkv type=%s dev=%s dtype=%s "
                 "lin_id=%s blk_attn_id=%s",
                 li, turn, type(_q).__name__, _q.data.device, _q.data.dtype,
                 id(_blk.attn.qkv_proj), id(_blk.attn))
        if li + 1 <= N_MAIN_BLOCKS:
            pipeline.request(li + 1)   # 流水：算 li 时预取 li+1
        out = replacement_context["original_block"](args)
        if executor.hash_enabled:
            _hash_capture(executor, li, out, executor.step_count)
        return out

    return block_patch


def _make_last_block_patch(executor):
    """层 49 也在主卡执行（B 槽轮换）。副卡 3080 仅承载 TE（14.5G 常驻），
    层 49 若在副卡算会 OOM（20G - TE 14.5G - 槽 0.54G < sage 工作区）。
    数值等价：同权重同算子同设备主卡。"""
    def block_patch(args, replacement_context):
        pipeline = executor.pipeline
        li = pipeline.n_layers - 1
        # 权重注入（主卡轮换槽；load_slot 末层识别见 H3_LAST_ON_SEC 开关）
        host = pipeline.wait_layer(li)
        turn = pipeline.load_slot(li, host)
        pipeline.install(executor.diffusion_model, li, turn)
        out = replacement_context["original_block"](args)
        if executor.hash_enabled:
            _hash_capture(executor, li, out, executor.step_count)
        return out

    return block_patch


def _make_baseline_patch(executor, li):
    """基线模式：不动权重，只在原块后抓 hash。"""

    def block_patch(args, replacement_context):
        out = replacement_context["original_block"](args)
        if executor.hash_enabled:
            _hash_capture(executor, li, out, executor.step_count)
        return out

    return block_patch


def diffusion_model_wrapper(wrap_executor, x, timestep, context,
                            transformer_options=None, minimax_payload=None,
                            **kwargs):
    """DIFFUSION_MODEL wrapper：接管一次完整模型 forward。"""
    options = dict(transformer_options or {})
    executor = options.get(_STATE_KEY)
    LOG.info("[H3FixedSlot] wrapper 调用: executor=%s mode=%s",
             "有" if executor is not None else "无",
             getattr(executor, "mode", "-"))
    if executor is None:
        return wrap_executor(x, timestep, context, options,
                             minimax_payload=minimax_payload, **kwargs)

    if executor.mode == "dual_gpu":
        import os as _os
        if _os.environ.get("H3_SKIP_PREPARE") != "1":
            executor.prepare()
        patches_replace = dict(options.get("patches_replace") or {})
        dit = dict(patches_replace.get("dit") or {})
        # 二分调试开关：H3_SKIP_BLOCKS=1 时只挂 baseline patch（不装槽权重），
        # 用于把错误定位到 prepare/常驻件（仍炸） vs 槽换装（不炸）
        import os as _os
        if _os.environ.get("H3_SKIP_BLOCKS") == "1":
            for li in range(executor.pipeline.n_layers):
                dit[("double_block", li)] = _make_baseline_patch(executor, li)
        else:
            for li in range(N_MAIN_BLOCKS):
                dit[("double_block", li)] = _make_block_patch(executor, li)
            dit[("double_block", executor.pipeline.n_layers - 1)] = \
                _make_last_block_patch(executor)
        patches_replace["dit"] = dit
        options["patches_replace"] = patches_replace
        executor.pipeline.request(0)   # 首层预取
    else:
        # baseline_capture：只挂 hash 钩子，权重路径完全是原生 legacy 行为
        if executor.hash_enabled:
            patches_replace = dict(options.get("patches_replace") or {})
            dit = dict(patches_replace.get("dit") or {})
            n_layers = len(executor.diffusion_model.blocks)
            for li in range(n_layers):
                dit[("double_block", li)] = _make_baseline_patch(executor, li)
            patches_replace["dit"] = dit
            options["patches_replace"] = patches_replace

    try:
        # 探针G：验证 x / context / timestep 的设备（怀疑 x 在副卡）
        try:
            if executor.mode == "dual_gpu":
                _gd = executor.main_device()
                _x0 = x[0] if isinstance(x, (list, tuple)) else x
                LOG.info("[H3FixedSlot] 探针G x0 dev=%s ctx dev=%s ts dev=%s shape=%s",
                         _x0.device, context.device if context is not None else None,
                         timestep.device if hasattr(timestep, "device") else None,
                         tuple(_x0.shape))
        except Exception as _ex:
            LOG.info("[H3FixedSlot] 探针G 异常: %r", _ex)
        # 探针F：前向前诊断 payload 的 cond latents 设备 + 真实拷贝
        try:
            if executor.mode == "dual_gpu" and minimax_payload is not None:
                _dev = executor.main_device()
                for _zk, _zs in (("video", minimax_payload.get("cond_video_latents") or []),
                                 ("audio", minimax_payload.get("cond_audio_latents") or [])):
                    for _i, _z in enumerate(_zs[:1]):
                        LOG.info("[H3FixedSlot] 探针F %s_cond[%d]: shape=%s dev=%s dtype=%s",
                                 _zk, _i, tuple(_z.shape), _z.device, _z.dtype)
                        _t0 = time.perf_counter()
                        _probe = _z.to(torch.float32).flatten()[:1024].to(_dev)
                        torch.cuda.synchronize(_dev)
                        LOG.info("[H3FixedSlot] 探针F %s_cond[%d] → 主卡拷贝 OK %.3fs",
                                 _zk, _i, time.perf_counter() - _t0)
                        # 探针F2：完整复刻 _cond_video_rows 操作链（CPU einsum+randn+混合+H2D）
                        if _zk == "video":
                            import comfy.ldm.minimax.model as _mm
                            _r = _mm.patchify_video(_z.to(torch.float32),
                                                    executor.diffusion_model.patch_size)
                            LOG.info("[H3FixedSlot] 探针F2a patchify后 dev=%s", _r.device)
                            _r_c = _r.clone()                       # 连续化
                            _p1 = _r_c[:256].to(_dev); torch.cuda.synchronize(_dev)
                            LOG.info("[H3FixedSlot] 探针F2b 前256行H2D OK")
                            _p2 = _r_c.to(_dev); torch.cuda.synchronize(_dev)
                            LOG.info("[H3FixedSlot] 探针F2c 全量H2D OK")
        except Exception as _ex:
            LOG.info("[H3FixedSlot] 探针F 异常: %r", _ex)
        out = wrap_executor(x, timestep, context, options,
                            minimax_payload=minimax_payload, **kwargs)
    finally:
        if executor.mode == "dual_gpu":
            pipeline = executor.pipeline
            pipeline.stats["steps"] += 1
            executor.step_count += 1
            pipeline.recycle_pages()   # 步末回收干净页
            if pipeline.stats["steps"] % 4 == 0:
                pipeline.log_stats()
        else:
            executor.step_count += 1
        # 逐层 hash 落盘（每次采样步覆盖写，最终保留全部步×层）
        if executor.hash_enabled:
            import json
            path = executor.hash_file or os.path.join(
                os.path.dirname(executor.source_path or "."),
                f"h3_layer_sums_{executor.mode}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in sorted(executor.layer_hashes.items())},
                          f, indent=1)
    return out


# ======================================================================
# 节点定义
# ======================================================================

class H3DualGPUPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "secondary_gpu": ("INT", {"default": 1, "min": 0, "max": 7,
                                          "label": "副卡序号 Secondary GPU"}),
                "model_path": ("STRING", {"default": "",
                                          "label": "权重文件路径(留空自动) Weight path"}),
                "layer_debug": ("BOOLEAN", {"default": False,
                                            "label": "逐层求和记录 Layer debug"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "H3DualGPU"
    DESCRIPTION = "MiniMax H3 双卡固定槽流水：主卡层0-48两槽交替，副卡层49"

    def apply(self, model, secondary_gpu, model_path, layer_debug):
        model_clone = model.clone()
        if not model_path:
            model_path = _locate_weight_file(model_clone)
        sec_device = torch.device("cuda", secondary_gpu)

        executor = DualGPUExecutor(model_clone, sec_device, model_path,
                                   mode="dual_gpu",
                                   hash_file="")
        if layer_debug:
            executor.hash_enabled = True   # 启用逐层求和
        to = model_clone.model_options.setdefault("transformer_options", {})
        to[_STATE_KEY] = executor
        model_clone.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAP_KEY,
            diffusion_model_wrapper,
        )
        return (model_clone,)


def _locate_weight_file(model_patcher):
    """从 ModelPatcher 找回 safetensors 路径。"""
    init = getattr(model_patcher, "cached_patcher_init", None)
    if init is not None and len(init) > 1 and init[0].__name__ == "load_diffusion_model":
        return init[1][0]
    raise RuntimeError("无法自动定位权重文件，请在节点里填 model_path")


class H3TESecondaryGPU:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "secondary_gpu": ("INT", {"default": 1, "min": 0, "max": 7,
                                          "label": "副卡序号 Secondary GPU"}),
            },
        }

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "apply"
    CATEGORY = "H3DualGPU"
    DESCRIPTION = "Qwen3-VL TE 卸载到副卡（不占主卡显存，与 DiT 主卡并行）"

    def apply(self, clip, secondary_gpu):
        clip_clone = clip.clone()
        clip_clone.patcher.load_device = torch.device("cuda", secondary_gpu)
        # 低显存副卡（如 16G 5070Ti）放不下 TE 全量常驻 14.5G：
        # offload_device 设 CPU，comfy 按需搬层，常驻降到 ~2G
        clip_clone.patcher.offload_device = torch.device("cpu")
        # 模型若已按主卡加载（loaded_models 注册在 cuda:0），须先卸载，
        # 否则 load_models_gpu 视为已加载直接复用，load_device 改动不生效
        try:
            comfy.model_management.unload_model_and_clones(clip_clone.patcher)
        except Exception as e:
            LOG.warning("[H3FixedSlot] TE 预卸载跳过: %s", e)
        LOG.info("[H3FixedSlot] TE load_device -> %s (offload: cpu)",
                 clip_clone.patcher.load_device)
        # 保存 patcher 引用：TE 编码在 136 节点完成，采样开始后不再需要——
        # wrapper 首次调用时卸载到内存，释放副卡显存（TE 用后即弃）
        global _TE_PATCHER
        _TE_PATCHER = clip_clone.patcher
        return (clip_clone,)


class H3BaselineCapture:
    """基线 hash 采集：不动权重，原生 legacy 路径跑，抓逐层求和。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "H3DualGPU"
    DESCRIPTION = "基线逐层求和采集（原生权重路径，用于数值一致性对比）"

    def apply(self, model):
        model_clone = model.clone()
        executor = DualGPUExecutor(
            model_clone, None, _locate_weight_file(model_clone),
            mode="baseline_capture")
        executor.hash_enabled = True   # 启用逐层求和
        to = model_clone.model_options.setdefault("transformer_options", {})
        to[_STATE_KEY] = executor
        model_clone.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAP_KEY,
            diffusion_model_wrapper,
        )
        return (model_clone,)


NODE_CLASS_MAPPINGS = {
    "H3DualGPUPipeline": H3DualGPUPipeline,
    "H3TESecondaryGPU": H3TESecondaryGPU,
    "H3BaselineCapture": H3BaselineCapture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3DualGPUPipeline": "H3 双卡固定槽流水 Dual-GPU Fixed-Slot Pipeline",
    "H3TESecondaryGPU": "H3 TE 卸载副卡 TE → Secondary GPU",
    "H3BaselineCapture": "H3 基线逐层求和 Baseline Layer-Sum Capture",
}
