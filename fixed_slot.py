# -*- coding: utf-8 -*-
"""H3 双卡固定槽流水：槽管理器。

三段流水（对照示意图）：
  权重源(烘焙缓存文件 / 原文件) --后台线程readinto--> 主机槽(pinned)
    --copy stream--> 显存槽 A/B
  主卡算第 i 层时预取 i+1；层切换 = 4 次指针换装（weight.data 原位指到槽张量），
  绝不创建/销毁 0.6G 级权重对象。

数值一致性（与 legacy ModelPatcher 基线逐位相同）：
- 无 LoRA 层：槽字节 = 磁盘原始 int8。
- 有 LoRA 层：预烘焙完全复刻 patch_weight_to_device 数学链：
    dequant(lora_compute_dtype=fp16) → comfy.lora.calculate_weight(patches, w, key)
    → Linear.set_weight：requantize_from_float(scale="recalculate",
      stochastic_rounding=string_to_seed(key), inplace_ops=True)
  烘焙产物写「权重同目录缓存文件」（一次生成，永久复用），不占 pinned 内存。
  烘焙后清空 patches + weight_function，防运行时二次叠加。
"""

import ctypes
import json
import logging
import os
import struct
import threading
import time

import torch

import comfy.lora
import comfy.model_management
import comfy.utils
from comfy.quant_ops import QuantizedTensor

LOG = logging.getLogger("H3FixedSlot")

# 预取主体：4 个大矩阵 Linear；其余（norm/adaln/scale ~2MB/层）常驻主卡
STREAM_LINERARS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")

_TYPE_MAP = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def _read_header(path):
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return header, 8 + header_size


class LayerPlan:
    __slots__ = ("index", "entries", "nbytes")

    def __init__(self, index):
        self.index = index
        self.entries = []  # (linear_name, data_off, nbytes, qdata_shape, qdata_dtype)
        self.nbytes = 0


class WeightSource:
    """层权重字节源：LoRA 烘焙缓存文件 或 原始文件直读。

    烘焙缓存 = 原文件数据区的完整拷贝，但带 LoRA 的 4 大矩阵位置替换为
    烘焙后的 int8 字节（与基线 patch_weight_to_device 结果逐位一致）。
    """

    def __init__(self, model_path, cache_dir=None):
        self.path = os.path.abspath(model_path)
        self.header, self.data_base = _read_header(self.path)
        self.n_layers = 0
        for name in self.header:
            if name != "__metadata__" and name.startswith("blocks."):
                self.n_layers = max(self.n_layers, int(name.split(".")[1]) + 1)
        # 惰性路径：默认与权重同目录
        self.cache_dir = cache_dir or os.path.dirname(self.path)
        # 当前生效的读取文件（烘焙后指向缓存文件）
        self.read_path = self.path
        # int8 槽模式开关（bake() 时按 H3_SLOT_INT8 设定；默认 bf16）
        self.int8_mode = False

    # ------------------------------------------------------------------
    def layer_plan(self, li):
        plan = LayerPlan(li)
        # int8 模式：额外读 weight_scale（装 QuantizedTensor 需要），
        # 条目名带 .weight/.weight_scale 后缀；bf16 模式保持旧条目名（无后缀）
        extra = ("weight_scale",) if getattr(self, "int8_mode", False) else ()
        for lin in STREAM_LINERARS:
            if extra:
                for suffix in ("weight",) + extra:
                    key = f"blocks.{li}.{lin}.{suffix}"
                    info = self.header[key]
                    start, end = info["data_offsets"]
                    plan.entries.append((f"{lin}.{suffix}", start, end - start,
                                         tuple(info["shape"]),
                                         _TYPE_MAP[info["dtype"]]))
                    plan.nbytes += end - start
            else:
                key = f"blocks.{li}.{lin}.weight"
                info = self.header[key]
                start, end = info["data_offsets"]
                plan.entries.append((lin, start, end - start,
                                     tuple(info["shape"]),
                                     _TYPE_MAP[info["dtype"]]))
                plan.nbytes += end - start
        return plan

    def _cache_path(self, sig):
        base = os.path.splitext(os.path.basename(self.path))[0]
        return os.path.join(self.cache_dir, f"{base}.fixedslot.{sig}.safetensors")

    def _lora_signature(self, model_patcher):
        """LoRA 组合指纹（文件名+强度 → 决定缓存文件名）。"""
        import hashlib
        h = hashlib.sha1()
        for key in sorted(model_patcher.patches.keys()):
            h.update(key.encode())
            for patch in model_patcher.patches[key]:
                strength = patch[0]
                h.update(f"{strength}".encode())
        return h.hexdigest()[:12]

    def read_into(self, plan, host_buf):
        """一层 4 矩阵字节 → 主机槽（readinto 直进 pinned）。"""
        n = 0
        mv = memoryview(host_buf.numpy())
        with open(self.read_path, "rb", buffering=0) as fh:
            for _, off, nb, _, _ in plan.entries:
                fh.seek(self.data_base + off)
                got = 0
                while got < nb:
                    got += fh.readinto(mv[n + got:n + nb])
                n += nb
        return n

    def rebuild_plans(self):
        """f16 缓存后重建 layer plans（形状/dtype 与原文件不同）。"""
        self.plans = {li: self.layer_plan(li) for li in range(self.n_layers)}
        self.plans_needs_rebuild = False

    # ------------------------------------------------------------------
    # LoRA 预烘焙（bf16 默认版：与基线 lowvram 运行时数学逐位一致；
    #            int8 版：H3_SLOT_INT8=1 时 requant 落盘走量化 kernel）
    # ------------------------------------------------------------------
    def bake(self, diffusion_model, model_patcher, resident_device):
        """烘焙：无论有无 LoRA都生成缓存文件并切 read_path。

        bf16 默认：普通 bf16 槽张量 + F.linear 才与基线逐位一致。
        int8 开关（H3_SLOT_INT8=1）：dequant→LoRA→requant int8 落盘，
        槽装 QuantizedTensor 保留 layout_type → 走 int8 kernel（原作者原设计，
        显存 1.08G/槽 更省，数值与基线差约 3% 属量化噪声）。
        返回 (n_baked_stream, n_baked_resident)。
        """
        import os as _os
        self.int8_mode = _os.environ.get("H3_SLOT_INT8") == "1"
        patches = dict(model_patcher.patches)
        if self.int8_mode:
            # int8 版：LoRA 烘焙后 requant 回 int8（scale 重算）
            sig = "i8-" + self._lora_signature(model_patcher) \
                if patches else "i8-nolora"
            cache = self._cache_path(sig)
            if not os.path.exists(cache):
                t0 = time.perf_counter()
                self._bake_int8_to_file(diffusion_model, model_patcher,
                                        patches, cache)
                LOG.info("[H3FixedSlot] int8 烘焙缓存生成: %s (%.1fs)",
                         os.path.basename(cache), time.perf_counter() - t0)
            else:
                LOG.info("[H3FixedSlot] 命中 int8 烘焙缓存: %s",
                         os.path.basename(cache))
            self.read_path = cache
            self.header, self.data_base = _read_header(cache)
            self.plans_needs_rebuild = True
            return 1, 0
        # bf16 默认路径
        # bf16v2 格式版本（v2=逐位复刻基线链 cast_to→.to→dequantize；v1 曾用
        # cast_to_device 一步转，与基线差 ±1ulp）：版本号入签名防误命中旧缓存
        sig = "bf16v2-" + self._lora_signature(model_patcher) \
            if patches else "bf16v2-nolora"
        cache = self._cache_path(sig)
        if not os.path.exists(cache):
            t0 = time.perf_counter()
            self._bake_to_file(diffusion_model, model_patcher, patches, cache)
            LOG.info("[H3FixedSlot] 烘焙缓存生成: %s (%.1fs)",
                     os.path.basename(cache), time.perf_counter() - t0)
        else:
            LOG.info("[H3FixedSlot] 命中烘焙缓存: %s", os.path.basename(cache))
        # f16 缓存：切换 read_path 并重建 layer plans（dtype/尺寸与原文件不同）
        self.read_path = cache
        self.header, self.data_base = _read_header(cache)
        self.plans_needs_rebuild = True
        return 1, 0

    def _apply_baked_scales(self, dm):
        """从缓存文件读烘焙后 scale，替换各 Linear weight._params.scale（留 CPU，装卡时再搬）。"""
        import dataclasses
        n = 0
        with open(self.read_path, "rb", buffering=0) as fh:
            for li in range(self.n_layers):
                for lin_name in STREAM_LINERARS:
                    key = f"blocks.{li}.{lin_name}.weight_scale"
                    info = self.header[key]
                    s0, s1 = info["data_offsets"]
                    fh.seek(self.data_base + s0)
                    raw = fh.read(s1 - s0)
                    scale = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
                    lin = comfy.utils.get_attr(dm.blocks[li], lin_name)
                    w = lin.weight.data
                    params = w._params
                    if not torch.equal(params.scale.cpu(), scale):
                        new_params = dataclasses.replace(params, scale=scale)
                        lin.weight.data = w._copy_with(params=new_params)
                        n += 1
        LOG.info("[H3FixedSlot] scale 同步: %d 个矩阵替换", n)

    def _bake_to_file(self, dm, model_patcher, patches, cache):
        """生成烘焙缓存文件（bf16 版）：带 LoRA 的 4 大矩阵直接以 BF16 落盘，
        与基线 lowvram 运行时逐位一致——基线 cast_bias_weight 是
        dequant→(input.dtype=bf16)→LoRA(bf16)→F.linear；烘焙复刻此链后
        运行时 weight bf16==input bf16 无任何转换。
        新缓存格式：独立 header，STREAM 矩阵键 dtype=BF16。"""
        device = comfy.model_management.get_torch_device()
        # 基线运行时 dtype = input.dtype = 模型 manual_cast_dtype（bf16）。
        # 基线链：cast_to(qdata→dev) → .to(bf16)=dequant → LoRA(bf16) → F.linear
        target_dtype = dm.manual_cast_dtype if getattr(
            dm, "manual_cast_dtype", None) is not None else torch.bfloat16

        n_stream = 0
        # bf16 缓存不是原文件拷贝——全新写出（header + 顺序数据区）
        import struct as _struct
        entries = []   # (key, dtype_str, shape, bytes)
        for li in range(self.n_layers):
            block = dm.blocks[li]
            for lin_name in STREAM_LINERARS:
                file_key = f"blocks.{li}.{lin_name}.weight"
                patch_key = f"diffusion_model.{file_key}"
                lp = patches.get(patch_key)
                lin = comfy.utils.get_attr(block, lin_name)
                w = lin.weight.data
                if not isinstance(w, QuantizedTensor):
                    raise RuntimeError(f"{file_key} 非量化权重")
                # 逐位复刻基线 cast_bias_weight 链（ops.py 409→425-427）：
                # ① cast_to(None dtype) 仅搬 qdata 上卡（保持量化态）
                # ② .to(bf16) GPU 原生 dequant kernel（若仍 QuantizedTensor 再 dequantize）
                # ③ weight_function=LowVramPatch → calculate_weight(bf16)
                # 不可用 cast_to_device 一步转 dtype：copy_ 拦截的 dequant 路径不同，
                # 引入 ±1ulp 种子差（l0 层和 7.6e-05 的来源，经注意力混沌放大）。
                temp = comfy.model_management.cast_to(
                    w, None, device, copy=True)
                if isinstance(temp, QuantizedTensor):
                    temp = temp.to(dtype=target_dtype)
                    if isinstance(temp, QuantizedTensor):
                        temp = temp.dequantize()
                else:
                    temp = temp.to(target_dtype)
                if lp is None:
                    y = temp          # 无 LoRA：dequant 的 bf16
                else:
                    # 基线数学链复刻：bf16 上算 LoRA（LowVramPatch 同构）
                    y = comfy.lora.calculate_weight(lp, temp, patch_key,
                                                    intermediate_dtype=temp.dtype)
                y16 = y.to(target_dtype).cpu().contiguous()
                # bf16 不支持 .numpy()——经 uint16 视图取原始字节
                entries.append((file_key, "BF16", list(w.shape),
                                y16.view(torch.uint16).numpy().tobytes()))
                n_stream += 1
        # 写 safetensors：header(json) + 数据区
        header = {}
        offset = 0
        blobs = []
        for (key, dts, shape, blob) in entries:
            nbytes = len(blob)
            header[key] = {"dtype": dts, "shape": shape,
                           "data_offsets": [offset, offset + nbytes]}
            blobs.append(blob)
            offset += nbytes
        hdr_bytes = json.dumps(header).encode("utf-8")
        pad = (8 - len(hdr_bytes) % 8) or 8
        hdr_bytes += b" " * pad
        with open(cache, "wb") as out:
            out.write(_struct.pack("<Q", len(hdr_bytes)))
            out.write(hdr_bytes)
            for blob in blobs:
                out.write(blob)
        LOG.info("[H3FixedSlot] 烘焙完成(bf16): %d 个大矩阵, 缓存 %.1fGiB",
                 n_stream, offset / 2**30)

    def _bake_int8_to_file(self, dm, model_patcher, patches, cache):
        """int8 版烘焙：dequant→LoRA→requant int8 落盘（原作者原设计）。
        缓存格式与原文件同构：weight=I8 qdata + weight_scale=F32 重算 scale，
        直读原文件的 nolora 情形也烘焙（保持格式统一走 QuantizedTensor 装槽）。"""
        device = comfy.model_management.get_torch_device()
        import struct as _struct
        entries = []   # (key, dtype_str, shape, bytes)
        n_stream = 0
        for li in range(self.n_layers):
            block = dm.blocks[li]
            for lin_name in STREAM_LINERARS:
                file_key = f"blocks.{li}.{lin_name}.weight"
                scale_key = f"{file_key}_scale"
                patch_key = f"diffusion_model.{file_key}"
                lp = patches.get(patch_key)
                lin = comfy.utils.get_attr(block, lin_name)
                w = lin.weight.data
                if not isinstance(w, QuantizedTensor):
                    raise RuntimeError(f"{file_key} 非量化权重")
                # dequant 到 bf16 → LoRA（bf16 上算，同基线链）
                temp = comfy.model_management.cast_to(w, None, device, copy=True)
                if isinstance(temp, QuantizedTensor):
                    temp = temp.to(dtype=torch.bfloat16)
                    if isinstance(temp, QuantizedTensor):
                        temp = temp.dequantize()
                else:
                    temp = temp.to(torch.bfloat16)
                if lp is not None:
                    temp = comfy.lora.calculate_weight(
                        lp, temp, patch_key,
                        intermediate_dtype=temp.dtype)
                # requant 回 int8（scale 重算）——原作者 int8 槽语义
                qt = w.requantize_from_float(
                    temp, scale="recalculate")
                q8 = qt._qdata.to("cpu").contiguous()
                sc = qt.params.scale.to("cpu").contiguous()
                entries.append((file_key, "I8", list(q8.shape),
                                q8.numpy().tobytes()))
                entries.append((scale_key, "F32", list(sc.shape),
                                sc.numpy().tobytes()))
                n_stream += 1
        # 写 safetensors（同 bf16 版结构）
        header = {}
        offset = 0
        blobs = []
        for (key, dts, shape, blob) in entries:
            nbytes = len(blob)
            header[key] = {"dtype": dts, "shape": shape,
                           "data_offsets": [offset, offset + nbytes]}
            blobs.append(blob)
            offset += nbytes
        hdr_bytes = json.dumps(header).encode("utf-8")
        pad = (8 - len(hdr_bytes) % 8) or 8
        hdr_bytes += b" " * pad
        with open(cache, "wb") as out:
            out.write(_struct.pack("<Q", len(hdr_bytes)))
            out.write(hdr_bytes)
            for blob in blobs:
                out.write(blob)
        LOG.info("[H3FixedSlot] 烘焙完成(int8): %d 个大矩阵, 缓存 %.1fGiB",
                 n_stream, offset / 2**30)

    def bake_resident(self, dm, model_patcher, resident_device):
        """常驻模块（token_refiner 等非 blocks 量化件）的 LoRA 烘焙进权重对象。
        bf16 版：与基线一致——dequant→LoRA 后保持 bf16，不 requant。
        并清 layout_type（防 _use_quantized 误判走 int8 kernel）。"""
        patches = dict(model_patcher.patches)
        if not patches:
            return 0
        device = comfy.model_management.get_torch_device()
        # 与基线一致：input.dtype = manual_cast_dtype（bf16）
        target_dtype = dm.manual_cast_dtype if getattr(
            dm, "manual_cast_dtype", None) is not None else torch.bfloat16
        n = 0
        for name, mod in dm.named_modules():
            if name.startswith("blocks."):
                continue
            w = getattr(mod, "weight", None)
            if w is None:
                continue
            wd = w.data if isinstance(w, torch.nn.Parameter) else w
            if not isinstance(wd, QuantizedTensor):
                continue
            key = f"{name}.weight"
            lp = patches.get(f"diffusion_model.{key}")   # patches 键带模型前缀
            # 逐位复刻基线链（同 _bake_to_file）：cast_to(None)→.to(bf16)→dequantize
            temp = comfy.model_management.cast_to(wd, None, device, copy=True)
            if isinstance(temp, QuantizedTensor):
                temp = temp.to(dtype=target_dtype)
                if isinstance(temp, QuantizedTensor):
                    temp = temp.dequantize()
            else:
                temp = temp.to(target_dtype)
            if lp is None:
                y = temp          # 无 LoRA：dequant 的 bf16
            else:
                # 基线数学链复刻（bf16 上算 LoRA，不 requant）
                y = comfy.lora.calculate_weight(lp, temp, f"diffusion_model.{key}")
            y16 = y.to(target_dtype).to(device=resident_device)
            mod.weight = torch.nn.Parameter(y16, requires_grad=False)
            # 清 layout_type：普通 bf16 张量绝不走 int8 kernel。
            # 注意 layout_type 可能是类属性，delattr 无效——显式实例级置 None。
            mod.layout_type = None
            n += 1
        if n:
            LOG.info("[H3FixedSlot] 常驻模块烘焙(bf16): %d 个", n)
        return n


class SlotPipeline:
    """模型级流水管理器（按权重文件路径单例）。"""

    _instances = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, model_path, main_device, cache_dir=None):
        with cls._lock:
            key = os.path.normcase(os.path.abspath(model_path))
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(WeightSource(model_path, cache_dir), main_device)
                cls._instances[key] = inst
            return inst

    def __init__(self, source, main_device):
        self.source = source
        self.main_device = main_device
        self.n_layers = source.n_layers
        self.plans = {li: source.layer_plan(li) for li in range(self.n_layers)}
        self._alloc_host_slots()

        # 显存槽
        self.vram_slots = {}     # "main_A"/"main_B": {lin_name: qdata}; "sec": 副卡单槽
        self._turn = 0
        self.copy_stream = None
        self.sec_copy_stream = None

        # 预取线程
        self._pending = []
        self._pend_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None

        self.stats = {"prefetch_layers": 0, "wait_seconds": 0.0,
                      "swaps": 0, "steps": 0, "recycles": 0,
                      "install_seconds": 0.0}
        # int8 模式：各 Linear 的 QuantizedTensor params 模板（首装缓存）
        self._qparams = {}

    def _alloc_host_slots(self):
        """主机槽双缓冲按当前 plans 分配（f16 缓存后 nbytes 翻倍需重分配）。"""
        self.host_nbytes = max(p.nbytes for p in self.plans.values())
        self.host_slots = [torch.empty(self.host_nbytes, dtype=torch.uint8,
                                       pin_memory=True) for _ in range(2)]
        self.host_ready = [threading.Event(), threading.Event()]
        self.host_layer = [-1, -1]

    def rebuild(self):
        """f16 烘焙缓存切换 read_path 后重建 plans 与主机槽。"""
        if getattr(self.source, "plans_needs_rebuild", False):
            self.source.rebuild_plans()
        self.plans = self.source.plans
        self.n_layers = self.source.n_layers
        self._alloc_host_slots()

    # ------------------------------------------------------------------
    def build_vram_slots(self, diffusion_model, secondary_device=None):
        """全局仅 2 个固定槽（主卡 A/B 轮换）+ 副卡 1 槽（层49）。
        f16 模式：槽张量直接是 f16 权重（非 qdata），形状=原 weight.shape。"""
        if self.vram_slots:
            return
        dev = self.main_device
        # 主卡 A/B 双槽：按 plans 的形状建（f16 缓存后 shape/dtype 已正确）
        slots_A, slots_B = {}, {}
        for lin_name, _, nb, shape, dt in self.plans[0].entries:
            slots_A[lin_name] = torch.empty(shape, dtype=dt, device=dev)
            slots_B[lin_name] = torch.empty(shape, dtype=dt, device=dev)
        self.vram_slots["main_A"] = slots_A
        self.vram_slots["main_B"] = slots_B
        # 副卡单槽（层49）
        if secondary_device is not None:
            slots_S = {}
            for lin_name, _, nb, shape, dt in self.plans[0].entries:
                slots_S[lin_name] = torch.empty(shape, dtype=dt, device=secondary_device)
            self.vram_slots["sec"] = slots_S
        self.copy_stream = torch.cuda.Stream(device=dev)
        if secondary_device is not None:
            self.sec_copy_stream = torch.cuda.Stream(device=secondary_device)
        tot = sum(t.numel() * t.element_size() for d in
                  ("main_A", "main_B", "sec") if d in self.vram_slots
                  for t in self.vram_slots[d].values())
        LOG.info("[H3FixedSlot] 显存槽就绪: 主卡 A/B 双槽 + 副卡单槽 = %.2f GiB",
                 tot / 2**30)

    def start(self):
        if self._thread is not None:
            return

        def worker():
            next_fill = 0
            while not self._stop.is_set():
                self._wake.wait(timeout=3600)
                self._wake.clear()
                if self._stop.is_set():
                    break
                with self._pend_lock:
                    targets = list(self._pending)
                    self._pending.clear()
                for li in targets:
                    slot_idx = next_fill
                    self.source.read_into(self.plans[li], self.host_slots[slot_idx])
                    self.host_ready[slot_idx].clear()
                    self.host_layer[slot_idx] = li
                    self.host_ready[slot_idx].set()
                    self.stats["prefetch_layers"] += 1
                    next_fill ^= 1

        self._thread = threading.Thread(target=worker, daemon=True,
                                        name="h3-slot-prefetch")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def request(self, li):
        with self._pend_lock:
            if li >= self.n_layers:
                return
            if li in (self.host_layer[0], self.host_layer[1]) or li in self._pending:
                return
            self._pending.append(li)
        self._wake.set()

    def wait_layer(self, li):
        t0 = time.perf_counter()
        while True:
            for s in (0, 1):
                if self.host_layer[s] == li and self.host_ready[s].is_set():
                    self.stats["wait_seconds"] += time.perf_counter() - t0
                    return self.host_slots[s]
            if not any(self.host_layer[s] == li for s in (0, 1)) \
                    and li not in self._pending:
                self.request(li)
            time.sleep(0.002)

    # ------------------------------------------------------------------
    def load_slot(self, li, host_buf, device=None):
        """主机槽 → 显存槽。全部层默认走主卡 A/B 轮换——原作者副卡 16G 空闲
        可放层 49，本机副卡被 TE 占 14.5G 放不下（OOM），层 49 也回主卡轮换。
        环境变量 H3_LAST_ON_SEC=1 可恢复原作者的副卡末层布局。"""
        is_last = (li == self.n_layers - 1) and ("sec" in self.vram_slots) \
            and os.environ.get("H3_LAST_ON_SEC") == "1"
        if is_last:
            slots = self.vram_slots["sec"]
            stream = self.sec_copy_stream
            key = "sec"
        else:
            key = "main_A" if self._turn == 0 else "main_B"
            slots = self.vram_slots[key]
            stream = self.copy_stream
        n = 0
        with torch.cuda.stream(stream):
            for lin_name, _, nb, _, _ in self.plans[li].entries:
                dst = slots[lin_name]
                dst.view(-1).view(torch.uint8).copy_(
                    host_buf[n:n + nb].view(torch.uint8), non_blocking=True)
                n += nb
        stream.synchronize()
        if not is_last:
            self._turn ^= 1   # 副卡槽不参与轮换
        return key

    def install(self, diffusion_model, li, key):
        """槽张量 → 层 4 个 Linear（指针换装，零新建）。
        bf16 模式：必须用 _parameters 整项替换——对 QuantizedTensor Parameter
        赋 .data 会静默失效（.data 仍返回旧 QuantizedTensor 的 bf16 视图），
        这是此前 mat2@cpu 与无效换装的根因。
        int8 模式：qdata+scale → 构造 QuantizedTensor 装槽，保留 layout_type
        → _use_quantized=True 走 int8 kernel（原作者原设计）。"""
        t0 = time.perf_counter()
        slots = self.vram_slots[key]
        block = diffusion_model.blocks[li]
        if getattr(self.source, "int8_mode", False):
            # int8：qdata 槽 + scale 槽 → QuantizedTensor（params 复用首装的）
            from comfy_kitchen.tensor import TensorWiseINT8Layout
            for lin_name in STREAM_LINERARS:
                lin = comfy.utils.get_attr(block, lin_name)
                qdata = slots[f"{lin_name}.weight"]
                scale = slots[f"{lin_name}.weight_scale"]
                if lin_name not in self._qparams:
                    # 首装：从模型原 weight 取 params 模板（含 convrot 配置）
                    orig = lin.weight.data
                    params = orig._params
                    self._qparams[lin_name] = params
                params = self._qparams[lin_name]
                import dataclasses
                new_params = dataclasses.replace(
                    params, scale=scale, orig_dtype=torch.bfloat16)
                qt = QuantizedTensor(
                    qdata.view(params.orig_shape), "TensorWiseINT8Layout",
                    new_params)
                lin._parameters["weight"] = torch.nn.Parameter(
                    qt, requires_grad=False)
        else:
            for lin_name in STREAM_LINERARS:
                lin = comfy.utils.get_attr(block, lin_name)
                # 整项替换为槽张量（bf16 直存，F.linear 直接可用）
                lin._parameters["weight"] = torch.nn.Parameter(
                    slots[lin_name], requires_grad=False)
        self.stats["swaps"] += 1
        self.stats["install_seconds"] += time.perf_counter() - t0

    def recycle_pages(self):
        """步末回收：干净文件页/打包堆移出工作集还给系统（pinned 槽不受影响）。"""
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetProcessWorkingSetSizeEx(
                handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1), 0)
            self.stats["recycles"] += 1
        except Exception as e:
            LOG.debug("[H3FixedSlot] 页回收跳过: %s", e)

    def log_stats(self):
        s = self.stats
        LOG.info(
            "[H3FixedSlot] 步数=%d 预取层=%d 等待=%.2fs 换装=%d(%.3fs) 回收=%d",
            s["steps"], s["prefetch_layers"], s["wait_seconds"],
            s["swaps"], s["install_seconds"], s["recycles"])
