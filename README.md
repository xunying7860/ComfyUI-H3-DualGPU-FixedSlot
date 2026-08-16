# ComfyUI-H3-DualGPU-FixedSlot

MiniMax-H3（Ref2VA / FL2VA）双卡推理节点——单卡算全部层，但权重保留在显存与计算完全异步重叠；副卡承载 TE 编码。

## 思路

与 Ulysses 序列并行（切序列分发多卡）不同，本方案**不切计算**，仅做权重搬运：

```
磁盘 mmap 预读 → 主机 pinned 槽（双缓冲）→ GPU 固定槽 A/B 轮换 → 层计算
         └────────── 后台线程异步 ──────────┘   └─ 与当前层计算重叠 ─┘
```

- 全局仅 2 个固定 GPU 槽（A/B 轮换）+ 主机双缓冲，显存占用恒定
- 依赖 PCIe 即可，不需要 NVLink
- 不改变任何算子/数值路径，与基线逐层 hash 对齐（bf16 模式 l0 差 ~7.6e-05 ≈ 1ulp）

## 实测（RTX 5070 Ti 16G 主卡 + RTX 3080 20G 副卡，0.6M×15s 4 步）

| 配置 | 速度 | 说明 |
|---|---|---|
| 双卡固定槽（5070Ti 主卡） | **41.0 s/it** | 单卡 5070Ti 的 143% |
| 双卡固定槽（3080 主卡） | 80.5 s/it | 单卡 3080 的 114% |
| 单卡 5070Ti 原版 | 58.6 s/it | 基准 |
| 单卡 3080 原版 | 91.7 s/it | 基准 |

## 双模式

| 模式 | 开关 | 说明 |
|---|---|---|
| **bf16 烘焙直存**（默认） | 无 | 权重烘焙为 bf16 直存直算 |
| **int8 槽** | `H3_SLOT_INT8=1` | int8 量化 kernel 直算 |

其他环境变量：
- `H3_LAST_ON_SEC=1`：末层放副卡
- `CUDA_VISIBLE_DEVICES=1,0`：切换主卡

## 启动命令示例

```bash
# 默认布局：5070Ti 主卡 + 3080 副卡 TE（int8/last_on_sec 在工作流节点 UI 里勾选）
cd /h/ComfyUI-v4-5070ti
./python_embeded/python.exe -s ComfyUI/main.py --port 5071   --use-sage-attention --disable-dynamic-vram

# 3080 当主卡（大 token 场景：20G 显存可跑 0.7M×22s+）：
# 交换 CUDA 枚举顺序 → cuda:0=3080 主卡、cuda:1=5070Ti 副卡 TE
# 注意：此时两个节点的 secondary_gpu 都应填 0
CUDA_VISIBLE_DEVICES=1,0 ./python_embeded/python.exe -s ComfyUI/main.py   --port 5071 --use-sage-attention --disable-dynamic-vram

# 命令行覆盖（节点参数未勾选时生效；勾选了以节点为准）：
H3_SLOT_INT8=1 H3_LAST_ON_SEC=1 ./python_embeded/python.exe -s ComfyUI/main.py   --port 5071 --use-sage-attention --disable-dynamic-vram
```

## 节点

- `H3DualGPUPipeline` —— 主卡层 0-48 固定槽流水 + 层 49 主卡 B 槽（`H3_LAST_ON_SEC=1` 可切副卡）
- `H3TESecondaryGPU` —— TE 加载到副卡 + 编码完成自动卸载

## 依赖

- ComfyUI（带 comfy_kitchen）
- sageattention（QK int8 / V fp16 CUDA 核）
- ComfyUI-GGUF（若 TE 用 GGUF 量化）

## 兼容性

- 权重格式：**int8_convrot**（Ref2VA / FL2VA 的 `*_pruned_int8_convrot.safetensors` 官方版）
- FL2VA 需用官方 int8_convrot 版（w4a8_mixed codebook 格式暂不兼容）
