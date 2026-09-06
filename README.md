# Qwen3.8-Flash-Next W4A16 全 GPU 部署指南（4×RTX 3090 / sm86 多卡）

> **适用人群**：想在消费级 Ampere 多卡（如 4×RTX 3090，sm86）上跑 Qwen3.8-Flash-Next 180B 的同学。
> 目前所有公开部署资料只覆盖 sm89+/Blackwell **单卡**场景，本文整理了在 sm86 **TP4 多卡**环境部署的完整流程、
> 6 个必然踩到的坑与对应补丁，以及实测基准。
>
> **成果**：180B-A6B MoE（296/512 专家裁剪版）全 GPU 推理——短请求 56.2 tok/s、95K 解码 40.9 tok/s、
> 95K prefill 3934 tok/s、262K 上下文、视觉可用。

**可视化分析页**：[docs/benchmark-analysis.html](docs/benchmark-analysis.html)

---

## 1. 这套方案能带来什么

Qwen3.8-Flash-Next（180B 总参 / 6B 激活 MoE，Qwen4 架构：GDN 线性注意力 + QSA 稀疏注意力 + PLE 预测头）在消费级多卡上此前只有两条路，各有硬伤：

- **llama.cpp/ik offload**（UD-Q4_K_XL GGUF，MoE 专家落 CPU）：短请求 36-40 tok/s，长上下文衰减到 18-24，投机解码全路线不可用；
- **ik 纯 GPU 128K**：56.9 tok/s 但无视觉、无并发、128K 上限。

W4A16 全 GPU 路线（SGLang 引擎 + 专家裁剪版模型）在 4×3090 上实测可以同时拿到：

| 能力 | 本方案 | 备注 |
|---|---|---|
| 180B 推理深度 | ✓ | 296/512 专家，质量门禁见模型仓库 |
| 27B 级速度 | ✓ 56.2 tok/s | 与 27B 稠密 + MTP 的生产基线持平 |
| 262K 上下文 | ✓ | 原生上限，QSA 稀疏注意力 |
| 视觉 | ✓ | 视觉塔 FP16 随包，实测识别准确 |

## 2. 前置资源

部署需要三样东西（均来自 ranxianglei 的开源，感谢发布）：

| 资源 | 来源 | 说明 |
|---|---|---|
| 模型 checkpoint | HF [ranxianglei/Qwen3.8-Flash-Next-W4A16-Modular](https://huggingface.co/ranxianglei/Qwen3.8-Flash-Next-W4A16-Modular) | Intel AutoRound W4A16 量化 + 按真实 agent 流量画像裁专家 512→296/层，modular 重打包（14 backbone 分片 61.9G + 48 逐层专家文件 36.3G） |
| 推理引擎源码 | GitHub [ranxianglei/sglang](https://github.com/ranxianglei/sglang/tree/ours/main)（分支 `ours/main`，基线 SGLang 0.5.6） | 内置四个关键补丁：Marlin repack GC（修 W4A16 加载 OOM）、mixed-chunk × QSA 崩溃修复、运行时专家 keep-mask / keep-only offload（`SGLANG_EXPERT_KEEP_MASK`/`SGLANG_EXPERT_KEEP_OFFLOAD` 环境变量换裁剪集）、冷专家动态加载（WIP） |
| 画像工具（可选） | GitHub [ranxianglei/sglang-expert-profile](https://github.com/ranxianglei/sglang-expert-profile) | 用自己的流量产出逐层专家 keep 集 JSON（约 30 分钟），自带社区 keep 集 daily-294 / daily-heal-296 |

相关：上游 PR [sgl-project/sglang#36497](https://github.com/sgl-project/sglang/pull/36497)（合入前 fork 已回移）；模型另有物理切片版 `W4A16-Pruned-294E`。

## 3. 硬件与环境要求

| 项 | 要求 | 实测环境 |
|---|---|---|
| GPU | ≥64G 显存（官方场景为单卡 96G）；**4×24G TP4 已验证可行**（本文） | 4×RTX 3090，权重 ~13.5G/卡 |
| CPU RAM | ≥64G 空闲（PLE ngram 表 pinned 驻留） | 实际占用 ~95G（4 rank × ~24G bf16） |
| 驱动 | 支持 CUDA 13（≥580） | 580.159.03；系统 toolkit 12.6 不影响（wheel 自带 runtime） |
| 磁盘 | ~100G | 74 文件全量 98.3GB |
| Python | 3.12 | 建议 venv 独立环境 |

## 4. 模型下载与完整性校验

- 总量 **98.3GB / 74 文件**：`backbone-0000X.safetensors`×14（61.9G，含 FP16 视觉塔、MTP 层）+ `experts-L00..L47.safetensors`×48（36.3G，**文件名零填充**）+ 配置文件。
- **务必全量 sha256 校验**：对照 `?blobs=true` API 返回的 `lfs.sha256`。实践中出现过 **experts-L14 大小精确但 sha256 不符**的"混合续传损坏"（断点续传遇上 CDN 返回 200 全量覆盖）——只对比文件大小发现不了。修复 = 删除后全新下载（不带续传头）。
- 视觉能力确认方法：`model.safetensors.index.json` 应含 333 个 `model.visual.*` 张量，config 中 `language_model_only=false`、量化配置显式 `.*visual.*` bits=16（视觉塔 FP16 随包）。
- 下载形态建议：`hf_transfer` 不走代理、python hub 部分代理下 SSL 断——**`curl -C -` 断点续传 + 重试循环 + 多并发**是最稳形态；下载完成后用 tar/rsync 等追加式同步时注意同样适用上述校验。

## 5. 环境安装

```bash
python3 -m venv venv && venv/bin/pip install -U pip
# torch 2.13.0 (cu13) —— fork pyproject 钉死版本
venv/bin/pip install torch==2.13.0
# 跳过 Rust 扩展（sm86 必需补丁均为纯 Python；无 cargo 环境时的官方开关）
export SGLANG_BUILD_RUST_EXTS=none
cd sglang-fork/python && ../path/to/venv/bin/pip install -e ".[srt]"
# 另需 flash-attn（见 §6.3），FA4 cute 仅支持 Blackwell
```

## 6. sm86 / 多卡已知坑与修复（核心章节）

> 以下问题在 sm89+/Blackwell 上**不会出现**，是 sm86（无硬件 FP8、无 FA4）+ TP4 多卡的特有坑。
> 补丁基于 fork 的 `python/sglang/srt/models/qwen4_exp.py` 与
> `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`，动手前先备份原文件。

### 6.1 PLE 查表内核 `tl.float8e4nv` 在 sm86 无法编译

**现象**：CUDA graph 捕获期 Triton 报 `type fp8e4nv not supported in this architecture`。
**原因**：PLE ngram 表在 checkpoint 中为 `float8_e4m3fn`（无 per-tensor scale，scale=1 语义），pinned 查表内核 `_gather_ple_embedding_from_pinned_kernel` 的 `is_fp8` 分支把权重指针 cast 成 `tl.float8e4nv`——该类型需要硬件 FP8（sm89+）。

**修复（qwen4_exp.py，pinned 类构造函数）**：GPU 侧保持 fp8（塞得下），进 pinned 类时在 **CPU 分块反量化为 bf16**，使内核走自带 bf16 分支：

```python
source_weight = embedding.weight
if source_weight.dtype == torch.float8_e4m3fn:
    # sm86: fp8e4nv triton type needs sm89+; dequant to bf16 on CPU.
    # checkpoint has no per-tensor scale -> cast is exact.
    n_rows = source_weight.shape[0]
    bf = torch.empty(source_weight.shape, dtype=torch.bfloat16,
                     device="cpu", pin_memory=True)   # 必须显式 device="cpu"
    CH = 1 << 21
    for i0 in range(0, n_rows, CH):
        i1 = min(i0 + CH, n_rows)
        bf[i0:i1] = source_weight[i0:i1].detach().to("cpu").to(torch.bfloat16)
    cpu_weight = nn.Parameter(bf, requires_grad=False)
else:
    ...原逻辑...
```

三个子坑，缺一不可：
1. 直接把 `params_dtype` 改 bf16 → `VocabParallelEmbedding` 在 **GPU** 上分配 23.84G/卡 → OOM（表必须保持 fp8 由 GPU 临时承载）；
2. 4 个 TP rank 同时整表 `.to(bf16)` → 瞬时 ~144GB RAM → **系统 OOM killer**；必须分块（2M 行/块，峰值仅一个 chunk）；
3. `torch.empty(..., pin_memory=True)` 在 CUDA 设备上下文里会被劫持 → 报 "Only dense CPU tensors can be pinned"，必须显式 `device="cpu"`。

代价：PLE 表 bf16 后 RAM 占用翻倍（~95G）；查表输出本来就是 bf16，**精度无损**。

### 6.2 权重加载器护栏硬失败

**现象**：`ValueError: fp8 PLE auto-switch is unsupported with ple_offload_embedding; set text_config.ple_embedding_dtype="float8_e4m3fn" instead`。
**原因**：权重加载器发现 checkpoint 分片为 fp8 而目标存储为非 fp8 时，对 pinned 场景直接 raise（防止 pageable 交换容错）。
**修复**：给该条件加一个豁免——pinned bf16 表接收 fp8 分片时走 `copy_ple_rows_to_tp_embedding` 自带的 `.to(dtype)` 转换（同样精确）：

```python
if (
    loaded_weight.dtype == torch.float8_e4m3fn
    and emb.weight.dtype != torch.float8_e4m3fn
    and not isinstance(emb, Qwen4ExpPinnedHostEmbedding)   # 新增豁免
):
```

### 6.3 QSA 解码落到 FA4 cute（Blackwell 专用）

**现象**：图捕获期 `nvidia_cutlass_dsl ... MLIRError: unable to compute crd2idx`。
**原因**：`qwen_sparse_attn_backend.py::_resolve_flash_attn_varlen_func()` 优先经典 flash-attn (FA2, Ampere/Hopper)，缺失时落到 `flash_attn.cute.interface`（FA4，**仅 Blackwell**）。若 venv 只装了 flash-attn-4 就会踩到。
**修复**：两处——

```python
# a) resolver：计算能力 < 9 直接返回 None（调用方走兜底）
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        return None

# b) _forward_paged_attention 调用点：fa2 为 None 时用 SDPA 兜底
fa2 = _resolve_flash_attn_varlen_func()
if fa2 is not None:
    ...原 flash_attn_varlen_func 调用...
    return output.reshape(q.shape[0], -1)
# sm86 fallback: QSA 解码每请求仅 topk(~2048) 个 KV token，SDPA 足够
batch = cu_seqlens_k.numel() - 1
row_lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.long)
idx = torch.arange(topk, device=q.device, dtype=torch.long).unsqueeze(0) \
      + cu_seqlens_k[:-1].to(torch.long).unsqueeze(1)
valid = torch.arange(topk, device=q.device).unsqueeze(0) < row_lens.unsqueeze(1)
k_pad = packed_k[idx.clamp(max=packed_k.shape[0]-1)].transpose(1, 2)
v_pad = packed_v[idx.clamp(max=packed_k.shape[0]-1)].transpose(1, 2)
q_pad = q.view(batch, 1, q.shape[1], q.shape[2]).transpose(1, 2)
out = torch.nn.functional.scaled_dot_product_attention(
    q_pad, k_pad, v_pad,
    attn_mask=valid.view(batch, 1, 1, topk),
    scale=layer.scaling,
    enable_gqa=(q.shape[1] != packed_k.shape[1]),
)
return out.squeeze(2).reshape(q.shape[0], -1)
```

要点：**全程避免 host 同步**（`row_lens.max().item()` 之类会炸 CUDA graph 捕获），maxlen 直接用常量 `topk`；SDPA 的 `enable_gqa` 处理 Q/KV 头数不等的 GQA。QSA 解码算力极小（每请求仅 topk 个 KV），SDPA 无性能压力。

### 6.4 经典 FA2（FA2 优先于 FA4 生效）

FA2 2.8.3.post1 官方已有 `cu13torch2.13cxx11abiTRUE-cp312` 预编译 wheel（setup.py 会自动"猜测"该 URL），但 GitHub release 资产常被网络环境/代理拒绝。两条路：

```bash
# 路 A：有海外网络时直接抓官方 wheel
pip install flash_attn-2.8.3.post1+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 路 B：源码编译（对齐工具链版本后可行）
export FLASH_ATTENTION_FORCE_BUILD=TRUE        # 注意是字符串 TRUE，不是 1
export CUDA_HOME=<python 包 nvidia/cu13 的目录>  # nvcc 与 cccl/cuda-toolkit 版本必须一致
export MAX_JOBS=48 TORCH_CUDA_ARCH_LIST="8.6"
pip install -v --no-build-isolation --no-deps /path/to/flash_attn-2.8.3.post1.tar.gz
```

坑：pip 环境 `nvidia-cuda-nvcc`(13.3) 与 `cuda-toolkit`(13.0) 版本错位会触发 cccl 头的 `#error` 版本守卫——**对齐到 13.3.1**（`pip install -U cuda-toolkit==13.3.1`）。装好 FA2 后 §6.3 的 SDPA 兜底自动退位（resolver 优先 FA2）。

### 6.5 启动期杂项

| 问题 | 修复 |
|---|---|
| 图捕获期 `FileNotFoundError: 'ninja'` | 启动脚本 `export PATH=venv/bin:$PATH` |
| tilelang JIT 头冲突（pip nvcc 13.3 vs 自带 cccl） | CUDA_HOME 指向对齐后的工具链 |
| `RuntimeError: The memory capacity is unbalanced`（常驻服务占某卡 context 导致不对称） | `export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`（降级为警告） |
| 图捕获 OOM（实测 0.85 水位差 386M，恰为另一常驻服务多占的量） | `--mem-fraction-static 0.80` |

## 7. 启动命令（4×3090 实测可用）

```bash
export SGLANG_BUILD_RUST_EXTS=none
export PATH=/path/to/venv/bin:$PATH
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
python -m sglang.launch_server \
  --model-path /path/to/Qwen3.8-Flash-Next-W4A16-Modular \
  --served-model-name Qwen3.8-Flash-Next-W4A16 \
  --tp 4 \
  --ple-offload-embedding \
  --moe-a2a-backend none \
  --linear-attn-prefill-backend triton \
  --linear-attn-decode-backend triton \
  --mamba-ssm-dtype bfloat16 \
  --context-length 262144 \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 18089
```

- `--tp 4` 与 0.80 水位：官方场景是单卡 96G（0.93）；4×24G 下水位由图捕获余量决定，卡上有其他常驻 context 越多越要降。
- **思考模式默认开启**（`reasoning_config default_enabled=True`），直出答案需请求级 `"chat_template_kwargs": {"enable_thinking": false}`。
- 首个请求 ~6.6 tok/s 是 QSA/GDN Triton JIT 冷启动，第二个请求起进入稳态。

## 8. 实测基准（4×3090）

| 配置 | 短请求解码¹ | 95K 解码² | 95K prefill³ | 视觉 | 上下文 |
|---|---|---|---|---|---|
| 27B 稠密（vLLM W8A8+MTP2，同机生产） | 55.3 | 未复测（历史 ~50） | ~2800（历史） | ✓ | 256K |
| Flash-Next ik offload（llama.cpp 形态） | 35.6-38.4 | 30.5 | 657 | ✓ | 200-256K |
| Flash-Next ik 纯 GPU（无视觉） | 53.2-55.0 | 41.3 | — | ✗ | 128K |
| **Flash-Next W4A16 sglang（本指南）** | **55.9 / 56.4 / 56.3** | **40.9** | **3934** | **✓** | **262K** |

¹ 500 tok 中文生成 ×3 均值，同 prompt 同采样参数。² 95212 tok 填充后前缀命中解码。³ 填充阶段吞吐。
视觉实测：真实图片（二维码+文字）描述准确，703 prompt tokens。长上下文衰减曲线明显比 offload 形态平缓（QSA + 全 GPU 带宽）。

## 9. 已知限制与未验证项

- QSA 解码 SDPA 兜底为自定义路径（非官方内核）；装好 FA2 后应优先走 FA2。
- 思考模式默认开；引擎侧无思考预算参数，靠请求级开关或 max_tokens 约束。
- `mem-fraction-static 0.80` 偏保守（给常驻 context 让位），KV 池还有上调空间。
- 未验证：262K 满长度压测、keep-mask 运行时裁剪（296→294+自愈集可再省显存）、多并发吞吐曲线、长稳。
- fork 基线较旧（SGLang 0.5.6 + 回移的 #36497），上游合入后建议跟随升级。
- 模型卡声称"weights ~45GB"实为显存占用口径（PLE 表 offload 到 RAM 后）；磁盘全量 98.3GB。

## 10. 参考

- 模型：[ranxianglei/Qwen3.8-Flash-Next-W4A16-Modular](https://huggingface.co/ranxianglei/Qwen3.8-Flash-Next-W4A16-Modular)（另有物理切片版 `W4A16-Pruned-294E`）
- 引擎：[ranxianglei/sglang `ours/main`](https://github.com/ranxianglei/sglang/tree/ours/main) · [expert-profile 工具](https://github.com/ranxianglei/sglang-expert-profile)
- 上游：[sgl-project/sglang PR #36497](https://github.com/sgl-project/sglang/pull/36497)
- 可视化分析页：[docs/benchmark-analysis.html](docs/benchmark-analysis.html)
