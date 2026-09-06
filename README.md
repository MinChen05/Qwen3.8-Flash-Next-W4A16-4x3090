# Qwen3.8-Flash-Next W4A16 全 GPU 部署实战（4×RTX 3090 / sm86）

> **结论先行**：180B-A6B MoE（512→296 专家裁剪版 W4A16）通过 SGLang 定制 fork 在 4×RTX 3090（96GB，无 NVLink）上实现**全 GPU 推理**：
> 短请求解码 **56.2 tok/s**（追平 27B 稠密模型生产基线）、95K 上下文解码 **40.9 tok/s**、95K prefill **3934 tok/s**（offload 方案的 6 倍）、**262K 上下文 + 视觉**全部达成。
>
> 官方与社区的所有部署资料均只覆盖 sm89+/Blackwell 单卡场景；本文记录将这套方案移植到消费级 Ampere 多卡（sm86/TP4）过程中踩过的全部坑与修复补丁。

**可视化分析页**：[docs/benchmark-analysis.html](docs/benchmark-analysis.html)（浏览器直接打开）

---

## 1. 背景

- 主力模型 Qwen3.8-Flash-Next（180B 总参 / 6B 激活 MoE，Qwen4 架构：GDN 线性注意力 + QSA 稀疏注意力 + PLE 预测头）此前在本机只有两条路：
  - **llama.cpp/ik offload**：UD-Q4_K_XL GGUF，MoE 专家落 CPU——短请求 36-40 tok/s，长上下文衰减到 18-24，**投机解码全路线不可用**；
  - **ik 纯 GPU 128K**：56.9 tok/s 但**无视觉、无并发、128K 上限**。
- 切换 27B 稠密模型（vLLM W8A8+MTP，55 t/s）解决了速度，但放弃了 180B 的推理深度（埋 bug 测试 4:3 胜出）。
- 本方案的组合拳首次做到"**180B 深度 + 27B 级速度 + 262K + 视觉**"四项同时成立。

## 2. 方案组成（三件套）

| 组件 | 仓库 | 作用 |
|---|---|---|
| 模型 | HF `ranxianglei/Qwen3.8-Flash-Next-W4A16-Modular` | Intel AutoRound W4A16 量化 + 按真实 agent 流量画像裁专家 512→296/层，modular 重打包（14 backbone 分片 61.9G + 48 逐层专家文件 36.3G）。296 = daily-294 ∪ 异常画像 2，支持自愈 |
| 推理引擎 | GitHub `ranxianglei/sglang`（分支 `ours/main`，基线 SGLang 0.5.6） | 4 补丁：① Marlin repack GC（修 W4A16 加载 OOM）② mixed-chunk × QSA 崩溃修复 ③ **运行时专家 keep-mask / keep-only offload**（`SGLANG_EXPERT_KEEP_MASK`/`SGLANG_EXPERT_KEEP_OFFLOAD` 两个环境变量换裁剪集，无需重下模型）④ 冷专家动态加载（WIP） |
| 画像工具 | GitHub `ranxianglei/sglang-expert-profile` | 30 分钟用自己流量产出逐层 keep 集 JSON；自带社区 keep 集 daily-294 / daily-heal-296 |

官方上游：SGLang PR #36497（Qwen3.8-Flash-Next 支持，合入前 fork 已回移）；模型另有物理切片版 `W4A16-Pruned-294E`。

## 3. 硬件与环境要求

| 项 | 要求 | 本机实测 |
|---|---|---|
| GPU | ≥64G 显存（作者：单卡 96G）；**本机 4×24G TP4 亦可行** | 4×RTX 3090，权重 ~13.5G/卡 |
| CPU RAM | ≥64G 空闲（PLE ngram 表 pinned） | 实际占用 ~95G（4 rank × ~24G bf16） |
| 驱动 | 支持 CUDA 13（≥580） | 580.159.03 ✓（系统 toolkit 12.6 不碍事，wheel 自带 runtime） |
| 磁盘 | ~100G | 74 文件全量 98.3GB |
| Python | 3.12 | venv 独立安装 |

## 4. 模型下载与完整性校验

- 总量 **98.3GB / 74 文件**：`backbone-0000X.safetensors`×14（61.9G，含 FP16 视觉塔、MTP 层）+ `experts-L00..L47.safetensors`×48（36.3G，**文件名零填充**）+ 配置文件。
- **必须全量 sha256 校验**：对照 `?blobs=true` API 返回的 `lfs.sha256`。实测曾出现 **experts-L14 大小精确但 sha256 不符**的"混合续传损坏"（curl -C - 续传遇上 CDN 返回 200 全量覆盖）——只看大小必踩雷。修复 = 删除重下（不带续传头）。
- 国内通道：该仓库无 ModelScope 镜像（已排查），hf-mirror 已迁境外不可达；实测可行 = 海外代理多连接（单连接 ~2MB/s、4 并发 ~5MB/s）或本地自量化。`hf_transfer` 不走代理、python hub 过此类代理 SSL 断，**curl -C - + 重试循环 + 多并发**是最稳形态。
- **视觉能力确认**：`model.safetensors.index.json` 含 333 个 `model.visual.*` 张量（Qwen3-VL 风格 + deepstack），config `language_model_only=false`、量化配置显式 `.*visual.*` bits=16——视觉塔以 FP16 全精度随包。

## 5. 安装

```bash
python3 -m venv venv && venv/bin/pip install -U pip
# torch 2.13.0 (cu13) —— fork pyproject 钉死版本
venv/bin/pip install torch==2.13.0
# 跳过 Rust 扩展（四个关键补丁均为纯 Python；无 cargo 环境时的官方开关）
export SGLANG_BUILD_RUST_EXTS=none
cd sglang-fork/python && ../venv/bin/pip install -e ".[srt]"
# 另需编译经典 flash-attn FA2（见 §6.5），FA4 cute 仅支持 Blackwell
```

## 6. sm86 / 4×3090 移植：六连崩与修复（核心章节）

> 以下补丁均基于 `ours/main` 的 `python/sglang/srt/models/qwen4_exp.py` 与
> `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py`。原文件务必备份。

### 6.1 CRASH 1：PLE 查表内核 `tl.float8e4nv` 在 sm86 无法编译

**现象**：CUDA graph 捕获期 Triton 报 `type fp8e4nv not supported in this architecture`。
**根因**：PLE ngram 表在 checkpoint 中为 `float8_e4m3fn`（无 per-tensor scale，scale=1 语义），pinned 查表内核 `qwen4_exp.py::_gather_ple_embedding_from_pinned_kernel` 的 `is_fp8` 分支把权重指针 cast 成 `tl.float8e4nv`——该类型需要硬件 FP8（sm89+）。

**修复（qwen4_exp.py，pinned 类构造函数）**：GPU 侧保持 fp8（塞得下），搬入 pinned 类时在 **CPU 分块反量化为 bf16**，使内核走自带 bf16 分支：

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
1. 直接把 `params_dtype` 改 bf16 → VocabParallelEmbedding 在 **GPU** 上分配 23.84G/卡 → OOM（PLE 表必须保持 fp8 走 GPU 临时承载）；
2. 4 个 TP rank 同时整表 `.to(bf16)` → 瞬时 ~144GB RAM → **系统 OOM killer**；必须分块（2M 行/块，峰值仅一个 chunk）；
3. `torch.empty(..., pin_memory=True)` 在 CUDA 设备上下文里会被劫持 → 报 "Only dense CPU tensors can be pinned"，必须显式 `device="cpu"`。

代价：PLE 表 bf16 后 RAM 占用翻倍（~95G），251G 内存无压力；查表输出本来就是 bf16，**精度无损**。

### 6.2 CRASH 2：加载器护栏硬失败

**现象**：`ValueError: fp8 PLE auto-switch is unsupported with ple_offload_embedding; set text_config.ple_embedding_dtype="float8_e4m3fn" instead`。
**根因**：权重加载器发现 checkpoint 分片为 fp8 而目标存储为非 fp8 时，对 pinned 场景直接 raise（防止 pageable 交换容错）。
**修复**：给该条件加一个豁免——pinned bf16 表接收 fp8 分片时走 `copy_ple_rows_to_tp_embedding` 自带的 `.to(dtype)` 转换（同样精确）：

```python
if (
    loaded_weight.dtype == torch.float8_e4m3fn
    and emb.weight.dtype != torch.float8_e4m3fn
    and not isinstance(emb, Qwen4ExpPinnedHostEmbedding)   # 新增豁免
):
```

### 6.3 CRASH 3：QSA 解码落到 FA4 cute（Blackwell 专用）

**现象**：图捕获期 `nvidia_cutlass_dsl ... MLIRError: unable to compute crd2idx`。
**根因**：`qwen_sparse_attn_backend.py::_resolve_flash_attn_varlen_func()` 优先经典 flash-attn (FA2, Ampere/Hopper)，缺失时落到 `flash_attn.cute.interface`（FA4，**仅 Blackwell**）。venv 里只装了 flash-attn-4。
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

要点：**全程避免 host 同步**（`row_lens.max().item()` 之类会炸 CUDA graph 捕获），maxlen 直接用常量 `topk`；SDPA 的 `enable_gqa` 处理 24 头 Q / 8 头 KV 的 GQA。QSA 解码算力极小（每请求仅 topk 个 KV），SDPA 无性能压力。

### 6.4 经典 FA2 的获取

FA2 2.8.3.post1 官方已有 `cu13torch2.13cxx11abiTRUE-cp312` 预编译 wheel（setup.py 会"猜测"该 URL），但 GitHub release 资产被常见 gh 代理一律拒绝。两条路：

```bash
# 路 A：有海外网络时直接抓官方 wheel
pip install flash_attn-2.8.3.post1+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 路 B：源码编译（对齐工具链版本后可行）
export FLASH_ATTENTION_FORCE_BUILD=TRUE        # 注意是 TRUE，不是 1
export CUDA_HOME=<python 包 nvidia/cu13 的目录>  # nvcc 与 cccl/cuda-toolkit 版本必须一致
export MAX_JOBS=48 TORCH_CUDA_ARCH_LIST="8.6"
pip install -v --no-build-isolation --no-deps /path/to/flash_attn-2.8.3.post1.tar.gz
```

坑：pip 环境 `nvidia-cuda-nvcc`(13.3) 与 `cuda-toolkit`(13.0) 版本错位会触发 cccl 头的 `#error` 版本守卫——**对齐到 13.3.1**（`pip install -U cuda-toolkit==13.3.1`）；`FLASH_ATTENTION_FORCE_BUILD` 的值必须是字符串 `TRUE`。

### 6.5 启动期杂项

| 问题 | 修复 |
|---|---|
| 图捕获期 `FileNotFoundError: 'ninja'` | 启动脚本 `export PATH=venv/bin:$PATH` |
| tilelang JIT 头冲突（pip nvcc 13.3 vs 自带 cccl） | CUDA_HOME 指向对齐后的工具链 |
| `RuntimeError: The memory capacity is unbalanced`（常驻 reranker 占 GPU1 context 956M 导致天然不对称） | `export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`（降级为警告） |
| 图捕获 OOM（0.85 水位差 386M，恰为 GPU1 上 reranker 多占的量） | `--mem-fraction-static 0.80` |

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

- `--tp 4` 与 0.80 水位：作者场景是单卡 96G（0.93）；4×24G 下多卡分摊权重后水位由图捕获余量决定，reranker 等常驻 context 越多越要降。
- **思考模式默认开启**（`reasoning_config default_enabled=True`），直出答案需请求级 `"chat_template_kwargs": {"enable_thinking": false}`。
- 首个请求 6.6 tok/s 是 QSA/GDN Triton JIT 冷启动，第二个请求起进入稳态。

## 8. 基准数据（4×3090 实测）

| 配置 | 短请求解码¹ | 95K 解码² | 95K prefill³ | 视觉 | 上下文 |
|---|---|---|---|---|---|
| 27B 稠密生产（vLLM W8A8+MTP2） | 55.3 | 未复测（历史 ~50） | ~2800（历史） | ✓ | 256K |
| Flash-Next ik offload（原形态） | 35.6-38.4 | 30.5 | 657 | ✓ | 200-256K |
| Flash-Next ik 纯 GPU（无视觉） | 53.2-55.0 | 41.3 | — | ✗ | 128K |
| **Flash-Next W4A16 sglang（本方案）** | **55.9 / 56.4 / 56.3** | **40.9** | **3934** | **✓** | **262K** |

¹ 500 tok 中文生成 ×3 均值，同 prompt 同采样参数。² 95212 tok 填充后前缀命中解码。³ 填充阶段吞吐。
视觉实测：真实图片（二维码+文字）描述准确，703 prompt tokens。长上下文衰减曲线明显比 offload 形态平缓（QSA + 全 GPU 带宽）。

## 9. 已知限制与未验证项

- QSA 解码 SDPA 兜底为自写路径（非官方内核）；FA2 可用时应优先走 FA2。
- 思考模式默认开；SGLang 侧无思考预算参数，靠请求级开关或 max_tokens 约束。
- `mem-fraction-static 0.80` 偏保守（给常驻 context 让位），KV 池还有上调空间。
- 未验证：262K 满长度压测、keep-mask 运行时裁剪（296→294+自愈集可再省显存/提命中率）、多并发吞吐曲线、长稳。
- fork 基线较旧（SGLang 0.5.6 + 回移的 #36497），上游合入后建议跟随升级。
- 模型卡声称"weights ~45GB"实为显存占用口径（PLE 表 offload 到 RAM 后）；磁盘全量 98.3GB。

## 10. 生产切换决策（待拍板）

- **A 转正观察**：与 27B 并行常驻数日，观察稳定性/显存水位/JIT 首请延迟。
- **B 立即转正**：18080 切 sglang，27B 保留秒级回退；客户端别名兼容方案已有先例。
- ik offload 形态（补丁后 38 t/s 级）已全面落后，建议退役为应急备份。

## 附录

- 可视化分析页：[docs/benchmark-analysis.html](docs/benchmark-analysis.html)
- 相关链接：[HF 模型](https://huggingface.co/ranxianglei/Qwen3.8-Flash-Next-W4A16-Modular) · [sglang fork](https://github.com/ranxianglei/sglang/tree/ours/main) · [expert-profile](https://github.com/ranxianglei/sglang-expert-profile) · [上游 PR #36497](https://github.com/sgl-project/sglang/pull/36497)
- 补丁备份：`qwen4_exp.py.bak-fp8sm86`；启动脚本与测试脚本均留档于部署机 `/tmp`（建议自行归档）
- 环境内网地址、凭据等敏感信息一律未收录本文档。
