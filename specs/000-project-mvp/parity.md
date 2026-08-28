# Parity matrix (统一对拍矩阵)

> 2026-08-25 建立（SDD 评审 B-L1/M2/M3、C-低、B-L5 采纳统一）。所有 spec 引用的模型/prompt/门禁阈值以此为准。
> 约定：某行 sha256/golden 尚未生成时以 `TODO(generate)` 占位，生成后必须随 golden 文件一起提交（PR 审查）。
> 2026-08-28 互注（specs/014 r2 评审唯一化）：若本文件各行 golden 生成前一律以 013 端到端 sha 锚点为准（014 spec 模型行已钉）；**金块数据 = `tests/golden/`（q8_0/tokenizer 金块 JSON）；生成器 = `scripts/golden/`（gen_q8_0_golden.sh 已在仓）；`bench/golden/gen_tokens.sh` 仅属 tokenizer 行**——三处不混（014 plan Module Breakdown 同注）。

## 模型与权重

| 行 | 模型（GGUF） | quant | tokenizer | sha256 (TODO 后固化) | 用途 |
|---|---|---|---|---|---|
| M1 | Llama-3-8B（或等 vLLM 库名：Meta-Llama-3-8B） | Q8_0 | BPE | TODO(generate) | 003 文本对齐、005 吞吐基准 |
| M2 | 同 M1 | F16 | BPE | TODO(generate) | 003 三层门禁 F16 档 |
| M3 | Qwen2.5-1.5B-Instruct | Q8_0 | BPE(Qwen) | TODO(generate) | 004 tokenizer 对拍 |
| M4 | Llama-2-7B | Q8_0 | SPM | TODO(generate) | 004 SPM 档门禁（一级，非 best-effort） |

## Prompts 集

- `bench/prompts/{pr1..pr20}.txt`：20 条固定 prompt（两条目：中文/英文/长文 4K/代码/JSON 等，**维护者入库，随 golden 提交**）；loadgen 一致性套件（101 条）另行定义在 008-ci-infra（引用本文件路径，不在此列）。
- 生成/重建脚本：`bench/golden/gen_tokens.sh`（调用 `llama-tokenize --ids`，锚定 commit/flag；见 004 spec AC）。

## Referee 参数锁定（llama.cpp）

| 项 | 值 |
|---|---|
| commit | f280b2698（build b10615） |
| 构建 | **r2 修订（014 T0）**：CPU 档（`-DCMAKE_BUILD_TYPE=Release`；对拍协议全 CPU 工具——llama-bench `-ngl 0`/llama-tokenize/quantize/cli；CUDA 构建不用且 nvcc 12.6 判定机不可建 ARCH=120）；记录 CPU 型号/核数 |
| F16 对比 | 双方 compute type 一致：llama.cpp 侧 `GGML_CUDA_CUBLAS_COMPUTE_TYPE=16F` |
| 结论基准 | 见 specs/006 基准协议（llama-bench 参数、KV dtype=f16、graph on） |

## 门禁阈值（按 spec）

| 目标 | 阈值 | 性质 |
|---|---|---|
| 004 tokenizer | 编码/解码逐 token 100%（BPE+SPM） | 硬门禁 |
| 003 F16 | 同 compute type 时 100%；否则回退累积 drift ≤1e-4 | 硬门禁（回退已声明） |
| 003 Q8_0 | greedy 一致率 ≥99.9% + logits 相对漂移 ≤1e-2 | 记录项（恢复 100% 需 mmq RFC） |
| 003 CPU 档 | decode ≥3× llama.cpp CPU | 门禁（gpu-runner 判定） |
| 006 | sm100 ≥0.85× / sm90 ≥0.85×（T6 后）或回退档（详见 006 spec） | 门禁 arch 分档 |
| 005 | 确定性 2×bit-identical（硬）；批 vs 单 logits ≤1e-3 + token ≥99.9%（软，记录） | 硬+软 |

> 修订：任何行变化须经 PR 并在本文件 changelog 追加（防漂移）。
