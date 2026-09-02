# Gate fixture — decode regression gate protocol card（006 T7 / S1-8 生效点）

> 判定协议卡（doc + machine-readable 数值镜像 = `bench/gate-fixture.json`，脚本/测试读 JSON）。
> 门禁定案（2026-09-01）：**decode 唯一门禁 = 0.85× llama.cpp CUDA**；CI 红判据 = 中位数
> ≤0.9× 基线（10% 阈值，与 000 的 5% CPU 档并存——CUDA 档 10%、CPU 档 5%，各自独立判定）。
> `docs/design/benchmark-gap-2026-08-29.md` §4 阶梯（150-250 tok/s…）为**预期轨道（记录档，
> 非判据）**。

## 1. 参照与阈值（定案，只读）

| 项 | 值 | 来源 |
|---|---|---|
| 参照引擎 | llama.cpp CUDA（tg512 单流 decode，`-b 1 -n 512 -fa 1 -ngl 99 -r 5`） | `bench/baseline-llamacpp.json` |
| 参照 commit | `f280b26983ad0fdb705a0d9ebf0503e76f2899b0`（tag b10615） | 同上 |
| 构建 flags | cmake `-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF -DCUDAToolkit_ROOT=/usr/local/cuda-13.2` | 同上 |
| 模型 | Qwen3-0.6B F16 GGUF，sha256 `d04bceb6…`（1.40 GiB） | 同上 |
| 机器 | RTX 5090 Laptop（sm_120a，~896 GB/s，driver 595.84，CUDA 13.2） | `bench/runner-info.json` |
| **参照中位数** | **352.70 tok/s**（5 次 raw 355.41/352.70/352.77/352.17/351.49） | 同上 |
| **门禁（0.85×）** | **299.8 tok/s**（tpot 3.3356 ms） | 计算：round(0.85×352.70, 1) |
| **CI 红（0.9×）** | **317.4 tok/s**（tpot 3.1506 ms） | 计算：round(0.9×352.70, 1) |

## 2. 计算式（tok/s 来源）

- 引擎侧测量：bench-vs-vllm 台面单流套件 `perf_c1`（并发 1、out=64、nreq=20；
  前 2 个探索请求 `is_warmup:true` 剔除，即 harness 预热语义）。
- 每请求 `tpot` = 该请求 ITL 序列均值（秒）；取全部非 warmup、无 error 请求的
  **中位数 tpot**（`statistics.median` 口径，与 gen_report 一致）。
- `tok/s = 1 / median_tpot_s`（tpot→tok/s 单调，等价于逐请求 tok/s 的中位数）。
- 判定（三态，与 `bench/gate-fixture.json` verdict_cases 同源）：

| 判定 | 条件 | 含义 | 处置 |
|---|---|---|---|
| GREEN（绿） | tok/s ≥ 0.9×（≥317.4） | 无 10% 回归，CI 绿 | 记录新值即可 |
| PASS-CI-RED（过） | 0.85× ≤ tok/s < 0.9×（299.8..317.4） | 门禁达成但 CI 红（10% 回归） | 记录 + 说明差距归因 |
| FAIL（红） | tok/s < 0.85×（<299.8） | 门禁未达成 | 修性能；notes 记录 |

- 计算/判定自动执行：`bench/perf-gate.sh`（唯一入口）；数值一致性由无 GPU 单测
  `gate_fixture_verdict_cases`（bin/reinfer）断言三 case（过/红/绿）+ 边界 + 与
  baseline-llamacpp.json 的派生一致性。

## 3. 可复制的门禁执行流程（每波 Wave 验证用）

```bash
cd /home/dora/Dev/ai-tokens/reinfer

# 0) 前置：真机（RTX 5090）+ REINFER_CUDA_NVCC（13.2 JIT 必需）+ 模型已下载
export REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc
export CUDA_VISIBLE_DEVICES=0

# 1) 构建 release（--features cuda；workspace default-members 不含 ascend）
cargo build --release --features cuda

# 2) 门禁（build + 参照存在性检查 + run_all.py perf_c1 测量 + 判定 PASS/FAIL）
bench/perf-gate.sh
#    或复用现有二进制：bench/perf-gate.sh --skip-build
#    退出码：0 = PASS（≥0.85×）；1 = FAIL（<0.85×）；2 = 前置缺失/测量失败

# 3) 记录（见 §4 表——手动填 commit/构建 flags；数值由脚本/JSON 自动保持）
```

## 4. 重跑登记（每波记录新值/阈值 table）

| 日期/波 | engine commit | 构建 flags（engine 侧） | 测量 tok/s | tpot p50 ms | 阈值（0.85×/0.9×） | 判定 |
|---|---|---|---|---|---|---|
| 2026-09-01 S1-8 建档 | — | — | — | — | 299.8 / 317.4（定案参照） | 门禁协议卡生效 |
| 2026-09-01 S1-9b 现状 | （notes.md 当日 commit） | release --features cuda | 284.8 / 270.3（run CLI 短 kv） | — | 299.8 / 317.4 | FAIL（≈95%） |
> **2026-09-02 硬度注记（C-option 关闭）**：`sudo nvidia-smi -pl 110` 在该机返
> "Changing power management limit is not supported for GPU: 00000000:02:00.0"——
> RTX 5090 Laptop VBIOS 不支持运行时功耗修改；95W = 硬件边界。结合 017 系列
> 全量零收益证据（p1_gu 650GB/s 扇区饱和 / barrier ~1µs / 层内链无重叠窗口），
> 单流 decode 门禁 299.8 判定为**本硬件+本架构下的不可达边界**；perf-gate 登记
> 值（249.8）即为终值记录。
| 2026-09-02 S1-11 W=2（017-a; 017-b 回退终态; perf-gate.sh --skip-build） | （工作区未提交; 待收口提交后 git rev-parse HEAD 回填） | release --features cuda（REINFER_CUDA_NVCC=13.2; 无 sched=串行路径; REINFER_FUSED_BW 缺省 2） | 249.8 | 4.003 | 299.8 / 317.4 | FAIL（83.3%; 较 243.9 +2.4%） |
| <!-- 每波一行：值+commit+flags 由执行者填写；判定=§2 表三态 --> | | | | | | |

> commit+构建 flags 采用 **manually fill** 纪律：脚本只写数值与判定；每波执行者在
> 本表登记 engine 侧 commit 与构建 flags（`git rev-parse HEAD`、`cargo build
> --release --features cuda` 等），并同步 notes.md 当日记录。

## 5. 参照变更（--update-baseline）

参照自身变更时（新 llama.cpp 构建/换机/驱动大版本），先按 §1 参数重跑 llama-bench 5 次
取中位数，再：

```bash
bench/perf-gate.sh --update-baseline <新参照 tok/s>   # 重写 baseline-llamacpp.json：
                                                       # 更新 reference_tok_s/median_5/gate_math
                                                       # 旧值入 history[]；commit/flags 手动填 §4
```

新参照生效后，`bench/gate-fixture.json` 的阈值须同步更新（或直接重跑无 GPU 单测——
`gate_fixture_verdict_cases` 会以 fixture JSON 与 baseline JSON 的派生一致性为断言失败点）。
