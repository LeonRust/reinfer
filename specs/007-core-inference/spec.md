# Spec: Core inference — CPU 全链路（无加速卡也能推理）

> Status: proposal · r2（2026-08-28 四代理评审修订：C3 档位分层/C4 记录不设档/启动条件/真机纪律）· Owner: maintainers · Created: 2026-08-28 · Parent: specs/001-gguf-loader（P0-06/P0-07 落地）
> 锚：feature-list P0-06/P0-07；CLI 契约 §6.2 `run`（r2 v2.15：`--backend cpu`）；`--backend cpu` 为 005 一致性套件与无 GPU CI 的载体
> 前序：001（GGUF/codec/arch）+ 004（**交付面=scaffold/BPE encode/decode；SPM encode 未交付——007 不依赖 SPM**）+ 012（sampler 管线 + refs）+ 013（model resolver）+ **014（referee/parity harness/gate 脚本——T3/T4 硬前置）**

## Problem Statement

014/015 让 CUDA/Ascend 跑出 token；引擎必须有**无加速卡也可推理**的兜底路径（项目初心：`--backend cpu` 恒定存在）。CPU 档是全链路正确性基准（模型加载→逐层执行→流式输出），也是无卡 CI 的载体；其数值语义（fp32 累加 naive）与 012/014 的参考函数同源——**CPU 后端 = 参考实现本身的完整化**，天然满足「参考必须本机可执行」铁律。

## 范围与验收

| 块 | 功能 | 锚 | 验收 gate（r2） |
|---|---|---|---|
| C1 执行器 | `crates/cpu`：layer loop 完整化（embedding → 逐层 RMSNorm/RoPE/GQA-attention/MLP → final norm → logits）——**fp32 累加 naive matmul；权重 Q8_0（dequant→fp16 RNE）与 F16** | 001 codec + 012 refs | golden：0.5B 真实存档（**013 获取，env 注入——r2 措辞修正：非「tiny」**，C1 golden 来源=000 存档或复用 014 T1 存档；**禁止自生成**——r2 反例修正）逐 token 对拍；错误路径（维数不齐/未知 arch/OOV id）报错即断言 |
| C2 CLI | `reinfer run <model> --backend cpu "<prompt>"`（契约 §6.2：`-n/-t/--seed` 等；prompt 流式 stdin 退化） | CLI 契约 §6.2/r2 v2.15 | 人工 + 脚本流式出 token；temp=0 复现；**ModelRef 三态（本地/repo 单候选/多候选 -q|-f）验证——r2 补** |
| C3 对拍 | 与 llama.cpp 同机对拍（golden ids 注入 + temp=0；referee=f280b2698） | 014 parity 协议 | **r2 档位分层：F16 档 token 100%（硬——双方 fp32 累加同构，可达）；Q8_0 档 ≥99.9%（硬）+ 回退档（一致率 + logits drift 记录——r2：llama.cpp Q8_0 CPU 走块量化点积 ~1e-3 误差，100% 不可达，同 014 判据教训）** |
| C4 记录 | CPU decode 吞吐记录 | — | **r2 记录不设档**：≥60% 判据删除——带宽分析（0.5B Q8_0：naive 单线程 fp32 每次 decode 读 ~2.1 GB 权重 7-12 tok/s vs llama.cpp 24 线程 SIMD 100-200 tok/s，差距 8-25×；预计 5-40%）→ 本档结构不可达，且 60% 会与 feature-list P0-07 造成虚假 gate；吞吐产出+notes；thread 并行（REINFER_THREADS）为 P 档延伸 |

## Success Metrics

- 无 GPU/NPU 环境：`reinfer run --backend cpu <model> "hello"` 直接流式出 token（零配置、零联网——013 `AUTODOWNLOAD` 纪律同上）
- 对拍达成：F16 token 100%（硬）+ Q8_0 ≥99.9%（硬，回退档记录）
- 吞吐记录产出（notes 四元组；不设 % 闸）

## User Stories

1. 作为新用户：无卡设备第一次跑通——「装完就能跑」是引擎底座承诺。
2. 作为验证者：CPU 档即差分参考本体（014 的 CPU naive 全链路化）。
3. 作为 CI 作者：无 GPU/NPU CI 用例的载体（008 can-gpu 缺位时 CPU 全链仍可冒烟）。
4. 作为服务化读者：005 的 `--backend cpu` 一致性套件（P0-06）依赖本 spec 的 C1/C2。

## Acceptance Criteria

- [ ] C1：layer loop 完成（逐层单测 + 0.5B 存档逐 token 对拍）
- [ ] C2：`run` 流式输出；temp=0 确定性（seed 固定复现）；ModelRef 三态齐全
- [ ] C3：F16 token 100% + Q8_0 ≥99.9%（20 prompts；llama.cpp golden ids 注入；`#[ignore]`/脚本）
- [ ] C4：吞吐 notes 产出（含与 llama.cpp 比值记录）
- [ ] 模型标识零硬编码（013 铁律；grep gate）
- [ ] P0-06/P0-07 feature-list 回写（判据改为 r2 档位分层——P0-07 原「≥60%」删除）；phase-plan 回写

## Non-Goals

- SIMD/多线程性能冲锋（本档单线程 naive；thread 并行=记录项延伸）；KV chunk/量化分页等内存工程（memory crate 服务设备侧——r2 措辞）
- 超越 llama.cpp 的对拍强度（不做流式 detokenizer 逐 token 对拍——004 判据原文）
- batch/continuous batching（005）；Radix/grammar（P3）；turbo/static KV（P 档）

## Constraints

- **判据可执行性**：对拍=同机 llama.cpp（referee=f280b2698，**014 T0**）；golden=013 真实存档（env 注入）；数值=refs 同源（dequant=001 codes 单乘——r2 单源化：kernels 不再定义 dequant_ref）
- **模型标识零硬编码**（013）：代码零模型常量；测试虚构（golden 存档除外）
- **llama.cpp 对拍协议继承 014**（temp=0/golden ids/referee pinned/special 固定）
- **tokenizer 判据继承 004**（encode 100% + 增量 decode 自洽；不与流式 detokenizer 逐 token 对拍）
- **真机纪律（r2 增补）**：loadavg<1 才采信 · 同机同参顺序执行 · temp=0 固定 seed 复现
- **确定性（r2 增补）**：单线程顺序 fp32 累加（IEEE）、无锁无设备 → 同机跨机位精确（仅数值整数序依赖平台 `f32` 语义——声明范围照 012）
