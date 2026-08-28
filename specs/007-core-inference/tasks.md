# Tasks: Core inference — CPU 全链路

> Derived from specs/007-core-inference/plan.md · r2（2026-08-28 四代理评审修订）
> **实施启动条件（r2 修订）**：001/012/013/004 交付（**004 交付面=scaffold/BPE decode+encode；SPM encode 未交付——007 仅用 BPE，不依赖 SPM**）；**014 达 M4/M5（T0 referee / T10 gate_throughput.sh+prompts / tests/parity.rs）为 T2–T4 硬前置**（r2：阻止先引用后创建——原「001/004/012/013 交付」表述不足以覆盖）；T1 可与 014 并行。
> 提交类型：T1→`feat(cpu)`；T2→`feat(bin)`；T3→`test:`；T4→`ci:`+`docs:`。模型标识零硬编码（013 铁律）。

## T1: crates/cpu layer loop（plan D1/D2）

- `model.rs`/`ops.rs`：embed（OOV 防呆）→ per-layer（RMSNorm/RoPE/GQA-attention/MLP gate，**silu 按 arch**）/final norm/logits；权重 Q8_0（001 codec 按需解量化——单乘语义 + **f32→f16 RNE**）与 F16 直读；GQA = 014 公式 + **三例（14/2、12/2、5/2）核验（r2）**；KV 连续矩形；fp32 累加
- Verification: 0.5B 真实存档（013，env 注入）prefill 单层/全链 diff（与 refs 同源 NaN 检查）；错误路径单测（维数不齐/未知 arch/**embedding 越界 id → LaunchError——r2**）

## T2: run 命令 CPU 接线（plan D3/D5；CLI 契约 §6.2 + r2 v2.15）

- **`reinfer run <model> --backend cpu`**（r2 命令面；`-q|-f/-n/-t/--top-p/--top-k/--seed/--max-model-len` 契约表）；streaming 打印（014 bin 侧 UI scaffold）；temp=0 argmax（tie-break 首个最大；**`-t 0`/`-t 1e-9` 边界单测——r2**）；seed 固定复现；stdin 退化（契约 §6.2）；**ModelRef 三态（本地路径 / repo 单候选 ensure / 多候选 -q|-f 缺→exit 2——r2 补：`-q|-f` 互斥用例见 013 交付验收，此处为集成验证）**
- Verification: 流式 token；`--seed 1` 两次一致；`-n` 截断；EOS 停（**r2：014 T9 生成语义同款——EOS 停/NaN logits 显式错误，不归 005**）；三态用例

## T3: token 对拍（plan D4；r2 档位分层）

- `bench/prompts/`（014 T10 同址）+ 复用 014 `tests/parity.rs` harness（`--backend cpu`）：golden ids 注入双方（llama.cpp referee f280b2698 = **014 T0**）+ temp=0；**r2：F16 档 token 100%（硬）+ Q8_0 档 ≥99.9%（硬；回退档=一致率+drift 记录——llama.cpp Q8_0 CPU 块量化点积有 ~1e-3 误差，100% 不可达）**；logits 全量 finite 硬断言（014 harness 先验）
- Verification: 20/20 prompts：F16 100% + Q8_0 ≥99.9%（`#[ignore]`/脚本，allowlist `l3-cpu` 行——**008 接线 r2 显式**；referee 未建（014 T0 未达）→ 记录档 + notes 声明）

## T4: 吞吐记录 + 文档（plan D5；r2）

- `gate_throughput.sh`（**014 T10 参数化产物，传参数不改造**）`--backend cpu`：gen tok/s + llama.cpp CPU 3 次中位数（loadavg<1；同机同参）；notes 四元组（CPU 身份）
- feature-list P0-06/P0-07 回写（**P0-07 判据删除「≥60%」——r2 记录不设档**）；phase-plan；grep 模型名 gate（014 T11 r2 命令复用）
- Verification: 记录产出（无 % 闸——spec C4 说理）；差异分析（带宽）入 notes

---

依赖表（r2）：T1（可与 014 M1 并行）；T2←T1；T3←**T2 + 014 T0/T10（referee）**；T4←**T3 + 014 T10（gate_throughput.sh）**。「先到先写、后到增补」仅防 `bench/prompts/` 文件冲突——不替代构建前置（r2 明示）。
