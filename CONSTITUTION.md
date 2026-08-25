# reinfer 项目宪法 v1.0

> 项目：**reinfer** —— 用 Rust 构建支持 CUDA + 昇腾 CANN 的推理引擎
> 生效：自维护者批准之日起；修订须经 RFC 流程（见 §6）
> 关联文档：`docs/design/Rust推理引擎设计报告.md` · `docs/design/Rust推理引擎-深入设计补充.md` · `docs/design/ai-tokens-分析报告.md`

宪法是最高治理文件：所有代码、提交、评审、AI 代理行为必须遵守。与宪法冲突的改动一律拒绝。

---

## 第 1 章 技术栈

1.1 **语言**：Rust，`edition = "2024"`（标准生成项目默认），MSRV 跟随官方 stable。禁止依赖要求 nightly；唯一例外：`crates/kernels-*`（kernel 生成器）可显式使用 nightly，须有 MSRV 注释与 CI 门禁。

1.2 **工具链**：rustfmt + clippy 强制；`cargo-deny` 检查依赖许可与安全告警；release 构建 `lto = "thin"`、`codegen-units = 1`。

1.3 **禁止 `torch` 依赖**——模型权重从 GGUF / safetensors 直读（见设计报告 §3.5）。

1.4 后端按 Cargo feature 编译（`cuda` / `ascend` / `cpu`），任何后端不得引入强制依赖；无 GPU 环境必须能编译出 CPU 参考实现。

## 第 2 章 架构不变量（不可违反）

2.1 **窄 FFI**：`engine/`、`scheduler/`、`radix_cache/` 等核心 crate 禁止 unsafe（`#![forbid(unsafe_code)]`）；所有 unsafe 收敛于 kernel FFI crate（`crates/cuda`、`crates/ascend`、`crates/jit`；昇腾侧 unsafe 宿主为外部 `cann` crate，经窄接口消费）。

2.2 **三档 KernelProvider**（Vendor > Native > Jit）：每个 OpKind 至少一个纯 CPU 参考实现；任何档的 kernel 必须通过与 naive 参考的数值差分测试（FLA 文化）。

2.3 **确定性铁律**：decode batch 一律按 `req_id` 排序；采样种子固定；TP rank 间逐 token 一致（mini-sglang #113 教训）。

2.4 **Registry 插件化**：attention / quant / backend / 采样器均须经注册表接入；新增后端必须附带双端（CUDA/CANN）集成测试。

2.5 **内存不变量**：KV 页生命周期由所有权 + epoch 保证；禁止裸指针越过 FFI 边界；OOM 级联四步链（**驱逐 → 抢占 → 拒绝新请求 → 保底预留**）必须保留。

2.6 **数值纪律**：精度或算法变更须 RFC（附 bit-match / 差分对照数据）；kernel 变更必须附 benchmark 数据（或标注"无性能变化"）。

2.7 **单二进制交付**：`cargo build --release` 产出一体化 `reinfer`（server/cli/bench）；Python 绑定仅作附加 crate。

## 第 3 章 提交与 Git（最高强制条款）

3.1 **Conventional Commits 1.0**：`type(scope): summary`
   - type ∈ `feat | fix | perf | refactor | docs | test | chore | ci | build`；
   - summary ≤ 70 字符、祈使句、句末无标点；**summary 与 body 一律使用英文**；scope 为 crate 或子系统名（`kernel` / `scheduler` / `radix` / `can` / `cuda` / `ipc` / `server` / `ci` …）；
   - body 与 summary 之间空行；一条提交只做一件事。

3.2 **AI 署名禁令**：任何提交（人工或 AI 代理）**不得包含** `Co-Authored-By: Claude <noreply@anthropic.com>`，也不得包含任何形式的 AI `Co-Authored-By:` Trailer。AI 参与情况写在 **PR 描述**中，不进入 commit message。启动钩子强制校验：`git config core.hooksPath .githooks`。

3.3 **分支**：`main` 受保护（禁止直推，需 PR + 2 个批准）；分支名 `feat/{issue}-{slug}` / `fix/...` / `perf/...`；PR 可 squash；PR 描述必须含"改动 / 原因 / 验证"三段。

3.4 **门禁**（全部通过才能合入）：`cargo fmt` 无 diff、`cargo clippy -D warnings`、`cargo test` 全绿、`cargo deny`；涉及 GPU 的改动附对应后端 bench 对比；**性能回归 > 5% 视为阻断**。

3.5 每条提交应当可独立编译；禁止"格式化 + 逻辑"混在一个提交里。

## 第 4 章 代码规范

4.1 rustfmt 默认配置（见 `rustfmt.toml`，max_width = 100）；命名 snake_case / SCREAMING_SNAKE_CASE。

4.2 lint 基线：`unsafe_op_in_unsafe_fn = deny`；`missing_docs = warn`（公开 API 必须 `///` 文档）。

4.3 **模块上限**：单文件 > 500 行应拆分（SGLang 5291 行调度器即反面教材）；数据流用显式类型（`OpConfig` / `ReqId`），禁止字符串散传。

4.4 公开 API 必须文档化；设计决策写入 `docs/`（借鉴 vLLM `docs/design/` 文化），一次改动只改变一个决策点；**仓库内文档默认英文，中文版本以 `-zh-CN` 后缀文件并存（如 `README.zh-CN.md`），并在主文档顶部提供双语切换链接**。

## 第 5 章 AI 代理条款

5.1 允许 AI 辅助开发（含 Claude Code），但：任何 AI 修改必须经 PR 人工 review；**禁止 AI 自主直接提交到受保护分支**。

5.2 AI 必须遵守 §3.2（禁止 AI 署名 Trailer）；触及第 2 章不变量的 AI 改动必须先有 RFC。

5.3 AI 提交必须伴生测试（无测试的 AI 改动一律拒绝）；对不熟悉的子系统应先读 `docs/` 三份设计文档，不确定时显式标注 `[speculative]` 并向维护者提问。

## 第 6 章 治理

6.1 面向人类的贡献流程见 `CONTRIBUTING.md`；面向 AI 代理的章程见 `AGENTS.md` / `CLAUDE.md`。

6.2 宪法修改与第 2 章不变量的变更：必须走 RFC 流程（`docs/rfcs/NNN-{name}.md`），由 2 名以上维护者批准。

6.3 里程碑门禁：P0–P4 验收标准见 `Rust推理引擎设计报告.md` §7；每阶段验收通过后方可宣告进入下一阶段。

6.4 **SDD 工作流（Spec-Driven Development，所有功能变更必走）**：按 `Specify → Plan → Implement → Validate` 四阶段执行，产物为 `specs/<NNN>-<slug>/` 三文件体系：
   - `spec.md`（需求规格 = 唯一真实来源）：Problem Statement / Success Metrics（**必须可测试**）/ User Stories / Acceptance Criteria / Non-Goals / Constraints；
   - `plan.md`（架构方案）：Architecture Decision / Module Breakdown / Interface Contracts / Risk Assessment；
   - `tasks.md`（任务清单）：原子任务 + 每条可独立验证（验证=验收标准）。
   规则：
   a. **粒度检验**：换一种技术栈实现该 Spec 仍须成立——Spec 只写 WHAT，Constraints 只允许"外部限制"；
   b. **增量 Spec**：发现新需求或 BUG 立即写/更新增量 Spec，禁止 Spec 腐烂（spec rot）；Spec 与代码变更同步提交；
   c. **微变更豁免**：纯局部改动（文案/样式/局部变量）免走全流程；
   d. **Validate 不可豁免**：Spec 替代需求文档，不替代 Code Review 与 CI 门禁（§3.4）；
   e. 与 RFC 的关系：RFC 决定"做不做/为什么做"，Spec 决定"做到什么程度"；冲突时以 RFC 为准。

## 附：许可证与合规

- 本项目采用 **Apache-2.0**（与 MIT / Apache 依赖兼容、含专利授权；与被借鉴的 vLLM / SGLang / FlashInfer 同许可）。
- 借鉴资产清单见 `深入设计补充.md` §3；各上游许可（MIT / Apache-2.0）按对应声明保留版权；通用算法（PagedAttention / RadixAttention / FlashAttention 系列）按论文与上游许可用途声明。
