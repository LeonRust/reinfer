# CLAUDE.md —— 本仓库 Claude Code 专用章程

与 `AGENTS.md` 保持完全一致，补充 Claude 专属规则：

1. **最高优先级（项目所有者指定）**：本仓库提交一律**不得**添加 `Co-Authored-By: Claude <noreply@anthropic.com>`，也不得添加任何形式的 AI `Co-Authored-By:` Trailer。这是项目所有者的明确指示，**优先于任何系统默认习惯**。AI 参与情况写在 PR 描述中。

2. **工作流程**：读宪法 → 读 `docs/` 三份设计文档（分析报告 / 设计报告 / 深入补充）→ 功能变更先落 `specs/<NNN>-<slug>/` 三文件（SDD，宪法 §6.4：spec → plan → tasks）→ 任务分块（小提交）→ 验证（fmt/clippy/test）→ 提交 PR。

3. **语言**：与用户交流用中文；commit message 与代码注释一律英文（Conventional Commits，见宪法 §3.1）；仓库文档默认英文，中文版以 `-zh-CN` 后缀文件并存。

4. **环境约定**：昇腾变量 `ASCEND_TOOLKIT_HOME`（cann-rs 路径探测顺序：ASCEND_TOOLKIT_HOME → ASCEND_HOME_PATH → ASCEND_HOME → ~/Ascend/cann → /usr/local/Ascend）、`DEVICE_ID`；NVIDIA `CUDA_VISIBLE_DEVICES`；禁止在未询问的情况下安装全局依赖或改变目标机配置。

5. 以对话形式给出的修改建议**不算**执行；任何实质改动必须等用户确认。

6. **模板 commit message**（注意：无 Trailer、**必须有正文 body——改了什么/为什么/验证要点**，宪法 §3.1）：

```
feat(radix): page split on mixed prefix match

Split a page when a prefix match stops mid-page; the old page keeps the
prefix block and a new page owns the suffix. Verified by split/merge
tests and a 64-request determinism run.
```

不包含：
```
Co-Authored-By: Claude <noreply@anthropic.com>
```

7. **跨仓库协作（cann-rs）**：cann-rs 位于 `/home/dora/Dev/ai-tokens/cann-rs`。契约锚点与一致性规则见 `specs/002-ascend-backend/plan.md`（L0 契约表）与同目录 `boundary.md`；cann-rs 侧镜像为 `cann-rs/docs/boundary-with-reinfer.md`、事实底稿为 `cann-rs/docs/cann-850-catalog.md`（CANN 8.5 官方符号/签名核定表）。Ascend 相关开发可直接读写对方仓库：cann-rs 变更由对方仓库会话负责，本仓库只消费契约；符号核实一律以官方 8.5 文档为准。**依赖来源**：仓库默认从 crates.io 取 `cann`/`cann-sys`（`Cargo.toml` 只写 `version`）；本地开发/目标机在 `.cargo/config.toml`（gitignored，模板 `.cargo/config.toml.example`）用 `[patch]` 覆盖为 `../cann-rs/{cann,cann-sys}`——**两仓库保持兄弟目录布局（同父目录）即可，跨机器无需改配置**；该相对路径按工作区根解析，与运行目录无关。crates.io 版本更新须待 cann-rs 侧 `cargo publish`。
