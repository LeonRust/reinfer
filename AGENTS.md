# AGENTS.md —— AI 代理章程

本文件是 **宪法第 5 章** 的机器可读版。任何 AI 编码代理（Claude Code、Copilot、Cline 等）在本仓库工作必须遵守：

1. **先读宪法**：任何编辑前先读 `CONSTITUTION.md`（尤其第 2、3 章）与 `docs/` 下三份设计文档。
2. **禁止 AI 署名 Trailer**：提交与生成的 commit message **严禁**包含 `Co-Authored-By: Claude <noreply@anthropic.com>`（或不含任何 AI `Co-Authored-By:` Trailer，宪法 §3.2）。AI 参与情况写在 PR 描述中。
3. **禁止直接提交 main**：所有改动走 PR 分支（`feat/`、`fix/`、`perf/`…），等待人工 review。
4. **强制验证**：提交前必须 `cargo fmt --check && cargo clippy -D warnings && cargo test`；涉及 GPU 后端的改动附 benchmark 数据。
5. **测试伴生**：每个 AI 提交必须附测试（差分测试 / 单测），无测试不提交。
6. **先问再猜**：对不熟悉的子系统，先读 `docs/` 设计文档；不确定时显式标注 `[speculative]` 并向维护者提问。
7. **架构不变量**：不得在未通过 RFC 的情况下修改第 2 章不变量（三档 kernel 分层、确定性铁律、窄 FFI、OOM 级联链等）。
8. **语言**：commit message 一律英文（宪法 §3.1）；仓库文档默认英文，中文版以 `-zh-CN` 后缀文件并存。
9. **SDD（Spec-Driven Development）**：任何功能变更必须先写/更新 `specs/<NNN>-<slug>/{spec,plan,tasks}.md`（宪法 §6.4）——先 Spec 后代码；无 Spec 的功能改动一律拒绝（纯局部微改豁免）。

维护者提示：`git config core.hooksPath .githooks` 可启用 commit-msg 钩子强制执行 §3.2。
