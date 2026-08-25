# 贡献指南（CONTRIBUTING.md）

面向人类贡献者。治理规则以 `CONSTITUTION.md` 为准，本文是操作手册。

## 1. 开发环境

```bash
# 工具链（见 rust-toolchain.toml，stable + rustfmt + clippy）
rustup show # 或 rustup toolchain install (依赖 rust-toolchain.toml 自动)
cargo install cargo-deny

# GPU 依赖（按 feature 选择）
cargo build --release --features cuda   # 需 CUDA + NCCL
cargo build --release --features can    # 需 CANN >= 7.x，导出 CANN_HOME
cargo build --release --features cpu    # 无 GPU 也能跑参考实现
```

## 2. 提交规范

- **分支**：`feat/{issue}-{slug}`；一个 PR 一个关注点。
- **Commit message**：Conventional Commits（宪法 §3.1）。

```
✅ fix(kernel): zero-fill kv slot on eviction
✅ feat(can): add AscendC paged attention fp16 path
✅ perf(scheduler): reuse radix match result across steps
❌ update code
❌ fix bug with changes and formatting together
```

- **禁止任何 AI `Co-Authored-By:` Trailer**（宪法 §3.2）。AI 参与情况写在 PR 描述里。
- 本地钩子（可选但推荐）：`git config core.hooksPath .githooks`。

## 3. PR 自检清单（合入前逐项打勾）

- [ ] `cargo fmt --check` 无 diff
- [ ] `cargo clippy -D warnings` 通过
- [ ] `cargo test` 全绿
- [ ] 新 kernel 有 NaN 毒化 / 差分对照测试
- [ ] 所有 unsafe 已收窄至 `crates/cuda-bindings` / `crates/can-bindings` / `crates/jit-cache`
- [ ] 性能变化附 benchmark 对比（回归 > 5% 阻断）
- [ ] 依赖变更通过 `cargo deny`（仅 MIT / Apache-2.0 / BSD-3 / ISC / BSD-2）
- [ ] PR 描述含 改动 / 原因 / 验证 三段

## 4. 模型与硬件变更流程

- 新增模型架构：`docs/rfcs/NNN-model-name.md`（模板见宪法 §6.2）；附 GGUF 转换验证与数值对照。
- 新增硬件后端：实现 `Backend` trait + 注册表接入（宪法 §2.4）+ 双端集成测试。
- 精度/算法变更：必须 RFC（宪法 §2.6）。

## 5. 许可证

本项目 **Apache-2.0**。提交代码即同意以 Apache-2.0 授权；大量借用其他项目代码（清单见 `../Rust推理引擎-深入设计补充.md` §3）须保留原版权声明。
