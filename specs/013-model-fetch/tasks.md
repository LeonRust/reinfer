# Tasks: Model fetch — pure-Rust ModelScope downloader

> Derived from specs/013-model-fetch/plan.md · 提交拆分：T1/T2→`feat(models): ...`；T3→`feat(cli): ...`；T4→`notes/docs`

## T1: `crates/models` — API 客户端（files 列表 + URL 模板）

- `list_files(owner_model)`：REST call（ureq+rustls）+ JSON 解析（Fixture = 本次实测 files JSON 摘录）；`FileEntry{name,size,sha256,is_lfs}`；URL 模板常量；owner/model 解析（`foo/bar` → 两段，非法→`Fatal`）
- Verification: 单测：fixture 解析（字段/大小/缺失 Sha256→Fatal）；owner_model 校验；URL 拼接快照断言（两个模板与实测一致）

## T2: `crates/models` — 下载/校验/原子落盘/manifest

- `download_file`：GET（302 跟随）→ 流式写 `<to>/.<name>.tmp-<pid>` → sha256 比对（files API 值）→ 不匹配删除 temp 重试一次 → 仍失败 Fatal → rename 为 `<name>` → manifest 追加（幂等：已存在且 sha 匹配 → 跳过）；ENOSPC → Oom
- Verification: stub HTTP 测试（本地 TcpListener：302→200 小文件 + 坏 sha256 失败路径 + temp 清理）；manifest 幂等

## T3: `reinfer model` 子命令

- `list`/`get`（--file/--all/--to；缺省 `~/models/reinfer`）+ help 文本；std 参数解析
- Verification: `cargo run -p reinfer -- model help` 输出规范；参数解析错误→exit 2 + 用法提示（单测解析函数）

## T4: 真机端到端（人工；不入日常 CI）

- `reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q8_0.gguf`
- Verification: 675,710,816 B；sha256 == `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`；manifest 留痕；记录 notes（含代理 env 示例）

## T5: 文档同步

- README（模型获取段）、CLAUDE.md（跨库纪律：模型一律 ModelScope + 示例命令）、feature-list（模型获取行）
- Verification: README 命令可复制；feature-list 新行锚 013

---

Completion gate：T1–T5 accepted；端到端 sha 通过 + notes 记录。后续：L3 数据管道（001 T1-T4 / 004）直接消费 `reinfer model get` 产物。
