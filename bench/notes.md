# Bench notes

> 真机执行痕迹（009 T6；判定机 = RTX 5090 Laptop，见 `runner-info.json`）。

## 2026-08-27 — L1 (specs/009) 收尾

- 截止 commit：`e038030`（T6 之前）→ T6 收尾 commit 于下方待补
- 真机命令：
  ```text
  CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda -- --test-threads=1
  CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda --test smoke -- --ignored --test-threads=1
  cargo run -p reinfer-cuda --features cuda --example device_info
  ```
- 结果：
  - 模块内真机测试 29 through（串行）：设备信息（cc=12.0, 23.42 GiB, uuid=0fd2ce94…）、流/事件（未 record=完成态/record+sync=true/Drop 不挂）、三链往返（同步+异步 event 同步，1 MiB 模式逐字节一致）、泄漏（1000×1MiB，F7 公式）、错误注入（Oom=2、Fatal=101）
  - `--test smoke -- --ignored`（本收尾记录时以模块档为准；T6 首跑结果待记录）
- 特性记录：
  - `--test-threads=1` 为必须（并行会将泄漏测试的 memset 流量污染事件/同步断言）
  - 野指针毒化实验（`tests/poison-cases.rs`）实测 SIGSEGV——占位 `#[ignore]` 实验，勿在无隔离进程运行
- 判定："阶段+真机混合"验收（编译绿 + 无 GPU 单测 ≥7 + 真机 smoke/差分）**L1 达成**；GPU 三层门禁（F16/Q8_0 文本一致、3× CPU）属 L3 收尾范围（届时同表记录）。

> 本文件由维护者持续记录；性能数值类门禁（006/003）按 008 `baseline.json` 机制另立。
