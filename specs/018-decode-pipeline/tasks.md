# Tasks: decode stage pipeline (specs/018)

> Status: open 2026-09-02 · Wave serial mini: P1a → P2a → P3a（零收益→回退纪律）
> Registration: roadmap S1-12 · 017 波终点后（barrier 归因更正、带宽饱和已证）。

## P1a — 双组 barrier 树 + 边同步（无算术改动）

- [ ] 审计：为每组选定参与块集合（A=gather/qkv/flash 生产/消费；
      B=o/gu/down；数据边=add_rms(o)（A→B 边）与 add_rms(down)（B→下轮 A）
      分别一个边 barrier）——从现有 partial-participant P 集合推导
- [ ] layer_fused_body 增加"组模式"：barrier 调用改为两棵（组内）+ 边
      （两点式集合）——**所有 stage 内部数值代码逐行不动**
- [ ] `REINFER_FUSED_PIPE=1` host 开关 + 集合上传
- [ ] 位级 5 门 + engine A/B（JIT 重编）全绿才保留
- [ ] REINFER_DECODE_PROFILE A/B（层均值/gpu busy）：≥5% 才留；

## P2a — 组 B 轮空期 L2 预取（条件于 P1a 保留）

- [ ] 在 B 组空闲（等待边 barrier）窗口发射 `prefetch.global.L2`（下一层
      qkv/gu 权重行）——**无数据变更**
- [ ] 同上判据（A/B 段表 ≥5%+位级+engine A/B）

## P3a — 关键边 producer-count 轮询（条件于 P1a 保留）

- [ ] 边 barrier 从 sense-reversal 广播改为"等待生产者计数"
      （global 计数器 + consumer 自旋）——集合/数据序不变
- [ ] 同上判据

## 收尾

- [ ] 每步结果（段表/位级/判定）入 notes.md S1-12 节
- [ ] 终态：REINFER_FUSED_PIPE 缺省值（保留的最大步）或 0（全零收益）
      + roadmap S1-12 记录 + 提交
