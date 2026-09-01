# Tasks: decode-side kernel performance (006-2, r2)

> Derived from specs/006-2-decoding-kernel-performance/plan.md · 容差表唯一源 = 003 plan D7；
> 基准协议/供应链接 006-cuda-perf；r2 修订：T0 前置、T1 条件化、G1/G5 挂起 006-2b、T5 删。
> 依赖图：T0 → {(T1|T2), T3} → T6。

## T0: 前置微基准与参照测量（决定 G3/T1/T2 存废；1-2 人日）

- 003 naive decode-attn 隔离微基准（同 006 协议：同模型 sha、KV f16、预热≥3 取中位数）。
- **llama.cpp CUDA 参照构建+测量**：nvcc 13.2 工具链、f280b2698、`-DGGML_CUDA=ON`、
  `CMAKE_CUDA_ARCHITECTURES=120`；llama-bench 按 006 plan D7 参数表（`-b 1 -n 512 -fa 1
  -ngl 99`）+ Qwen3-0.6B fp16 GGUF（**sha 钉死**）单流 decode，5 次中位数 → baseline.json。
  参照值 sanity：预期 400-900 tok/s；测得 <300 或 >1200 须复查协议后再判门禁。
- 附独立差分：llama.cpp 采样结果对拍（R3 自参照金标）。附加绝对下限判定：参照 <300 tok/s
  时门禁取 max(0.85× 实测, 300 tok/s)？
- Verification: baseline.json 落档（commit/构建标志）；记录"G3 存废"结论于 notes（T0 退出
  判据：003 naive ≥0.85× → G3 收口为记录，T1/T2 全砍；未达标 → 走 T1/T2 或按触发条款）。

## T1: decode-attn vendor 档（条件化：三条件齐备才执行）

- 条件①②③：现成 API 供应商资产 + **许可扫描通过（R2 前置：结论"不可分发"→ 跳过
  vendor 档记录）** + sm120a 有 cubin。manifest 入库 + sha256 校验、离线硬失败。
- Verification: 差分 ≤D7（fp16 出）逐 token 与 003 naive 一致；许可结论行入 notes；
  解码档对整机门禁的贡献值（相对整机 tps 记录，不做单核 0.85× 归因）。

## T2: decode-attn Jit FMHA 档（前置 gencode 表锚 003 plan D2）

- 摘录式 vendored 头 + 自有 `.cu`（JitCache；gencode per 003 plan D2 梯度表——
  **sm120a ≥12.8**（非 006 D2 旧值 13.0））；paged KV 接口对齐 003 T11（块 16/32 +
  MemOps）；GQA 头映射与 003 pin 一致；heuristics 按 flashinfer decode 模式启发。
- Verification: 差分 ≤D7；无匹配 arch/编译失败 → 回退 003 naive（不假装降级）；
  4K seq decode 文本 100% 一致 + logits drift 记录项；时间盒 2 周（R4）。

## T3: GPU sampler 链（r2：函数级 bit-identical + 005 D5 参数面 + 单 launch）

- 参数面/顺序 = 005 D5 全链（bias→penalties（freq/presence/repeat）→bad words→
  temperature→min_p→top_k→top_p→gumbel→argmax）；未覆盖参数 → 显式 CPU 回退并记录。
- 单流（D2，避免每步 event/同步）；每步 **1 次 CUDA launch**（计数规则：不含 lm_head
  GEMM；含 penalty+softmax+采样+TokenOut 统计）；graph 视图 = 桶内 sampler 节点 ≤1；
  temp=0 硬件 argmax（tie-break=首个最大）。
- Verification: ① 函数级：**同 logits 输入下** GPU vs CPU（llm-samplers 0.0.7）×
  10 prompt × 64 tok，temp=0 **bit-identical** 且 tie-break 规则钉死；② temp>0 分布：
  null 对照口径（与 CPU-vs-CPU 基线差比较，样本 ≥5000 tok 或逐步对齐 ≥10×64 步——
  拒绝裸阈值 0.05，其低于噪声底）；③ 计数：eager 单 launch（计数软件验证）、graph
  节点 ≤1；④ 无 GPU CI = 选择器恒 CPU 路径单测。

## T4: 融合核组（r2：仅当 006-2b 重开；②① 优先，③ 条件化，④ 不做）

- ② fused MLP-SiLU（首位：每层 3→1 核、权重占比 ~70% 主导族；CUTLASS epilogue 优先）
  ① fused norm+add（D1 自留地；与 CPU 路径共享语义）
  ③ fused QKV+RoPE —— 仅当 Jit attn 需要 contiguous qkv 前端；
  ④ ~~attn-out+add+norm~~ 不计划（graph-on 无收益；差分面照付）。
- 全组收益尺：≥max(5%, 2× 噪声带)（同 006 协议 5 次中位数 vs 锁定基线）；
  计数双指标：eager launch 数 + graph-on TPS 增量；per-layer ≤4、per-step = 4L+2。
- Verification: 每项 融合前/后 kernel 数+寄存器往返记录（ncu 或等价实测）；
  差分 ≤D7 表；eager 与 graph-on 双模式一致（006 D4 硬门）；回退=非融合组合恒可用。

## T5: ~~warp specialization~~（已删，理由见 plan D5；如 006-2b 评估支持则经新增量 spec 重开）

## T6: 基准回归门禁

- llama.cpp CUDA 参照（T0 产物）复跑；0.85× 唯一门禁判定；benchmark-gap §4 阶梯
  同步校订（**预期轨道记录，非判据**）。
- Verification: baseline.json 更新（5 次中位数、锁定 commit/构建标志）；CI 红判据
  （δ ≤0.9× 基线）；T1-T4 的 fallback/eager 计数器汇总入 notes；`run_all.py
  --engine both --suite perf_c1` 复核并与阶梯记录对齐。

## 依赖图

- T0 → 全部；T1/T2 并列（条件化）；T3 独立（可最早并行）；T6 最后；
  006-2b（G3/G5 重开）触发条件 = 006 落地 + ncu profile 评估通过，不在本轮门禁内。

## T-305 登记（2026-08-29：006 集成收口）

- **双流模式①（事件入图/prefetch）**：登记未实现——graph.rs 无事件节点支持
  （cudaGraphAddEventRecordNode 等），引擎单流设计（V1 串行）；模式②（捕获期
  no-overlap）由 graph.rs capture_in_progress() 提供（REINFER_GRAPH_NO_OVERLAP
  默认 on），运行期单流语义成立。模式① 需新增量 spec（graph 事件节点面）。
- **BLOCKER（decode 步图重放）**：decode 步 cublas gemm 节点无法声明 KernelSpec
  （每层 7 节点 × 28 层 + lm_head；handle/grid/block 不可得）→ graph.rs finish
  计数校验 fail-closed → 恒 eager 回退（engine 侧 graph_eager_fallbacks 计数）。
  解除需三步：gemm 稳定参数格改造（engine.rs，>20 行非接线改动——纪律条款触发，
  已停）；cudarc 按 cuda-13020 构建 + 13.2 运行时（节点参数读回符号）；引擎侧
  逐 launch KernelSpec 声明 + PtrUpdate 注册表。详见 bench/notes.md T-305 段。
- **CI ignored 清单（C4）**：全量补录（graph::ffi_tests ×3、graph_engine 新测试
  gpu.yml: graph-smoke、engine_smoke/fmha_prefill/dequant/gqa/gemm/sampler/
  arch 全部现存 ignore），checked-ignores.sh 转绿（此前红）。
