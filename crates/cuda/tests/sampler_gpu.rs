//! 006-2 T3C：GPU sampler 链真机冒烟（GpuSamplerChain + sampler_kernel.cu）。
//!
//! 判据面（D2 三层契约）：
//! ① temp=0 硬件 argmax，tie-break=LastMax（与 CPU 适配器 bit-identical，
//!    记录偏差经 TokenOut::tie_break）；rng 不得被消费；
//! ② temp>0 与 CPU 路径同分布（gumbel-max 技巧）；(i,p,v) 纯函数 RNG——
//!    每步恰消费一个 RngState u64；同 seed 逐 token 确定；
//! ③ 过滤器精确性：top_k（64 轮 pair-key 二分，含边界 tie 最大 tid 优先）、
//!    min_p（只留 max_prob*min_p 之上）、组合面；
//! ④ NotSupported（bad_words/gumbel）原子回退：不消费 rng、不推进自身状态；
//! ⑤ 选择器计数器 sampler_gpu / eager_fallback；
//! ⑥ 单 launch 门：launch_count == 成功采样数。
//!
//! 运行（本机必须 13.2 nvcc——12.6 编译产物输出全 0，项目已知）：
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc \
//! REINFER_JIT_CACHE=/tmp/reinfer-jit-sampler \
//! cargo test -p reinfer-cuda --features cuda --test sampler_gpu -- \
//!     --ignored --test-threads=1
//! ```

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
#![allow(clippy::print_stdout)] // 冒烟输出

mod smoke {
    use reinfer_core::DeviceId;
    use reinfer_cuda::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
    use reinfer_cuda::sampler::GpuSamplerChain;
    use reinfer_cuda::{CudaContext, CudaStream};
    use reinfer_kernels::{
        CpuSamplerChain, LogitsView, RngState, SampleError, SamplerChain, SamplerImpl,
        SamplerParams, TieBreak, UnsupportedParam, select_sampler,
    };
    use std::sync::{Arc, Mutex};

    fn setup() -> (CudaContext, DeviceId) {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).expect("stream");
        let _ = stream.synchronize().expect("sync");
        (ctx, dev)
    }

    fn new_chain(dev: DeviceId) -> GpuSamplerChain {
        GpuSamplerChain::new(
            dev,
            &reinfer_cuda::arch::resolve_arch().expect("arch"),
            Some(std::env::temp_dir().join("reinfer-jit-sampler")),
        )
        .expect("sampler chain")
    }

    /// Mock logits view: real H2D upload + D2H copy closure (the CPU adapter
    /// reads via `to_host`, the GPU kernel reads the device buffer directly).
    fn make_view(dev: DeviceId, logits: &[f32]) -> LogitsView {
        let n = logits.len();
        let db = DeviceBuffer::alloc(dev, n * 4).expect("dev logits");
        let hb = HostBuffer::alloc(n * 4).expect("host logits");
        // SAFETY: pinned host buffer holds exactly n f32s.
        unsafe {
            std::ptr::copy_nonoverlapping(
                logits.as_ptr() as *const u8,
                hb.as_ptr() as *mut u8,
                n * 4,
            );
        }
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), n * 4, None).expect("h2d");
        let ptr = db.as_ptr() as usize;
        let bytes = db.size();
        // The copy closure must be Send+Sync; DeviceBuffer is only Send, so
        // keep it behind an Arc<Mutex<..>> (single-threaded tests, no contention).
        let shared = Arc::new(Mutex::new(Some(db)));
        let read = shared.clone();
        LogitsView::new(dev, reinfer_kernels::DeviceBuffer::new(ptr, bytes), n, move || {
            let guard = read.lock().unwrap();
            let db = guard.as_ref().unwrap();
            let out = HostBuffer::alloc(bytes).unwrap();
            copy(&mut MemRef::Host(&out), &MemRef::Device(db), bytes, None).unwrap();
            // SAFETY: D2H snapshot of the n f32 logits.
            unsafe { std::slice::from_raw_parts(out.as_ptr() as *const f32, n).to_vec() }
        })
    }

    const MOCK: [f32; 16] =
        [0.1, -2.0, 1.5, 3.0, 0.5, -1.0, 2.0, 0.0, 1.0, 1.8, -0.5, 0.25, -3.0, 4.0, 0.75, -1.5];

    /// ① temp=0：LastMax tie-break（[1.0, 2.0, 2.0, 1.5] → token 2），rng 不消费。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn greedy_last_max_tie() {
        let (_ctx, dev) = setup();
        let view = make_view(dev, &[1.0f32, 2.0, 2.0, 1.5]);
        let mut gpu = new_chain(dev);
        let params = SamplerParams::default(); // temp=0 → greedy
        let mut rng = RngState::new(1);
        let out = gpu.sample(&view, &params, &mut rng).expect("gpu sample");
        assert_eq!(out.token, 2, "LastMax over equal maxima at tids 1,2");
        assert_eq!(out.tie_break, TieBreak::LastMax);
        assert_eq!(rng, RngState::new(1), "temp=0 must not consume RngState");
        // CPU reference (same input, same semantics).
        let mut cpu = CpuSamplerChain::new(&params).expect("cpu chain");
        assert_eq!(cpu.sample(&view, &params, &mut rng).expect("cpu sample").token, 2);
    }

    /// ① 位一致矩阵：greedy / top_k / top_p / repeat+top_k+top_p 各 32 步，
    /// GPU 与 CPU 适配器逐 token 相同（temp=0 无 RNG，rng 共享不消费）。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn bit_identical_matrix_vs_cpu() {
        let (_ctx, dev) = setup();
        let view = make_view(dev, &MOCK);
        let cases = [
            SamplerParams::default(),
            SamplerParams { top_k: Some(3), ..Default::default() },
            SamplerParams { top_p: Some(0.9), ..Default::default() },
            SamplerParams { top_k: Some(3), top_p: Some(0.9), ..Default::default() },
            SamplerParams { repeat_penalty: Some(1.1), repeat_last_n: 8, ..Default::default() },
            SamplerParams {
                repeat_penalty: Some(1.1),
                repeat_last_n: 8,
                top_k: Some(3),
                top_p: Some(0.9),
                ..Default::default()
            },
        ];
        for params in cases {
            // Fresh GPU chain per case: penalty history must start empty like
            // the fresh CPU adapter (repeat_penalty windows are history-sized).
            let mut gpu = new_chain(dev);
            let mut cpu = CpuSamplerChain::new(&params).expect("cpu chain");
            let mut rng = RngState::new(42);
            for step in 0..32 {
                let t_cpu = cpu.sample(&view, &params, &mut rng).expect("cpu sample");
                let t_gpu = gpu.sample(&view, &params, &mut rng).expect("gpu sample");
                assert_eq!(
                    t_gpu.token, t_cpu.token,
                    "step {step} (params {params:?}): GPU drifted from CPU"
                );
                assert_eq!(t_gpu.tie_break, TieBreak::LastMax);
            }
        }
    }

    /// ① top-k 边界 tie：GPU 恒 LastMax（spec r2 pin）；CPU 在过滤器面有
    /// llm-samplers stable-sort 怪癖（truncate 后 greedy 取 first() = 边界
    /// tie 组的最小 tid）——记录偏差，非 GPU 回退。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn near_ties_and_topk_boundary() {
        let (_ctx, dev) = setup();
        // top-k 边界 tie：top-2 = {(3.0,t0),(2.0,t1)}（tie 组小 tid 在前，
        // stable-sort 语义）→ 唯一最大值 → 双方都出 0
        let view = make_view(dev, &[3.0f32, 2.0, 2.0, 2.0, 1.0]);
        let params = SamplerParams { top_k: Some(2), ..Default::default() };
        let mut gpu = new_chain(dev);
        let mut cpu = CpuSamplerChain::new(&params).expect("cpu chain");
        let mut rng = RngState::new(1);
        assert_eq!(gpu.sample(&view, &params, &mut rng).unwrap().token, 0);
        assert_eq!(cpu.sample(&view, &params, &mut rng).unwrap().token, 0);
        // 全局最大值 tie + 过滤器启用：GPU = LastMax（2）；CPU = llm-samplers
        // stable sort 后 greedy first() = 最小 tid（1）——记录偏差（r2 pin
        // LastMax，CPU 的过滤器面 FirstMax 是遗留链怪癖）
        let view2 = make_view(dev, &[1.0f32, 2.0, 2.0, 1.5]);
        let mut gpu2 = new_chain(dev);
        let mut cpu2 = CpuSamplerChain::new(&params).expect("cpu chain");
        assert_eq!(gpu2.sample(&view2, &params, &mut rng).unwrap().token, 2, "GPU pins LastMax");
        assert_eq!(
            cpu2.sample(&view2, &params, &mut rng).unwrap().token,
            1,
            "CPU stable-sort artifact (recorded deviation)"
        );
    }

    /// ③ 过滤器精确性：top_k 只从 top-k 中采样；min_p 只留 max_prob*min_p 之上。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn filters_support() {
        let (_ctx, dev) = setup();
        // top_k=2 over [1.0, 10.0, 9.0, 0.5] → 只可能出 token 1/2
        let view = make_view(dev, &[1.0f32, 10.0, 9.0, 0.5]);
        let mut gpu = new_chain(dev);
        let params = SamplerParams { temperature: 1.0, top_k: Some(2), ..Default::default() };
        let mut rng = RngState::new(7);
        let mut seen = [0u32; 4];
        for _ in 0..4000 {
            let t = gpu.sample(&view, &params, &mut rng).expect("gpu sample").token;
            seen[t as usize] += 1;
        }
        assert_eq!(seen[0], 0, "token 0 filtered by top_k");
        assert_eq!(seen[3], 0, "token 3 filtered by top_k");
        assert!(seen[1] > 0 && seen[2] > 0, "both top-k survivors drawn: {seen:?}");

        // min_p=0.5（相对阈值：prob >= 0.5*max_prob）over [2.0, 0.2, 0.1, 0.1]
        // → 只有 token 0 存活（softmax: 0.68 vs 0.11/0.10/0.10）
        let view2 = make_view(dev, &[2.0f32, 0.2, 0.1, 0.1]);
        let mut gpu2 = new_chain(dev);
        let p2 = SamplerParams { temperature: 1.0, min_p: Some(0.5), ..Default::default() };
        let mut rng2 = RngState::new(7);
        for _ in 0..4000 {
            assert_eq!(gpu2.sample(&view2, &p2, &mut rng2).unwrap().token, 0, "min_p prune");
        }
    }

    /// ② temp>0 同分布：gumbel-max 抽取 = 幸存集上的 renormalized softmax
    /// categorical（与 CPU 路径同分布，D2 tier-2 承诺）；8000 次经验频率
    /// 对照解析 softmax。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn distribution_matches_softmax() {
        let (_ctx, dev) = setup();
        let logits = [0.5f32, 1.0, 2.0, 1.5, 0.2];
        let temp = 0.8f32;
        let view = make_view(dev, &logits);
        // analytic expected: softmax(logits/temp)
        let scaled: Vec<f32> = logits.iter().map(|l| l / temp).collect();
        let m = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|l| (l - m).exp()).collect();
        let z: f32 = exps.iter().sum();
        let expected: Vec<f32> = exps.iter().map(|e| e / z).collect();

        let mut gpu = new_chain(dev);
        let params = SamplerParams { temperature: temp, ..Default::default() };
        let mut rng = RngState::new(1234);
        let n = 8000;
        let mut obs = [0u32; 5];
        for _ in 0..n {
            let t = gpu.sample(&view, &params, &mut rng).expect("gpu sample").token;
            obs[t as usize] += 1;
        }
        println!("expected {expected:?} observed {obs:?}");
        for (i, &e) in expected.iter().enumerate() {
            if e as f64 >= 0.05 {
                let frac = obs[i] as f64 / n as f64;
                assert!(
                    (frac - e as f64).abs() < 0.04,
                    "token {i}: observed {frac:.4} vs expected {e:.4}"
                );
            }
        }
    }

    /// ② 确定性：同 seed 同输入 → 逐 token 相同（temp>0 面）。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn determinism_same_seed() {
        let (_ctx, dev) = setup();
        let view = make_view(dev, &MOCK);
        let params = SamplerParams {
            temperature: 0.9,
            top_k: Some(5),
            top_p: Some(0.9),
            ..Default::default()
        };
        let mut a = new_chain(dev);
        let mut b = new_chain(dev);
        let mut rng_a = RngState::new(99);
        let mut rng_b = RngState::new(99);
        for step in 0..64 {
            let ta = a.sample(&view, &params, &mut rng_a).unwrap().token;
            let tb = b.sample(&view, &params, &mut rng_b).unwrap().token;
            assert_eq!(ta, tb, "step {step}: same seed diverged");
        }
    }

    /// ② RNG 消费：temp=0 零消费；temp>0 每步恰一个 u64。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn rng_consumption_one_per_step() {
        let (_ctx, dev) = setup();
        let view = make_view(dev, &[0.5f32, 1.0, 2.0, 1.5, 0.2]);
        let mut gpu = new_chain(dev);
        // temp=0: never consumed
        let p0 = SamplerParams::default();
        let mut rng0 = RngState::new(5);
        gpu.sample(&view, &p0, &mut rng0).unwrap();
        assert_eq!(rng0, RngState::new(5), "temp=0 must not consume RngState");
        // temp>0: exactly one u64 per step (10 steps → 10 advances)
        let p1 = SamplerParams { temperature: 1.0, ..Default::default() };
        let mut rng1 = RngState::new(5);
        let mut ref1 = RngState::new(5);
        for _ in 0..10 {
            let _ = ref1.mix().next_u64();
            gpu.sample(&view, &p1, &mut rng1).unwrap();
        }
        assert_eq!(rng1, ref1, "temp>0: exactly one RngState u64 per step");
    }

    /// ④ NotSupported 原子回退：bad_words/gumbel → 显式错误；rng 与自身
    /// 状态（history/launch_count）均不得被推进。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn not_supported_preserves_rng_and_state() {
        let (_ctx, dev) = setup();
        let view = make_view(dev, &[1.0f32, 2.0, 3.0]);
        let mut gpu = new_chain(dev);
        let p_bad =
            SamplerParams { bad_words: vec![vec![1, 2]], temperature: 1.0, ..Default::default() };
        let p_gum = SamplerParams { gumbel: true, ..Default::default() };
        let mut rng = RngState::new(5);
        let err1 = gpu.sample(&view, &p_bad, &mut rng).expect_err("bad_words rejected");
        assert_eq!(err1, SampleError::NotSupported(UnsupportedParam::BadWords));
        let err2 = gpu.sample(&view, &p_gum, &mut rng).expect_err("gumbel rejected");
        assert_eq!(err2, SampleError::NotSupported(UnsupportedParam::Gumbel));
        assert_eq!(rng, RngState::new(5), "NotSupported must not consume RngState");
        assert_eq!(gpu.launch_count(), 0, "no launch on NotSupported");
        assert_eq!(gpu.history_len(), 0, "no history push on NotSupported");
    }

    /// ⑤ 选择器计数器：GPU 成功 → sampler_gpu；双方都不覆盖 → eager_fallback。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn fallback_selector_counts() {
        let (_ctx, dev) = setup();
        let view = make_view(dev, &[1.0f32, 2.0, 3.0]);
        let gpu = new_chain(dev);
        let cpu = CpuSamplerChain::new(&SamplerParams::default()).expect("cpu chain");
        let mut chain = select_sampler(vec![Box::new(gpu), Box::new(cpu)]).expect("selector");
        assert_eq!(chain.variant(), SamplerImpl::GpuSampler);
        let mut rng = RngState::new(1);
        // covered surface → GPU 成功计数
        let out = chain.sample(&view, &SamplerParams::default(), &mut rng).unwrap();
        assert_eq!(out.token, 2, "greedy argmax over [1.0, 2.0, 3.0]");
        assert_eq!(chain.counters().unwrap().sampler_gpu(), 1);
        assert_eq!(chain.counters().unwrap().eager_fallback(), 0);
        // 双方都不覆盖（bad_words）→ eager_fallback 计数并传播错误
        let p_bad = SamplerParams { bad_words: vec![vec![1]], ..Default::default() };
        let err = chain.sample(&view, &p_bad, &mut rng).expect_err("propagated");
        assert_eq!(err, SampleError::NotSupported(UnsupportedParam::BadWords));
        assert_eq!(chain.counters().unwrap().sampler_gpu(), 1);
        assert_eq!(chain.counters().unwrap().eager_fallback(), 1);
        assert_eq!(chain.counters().unwrap().padding_ratio(), 0.5);
    }

    /// ⑥ 单 launch 门：launch_count == 成功采样数（每步恰一次内核 launch）。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn launch_count_matches_samples() {
        let (_ctx, dev) = setup();
        let view = make_view(dev, &[0.5f32, 1.0, 2.0, 1.5, 0.2]);
        let mut gpu = new_chain(dev);
        let params = SamplerParams { temperature: 0.9, top_k: Some(3), ..Default::default() };
        let mut rng = RngState::new(3);
        for _ in 0..16 {
            gpu.sample(&view, &params, &mut rng).expect("sample");
        }
        assert_eq!(gpu.launch_count(), 16, "one kernel launch per sample");
    }

    /// 全非有限 logits → NoToken（fail-closed；CPU 侧同语义）。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn no_token_when_all_nonfinite() {
        let (_ctx, dev) = setup();
        let view = make_view(dev, &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        let mut gpu = new_chain(dev);
        let err = gpu
            .sample(&view, &SamplerParams::default(), &mut RngState::new(1))
            .expect_err("all non-finite must yield NoToken");
        assert_eq!(err, SampleError::NoToken);
    }

    /// GPU-only 面（CPU 适配器显式拒绝）：logit_bias / frequency_penalty /
    /// presence_penalty 翻转 greedy 结果。
    #[test]
    #[ignore = "gpu.yml: sampler-smoke"]
    fn bias_and_penalties_flip_greedy() {
        let (_ctx, dev) = setup();
        // logit_bias: -5.0 on token 2 flips argmax to token 1
        let view = make_view(dev, &[0.0f32, 1.0, 2.0, 0.0]);
        let mut gpu = new_chain(dev);
        let p_bias = SamplerParams { logit_bias: vec![(2, -5.0)], ..Default::default() };
        let t = gpu.sample(&view, &p_bias, &mut RngState::new(1)).unwrap().token;
        assert_eq!(t, 1, "bias -5 on token 2 must flip argmax to token 1");

        // frequency penalty: first sample = LastMax over [1.0,2.0,2.0] → 2;
        // then token 2 in the window is penalized by -1.0 → next = 1
        let view2 = make_view(dev, &[1.0f32, 2.0, 2.0]);
        let mut gpu2 = new_chain(dev);
        let p0 = SamplerParams::default();
        assert_eq!(gpu2.sample(&view2, &p0, &mut RngState::new(1)).unwrap().token, 2);
        let p_freq = SamplerParams { frequency_penalty: Some(1.0), ..Default::default() };
        let t2 = gpu2.sample(&view2, &p_freq, &mut RngState::new(1)).unwrap().token;
        assert_eq!(t2, 1, "token 2 penalized by cnt*freq = 1.0");

        // presence penalty: -0.5 on any token present in the window
        let view3 = make_view(dev, &[1.0f32, 2.0, 2.0]);
        let mut gpu3 = new_chain(dev);
        gpu3.sample(&view3, &p0, &mut RngState::new(1)).unwrap(); // history = [2]
        let p_pres = SamplerParams { presence_penalty: Some(0.5), ..Default::default() };
        let t3 = gpu3.sample(&view3, &p_pres, &mut RngState::new(1)).unwrap().token;
        assert_eq!(t3, 1, "token 2 penalized by presence 0.5");
    }
}
