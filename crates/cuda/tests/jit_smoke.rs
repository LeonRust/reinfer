//! 012 C3：vec_add Jit 链路真机冒烟（差分 / 命中预算 / 确定性）。
//!
//! 运行（判定机；`--test-threads=1` 强制）：
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-12.8/bin/nvcc \
//! REINFER_JIT_CACHE=/tmp/reinfer-jit-smoke \
//! CUDA_VISIBLE_DEVICES=0 \
//! cargo test -p reinfer-cuda --features cuda --test jit_smoke -- \
//!     --ignored --test-threads=1
//! ```

#[cfg(feature = "cuda")]
mod smoke {
    use reinfer_core::{DType, DeviceId};
    use reinfer_cuda::jit_provider::{VecAddArgs, VecAddProvider};
    use reinfer_cuda::{CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy};
    use reinfer_kernels::{KernelProvider, OpConfig};

    const N: u32 = 1 << 20; // 1 Mi 元素（4 MiB/缓冲）

    fn nvcc_arch() -> String {
        std::env::var("REINFER_CUDA_ARCH").unwrap_or_else(|_| "sm_120a".into())
    }

    fn host_data(seed: f32) -> (Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..N as usize)
            .map(|i| seed * (i as f32 * 0.001).sin() + (i % 7) as f32 * 0.25)
            .collect();
        let b: Vec<f32> =
            (0..N as usize).map(|i| (i as f32 * 0.0005).cos() - (i % 3) as f32 * 0.5).collect();
        (a, b)
    }

    fn fill(buf: &HostBuffer, data: &[f32]) {
        // SAFETY：pinned host 内存由结构持有；长度为 F32 计数 *4
        unsafe {
            let s = core::slice::from_raw_parts_mut(buf.as_ptr() as *mut f32, data.len());
            s.copy_from_slice(data);
        }
    }

    fn snapshot(buf: &HostBuffer, n: usize) -> Vec<f32> {
        // SAFETY：只读（同上）
        unsafe { core::slice::from_raw_parts(buf.as_ptr() as *const f32, n).to_vec() }
    }

    fn make_provider(ctx: &CudaContext, cache: &std::path::Path) -> VecAddProvider {
        let stream = CudaStream::new(ctx.device_id()).expect("stream");
        VecAddProvider::new(&nvcc_arch(), Some(cache.to_path_buf()), stream).expect("provider")
    }

    #[test]
    #[ignore = "gpu.yml: smoke / l2-jit"]
    fn vec_add_diff_and_determinism() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let cache = std::env::temp_dir().join("reinfer-jit-c3-diff");
        let _ = std::fs::remove_dir_all(&cache);

        let (a, b) = host_data(0.5);
        let hs = HostBuffer::alloc(N as usize * 4).expect("hs");
        let hb = HostBuffer::alloc(N as usize * 4).expect("hb");
        let hout = HostBuffer::alloc(N as usize * 4).expect("hout");
        fill(&hs, &a);
        fill(&hb, &b);

        let da = DeviceBuffer::alloc(dev, N as usize * 4).expect("da");
        let db = DeviceBuffer::alloc(dev, N as usize * 4).expect("db");
        let dout = DeviceBuffer::alloc(dev, N as usize * 4).expect("dout");
        copy(&mut MemRef::Device(&da), &MemRef::Host(&hs), N as usize * 4, None).expect("h2d a");
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), N as usize * 4, None).expect("h2d b");

        let provider = make_provider(&ctx, &cache);
        let cfg = OpConfig {
            op: "vec_add",
            device: dev,
            in_dt: DType::F32,
            out_dt: DType::F32,
            head_dim: 0,
            batch: 1,
            seq: 0,
        };
        assert!(provider.matches(&cfg));
        let args = VecAddArgs {
            a: da.as_ptr().cast::<f32>(),
            b: db.as_ptr().cast::<f32>(),
            out: dout.as_ptr().cast::<f32>() as *mut f32,
            n: N,
        };
        assert!(VecAddProvider::size_check(N, (&da, &db, &dout)));
        let mut args = args;
        eprintln!("[diff] launching kernel...");
        provider.launch(&cfg, &mut args).expect("launch1");
        eprintln!("[diff] kernel launched, syncing...");
        provider.sync_stream().expect("sync1");
        eprintln!("[diff] synced, copying back...");
        copy(&mut MemRef::Host(&hout), &MemRef::Device(&dout), N as usize * 4, None).expect("d2h");

        let got = snapshot(&hout, N as usize);
        let expect: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        // D7 容差（fp32 rtol/atol 逐项）
        let mut worst = 0.0f32;
        for (g, e) in got.iter().zip(&expect) {
            let d = (g - e).abs();
            let tol = 1e-5 * e.abs().max(f32::EPSILON) + 1e-7;
            assert!(d <= tol, "mismatch at {g} vs {e}");
            worst = worst.max(d);
        }
        eprintln!("vec_add: worst abs err = {worst:e}");

        // 确定性：同 kernel 再跑一次应 bit-exact
        let mut args2 = VecAddArgs {
            a: da.as_ptr().cast::<f32>(),
            b: db.as_ptr().cast::<f32>(),
            out: dout.as_ptr().cast::<f32>() as *mut f32,
            n: N,
        };
        provider.launch(&cfg, &mut args2).expect("launch2");
        provider.sync_stream().expect("sync2");
        let hout2 = HostBuffer::alloc(N as usize * 4).expect("hout2");
        copy(&mut MemRef::Host(&hout2), &MemRef::Device(&dout), N as usize * 4, None)
            .expect("d2h2");
        assert_eq!(snapshot(&hout2, N as usize), got, "determinism mismatch");
    }

    #[test]
    #[ignore = "gpu.yml: smoke / l2-jit"]
    fn jit_cache_reload_budget() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let cache = std::env::temp_dir().join("reinfer-jit-c3-budget");
        let _ = std::fs::remove_dir_all(&cache);

        let t0 = std::time::Instant::now();
        let p1 = make_provider(&ctx, &cache);
        let first = t0.elapsed();
        eprintln!("first provider build: {first:?}");

        let t1 = std::time::Instant::now();
        let _p2 = make_provider(&ctx, &cache);
        let hit = t1.elapsed();
        eprintln!("second provider (cache hit): {hit:?}");
        assert!(hit < std::time::Duration::from_millis(50), "hit took {hit:?}");
        assert_eq!(p1.arch(), nvcc_arch());
    }
}
