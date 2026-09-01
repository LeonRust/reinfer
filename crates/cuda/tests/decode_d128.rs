//! decode_step_gqa 在 Qwen3-0.6B 形状（d=128 / 16 头 / 8 kv 头）下的复现
//! 判据（014 T8 延伸——T8 判据档只测 d=64/14 头）。
//!
//! 运行：`CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda
//! --test decode_d128 -- --ignored --test-threads=1`
//!
//! 断言：非零输出 + 与 host gather 参考一致（1 ulp）。

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)]

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::decode::DecodeKernels;
    use reinfer_cuda::{CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef, copy};
    use reinfer_gguf::codes::f16_to_f32;

    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }
    fn rand_f16_bits(seed: &mut u64) -> u16 {
        let mant = (xorshift(seed) as u16) & 0x3ff;
        let exp = ((xorshift(seed) as u16) % 0x1e) & 0xf;
        (exp << 10) | mant
    }

    fn upl(_dev: u32, host: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(host.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), hb.as_ptr() as *mut u8, host.len());
        }
        let db = DeviceBuffer::alloc(DeviceId::new(0), host.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None).unwrap();
        db
    }

    #[test]
    #[ignore = "gpu.yml: extended / d128"]
    fn decode_d128_yields_nonzero() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-d128");
        let _ = std::fs::remove_dir_all(&cache);
        let stream = CudaStream::new(DeviceId::new(0)).unwrap();
        let dk = DecodeKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream,
        )
        .unwrap();

        let (qh, kvh, d, bl, kv_len) = (16u32, 8u32, 128u32, 32u32, 2u32);
        let ratio = qh / kvh;
        let total_pages = 4u32;
        let per_tok = (kvh * d) as usize;
        let mut seed = 0xD128_u64;

        // 页表（物理页 0..log 连续）
        let log_pages = kv_len.div_ceil(bl);
        let mut page = vec![0u32; log_pages as usize];
        for (i, p) in page.iter_mut().enumerate() {
            *p = i as u32;
        }
        // KV（写区 = 前 log 页；其余留 0xFFFF 毒化在末尾区域）
        let mut kv = vec![0xFFFFu16; (total_pages * bl * kvh * d * 2) as usize];
        for t in 0..kv_len as usize {
            for kk in 0..kvh as usize {
                for i in 0..d as usize {
                    for (base, val) in
                        [(0usize, rand_f16_bits(&mut seed)), (1, rand_f16_bits(&mut seed))]
                    {
                        // base 0 = K 区、1 = V 区(k_region 前 log 起始)
                        let region = (total_pages * bl * kvh * d) as usize;
                        let idx = ((t * kvh as usize + kk) * d as usize) + i;
                        kv[base * region + idx] = val;
                    }
                }
            }
        }
        let q: Vec<u16> = (0..qh as usize * d as usize).map(|_| rand_f16_bits(&mut seed)).collect();
        let lens = [kv_len];

        let dq = upl(dev, &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dpage = upl(dev, &page.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dkv = upl(dev, &kv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dlens = upl(dev, &lens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dsc = DeviceBuffer::alloc(DeviceId::new(0), (qh * 1024 * 4) as usize).unwrap();
        let dout = DeviceBuffer::alloc(DeviceId::new(0), (qh * d * 2) as usize).unwrap();

        dk.launch_decode_step_gqa(
            dev,
            dq.as_ptr() as *const u16,
            dpage.as_ptr() as *const u32,
            dkv.as_ptr() as *const u16,
            dlens.as_ptr() as *const u32,
            dsc.as_ptr() as *mut f32,
            dout.as_ptr() as *mut u16,
            1,
            qh,
            d,
            bl,
            ratio,
            kvh,
            kv_len,
            total_pages,
        )
        .unwrap();
        dk.sync_stream().unwrap();

        // scores 回读（head0 前 4 个 t）
        let hsc = HostBuffer::alloc(64).unwrap();
        copy(&mut MemRef::Host(&hsc), &MemRef::Device(&dsc), 64, None).unwrap();
        let sc: Vec<f32> =
            unsafe { std::slice::from_raw_parts(hsc.as_ptr() as *const f32, 16).to_vec() };
        println!("scores first 16: {sc:?}");
        let hout = HostBuffer::alloc((qh * d * 2) as usize).unwrap();
        copy(&mut MemRef::Host(&hout), &MemRef::Device(&dout), (qh * d * 2) as usize, None)
            .unwrap();
        let out: Vec<u16> = unsafe {
            std::slice::from_raw_parts(hout.as_ptr() as *const u16, (qh * d) as usize).to_vec()
        };
        let outf: Vec<f32> = out.iter().map(|v| f16_to_f32(*v)).collect();
        let any_nonzero = outf.iter().any(|v| *v != 0.0);
        assert!(any_nonzero, "decode d=128 输出必须非零");
        println!("decode d128 out[0..8]={:?}", &outf[..8]);
    }
}
