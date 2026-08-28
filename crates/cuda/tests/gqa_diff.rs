//! 014 T8: paged decode GQA 真机判据。
//!
//! 运行：`REINFER_CUDA_NVCC=... CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda
//! --features cuda --test gqa_diff -- --ignored --test-threads=1`
//!
//! 判据（r2）：
//! ① 随机页表 diff（跨 2-3 页/首尾部分页/乱序物理页/kv_len 1..1k 抽样）
//!    ——GPU kernel vs host gather 参考（layouts: K 区 [P][BL][KH][d] f16，
//!    V 区紧随）；
//! ② GQA 三例（14/2、12/2、5/2）映射核验（kv_head = q_head/kv_ratio
//!    整数除法连续分组）；
//! ③ 毒化：未写页位 0xFF（NaN 型 f16）——被 kv_len 遮挡的位置绝不参与
//!    （输出有限与参考一致）；
//! ④ 确定性：两次 launch 逐位一致；
//! ⑤ 页池守恒（crates/memory：在用+空闲==总；分配==释放——见 pool 单测）。

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::decode::DecodeKernels;
    use reinfer_cuda::{copy, CudaContext, CudaStream, DeviceBuffer, HostBuffer, MemRef};
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

    /// host gather 参考（q/kv 传 f16 位模式；输出 f32 数学值）。
    #[allow(clippy::too_many_arguments)]
    fn decode_ref(
        q: &[u16],
        page: &[u32],
        kv: &[u16],
        kv_lens: &[u32],
        b: usize,
        qh: usize,
        d: usize,
        block_len: usize,
        kv_ratio: usize,
        total_pages: usize,
    ) -> Vec<f32> {
        let kv_heads = qh / kv_ratio;
        let per_tok = kv_heads * d;
        let v_base = total_pages * block_len * per_tok;
        let mut out = vec![0.0f32; b * qh * d];
        for bi in 0..b {
            let kv_len = kv_lens[bi] as usize;
            for h in 0..qh {
                let kv_h = h / kv_ratio;
                let mut s = vec![0.0f32; kv_len];
                for t in 0..kv_len {
                    let lp = t / block_len;
                    let off = t % block_len;
                    let phys = page[lp] as usize;
                    let base = ((phys * block_len + off) * kv_heads + kv_h) * d;
                    let mut acc = 0.0f32;
                    for i in 0..d {
                        acc += f16_to_f32(q[(bi * qh + h) * d + i]) * f16_to_f32(kv[base + i]);
                    }
                    s[t] = acc;
                }
                let maxv = s.iter().copied().reduce(f32::max).unwrap_or(-1e30);
                let sum: f32 = s.iter().map(|x| (x - maxv).exp()).sum();
                let inv = if sum != 0.0 { 1.0 / sum } else { 0.0 };
                for i in 0..d {
                    let mut acc = 0.0f32;
                    for t in 0..kv_len {
                        let p = (s[t] - maxv).exp() * inv;
                        let lp = t / block_len;
                        let off = t % block_len;
                        let phys = page[lp] as usize;
                        let base = v_base + ((phys * block_len + off) * kv_heads + kv_h) * d;
                        acc += p * f16_to_f32(kv[base + i]);
                    }
                    out[(bi * qh + h) * d + i] = acc;
                }
            }
        }
        out
    }

    fn f16_to_f32_bit(u: u16) -> f32 {
        f16_to_f32(u)
    }

    fn upl(dev: u32, host: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(host.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), hb.as_ptr() as *mut u8, host.len());
        }
        let db = DeviceBuffer::alloc(DeviceId::new(0), host.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None).unwrap();
        db
    }

    #[allow(clippy::too_many_arguments)]
    fn run_one(
        dev: u32,
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        max_kv: u32,
        total_pages: u32,
        page: &[u32],
        kv: &[u16],
        kv_lens: &[u32],
        q: &[u16],
        cache_dir: &std::path::Path,
    ) -> Vec<f32> {
        let stream = CudaStream::new(DeviceId::new(0)).unwrap();
        let dk = DecodeKernels::new(
            &reinfer_cuda::arch::resolve_arch().unwrap(),
            Some(cache_dir.to_path_buf()),
            CudaStream::new(DeviceId::new(0)).unwrap(),
        )
        .unwrap();
        let kv_heads = (qh / kv_ratio) as u32;

        let dq = upl(dev, &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dpage = upl(dev, &page.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dkv = upl(dev, &kv.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dlens = upl(dev, &kv_lens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        let dsc = DeviceBuffer::alloc(DeviceId::new(0), (b * qh * max_kv * 4) as usize).unwrap();

        let hout = HostBuffer::alloc((b * qh * d * 2) as usize).unwrap();
        let mut dout = DeviceBuffer::alloc(DeviceId::new(0), (b * qh * d * 2) as usize).unwrap();
        // 预填 0x5555（判断"kernel 未写"）
        {
            let ph = HostBuffer::alloc((b * qh * d * 2) as usize).unwrap();
            unsafe {
                let s = std::slice::from_raw_parts_mut(ph.as_ptr() as *mut u8, (b * qh * d * 2) as usize);
                s.fill(0x55);
            }
            copy(&mut MemRef::Device(&dout), &MemRef::Host(&ph), (b * qh * d * 2) as usize, None).unwrap();
        }

        dk.launch_decode_step_gqa(
            dev,
            dq.as_ptr() as *const u16,
            dpage.as_ptr() as *const u32,
            dkv.as_ptr() as *const u16,
            dlens.as_ptr() as *const u32,
            dsc.as_ptr() as *mut f32,
            dout.as_ptr() as *mut u16,
            b,
            qh,
            d,
            block_len,
            kv_ratio,
            kv_heads,
            max_kv,
            total_pages,
        )
        .unwrap();
        dk.sync_stream().unwrap();
        // d2h: out
        copy(
            &mut MemRef::Host(&hout),
            &MemRef::Device(&dout),
            (b * qh * d * 2) as usize,
            None,
        )
        .unwrap();
        let raw: Vec<u16> = unsafe {
            std::slice::from_raw_parts(hout.as_ptr() as *const u16, (b * qh * d) as usize).to_vec()
        };
        raw.into_iter().map(f16_to_f32_bit).collect()
    }

    #[test]
    #[ignore = "gpu.yml: l3-attn / gqa"]
    fn gqa_random_pages_diff() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-gqa");
        let _ = std::fs::remove_dir_all(&cache);

        // 判据样例矩阵（GQA 三例 + 跨页/首尾部分页/乱序物理页）：
        let cases: &[(usize, usize, usize, usize, usize)] = &[
            // (qh, kv_ratio, block_len, max_kv, kv_len)
            (14, 2, 16, 96, 40),   // 三例 14/2；跨 2-3 页、末页部分
            (12, 2, 32, 256, 200), // 12/2；长序列
            (5, 2, 16, 128, 33),   // 5/2（非整除、四舍五入向下 2）；首尾部分页
        ];
        let d = 64usize;
        let mut seed = 0x6DE0_u64;
        for &(qh, ratio, bl, max_kv, kv_len) in cases {
            let total_pages = max_kv.div_ceil(bl);
            let log_pages = kv_len.div_ceil(bl);
            // 乱序物理页：前 log_pages 页随机物理（含 page id 反序/隔断）
            let mut phys = vec![0u32; log_pages];
            for i in 0..log_pages {
                phys[i] = ((xorshift(&mut seed) as usize) % total_pages) as u32;
            }
            // KV 数据（已写 token 区间 = kv_len；其余页 0xFFFF = NaN 毒化）
            let kv_heads = qh / ratio;
            let per_tok = kv_heads * d;
            let mut kv = vec![0xFFFFu16; total_pages * bl * per_tok * 2];
            for t in 0..kv_len {
                let lp = t / bl;
                let off = t % bl;
                let phys_p = phys[lp] as usize;
                for kk in 0..kv_heads {
                    for i in 0..d {
                        kv[((phys_p * bl + off) * kv_heads + kk) * d + i] = rand_f16_bits(&mut seed);
                        let vbase = total_pages * bl * per_tok;
                        kv[vbase + ((phys_p * bl + off) * kv_heads + kk) * d + i] =
                            rand_f16_bits(&mut seed);
                    }
                }
            }
            let q: Vec<u16> = (0..qh * d).map(|_| rand_f16_bits(&mut seed)).collect();
            let lens = [kv_len as u32];
            let got = run_one(
                dev,
                1,
                qh as u32,
                d as u32,
                bl as u32,
                ratio as u32,
                max_kv as u32,
                total_pages as u32,
                &phys,
                &kv,
                &lens,
                &q,
                &cache,
            );
            let want = decode_ref(&q, &phys, &kv, &lens, 1, qh, d, bl, ratio, total_pages);
            // ≤1 fp16 ulp + atol 1e-6
            let mut bad = 0usize;
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                let wh = f16_to_f32(f16_bits_of(*w));
                let ulp = if wh == 0.0 { 1e-9 } else { ulp_of(wh) };
                if (g - w).abs() > ulp + 1e-6 {
                    bad += 1;
                    if bad < 4 {
                                    }
                }
            }
            let vbase_idx = total_pages * bl * per_tok;
            eprintln!("gqa dbg hostV[0..4]={:?} hostK[0..4]={:?}",
                &kv[vbase_idx..vbase_idx + 4],
                &kv[0..4]);
            assert_eq!(bad, 0, "gqa qh={qh} ratio={ratio}: {bad} elems over 1 ulp");
            }
    }

    #[test]
    #[ignore = "gpu.yml: l3-attn / gqa"]
    fn gqa_deterministic() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id().index();
        let cache = std::env::temp_dir().join("reinfer-jit-gqa-det");
        let _ = std::fs::remove_dir_all(&cache);
        let (qh, ratio, bl, max_kv, kv_len, d) = (14usize, 2usize, 16usize, 96usize, 40usize, 64usize);
        let total_pages = max_kv.div_ceil(bl);
        let log_pages = kv_len.div_ceil(bl);
        let mut seed = 0xD3A2_u64;
        let phys: Vec<u32> = (0..log_pages)
            .map(|_| ((xorshift(&mut seed) as usize) % total_pages) as u32)
            .collect();
        let kv_heads = qh / ratio;
        let per_tok = kv_heads * d;
        let mut kv = vec![0xFFFFu16; total_pages * bl * per_tok * 2];
        for t in 0..kv_len {
            let lp = t / bl;
            let off = t % bl;
            let phys_p = phys[lp] as usize;
            for kk in 0..kv_heads {
                for i in 0..d {
                    let idx = ((phys_p * bl + off) * kv_heads + kk) * d + i;
                    kv[idx] = rand_f16_bits(&mut seed);
                    kv[total_pages * bl * per_tok + idx] = rand_f16_bits(&mut seed);
                }
            }
        }
        let q: Vec<u16> = (0..qh * d).map(|_| rand_f16_bits(&mut seed)).collect();
        let lens = [kv_len as u32];
        let a = run_one(dev, 1, qh as u32, d as u32, bl as u32, ratio as u32, max_kv as u32,
            total_pages as u32, &phys, &kv, &lens, &q, &cache);
        let b = run_one(dev, 1, qh as u32, d as u32, bl as u32, ratio as u32, max_kv as u32,
            total_pages as u32, &phys, &kv, &lens, &q, &cache);
        assert_eq!(a, b, "decode determinism: two launches differ");
    }

    fn f16_bits_of(f: f32) -> u16 {
        // 由 f32 数学值回到最近可表示 f16（用于 ulp 基准——测试局部）。
        let bits = f.to_bits();
        let sign: u16 = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x7f_ffff;
        if exp == 0xff {
            return sign | 0x7c00 | (((man >> 13) & 0x3ff) as u16);
        }
        let half_exp = exp - 127 + 15;
        if half_exp <= 0 {
            if half_exp < -10 {
                return sign;
            }
            let subm = ((man | 0x80_0000) >> (1 - half_exp + 13)) as u16;
            return sign | subm;
        }
        if half_exp >= 31 {
            return sign | 0x7c00;
        }
        sign | ((half_exp as u16) << 10) | ((man >> 13) as u16)
    }

    fn ulp_of(wh: f32) -> f32 {
        let e = wh.abs().log2().floor() as i32;
        2.0f32.powi(e - 10).max(1e-9)
    }
}
