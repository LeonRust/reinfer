//! S1-9: fused decode-step kernels — kernel-level bit-exact diff vs the
//! split sequences they replace (real machine), plus the engine A/B
//! (REINFER_FUSED on vs off).
//!
//! S1-10: the per-layer persistent kernel (decode_layer_fused_kernels.cu,
//! stages 0..8 in ONE launch) — kernel-level bit-exact diff vs the split
//! sequence and the engine A/B (REINFER_LAYER_FUSED on vs off).
//!
//! Gates:
//!   1. fused_layer_bit_exact_vs_split — one full layer through the fused
//!      sequence (p1 multi + p2_qkv + the fused flash [kv write +
//!      attention] + p1_o + p2_add_rms o + p1_gu + p2_gu_d [swiglu +
//!      down phase-1] + p2_add_rms down) vs the split sequence (7
//!      two-phase gemvs + casts + head norms + ropes + kv_write + flash
//!      decode + fused_add_rms + fused_cast_swiglu + add_cast + rms_norm)
//!      on the same random inputs: every output (q/k/v, the attention
//!      output, the residual stream x, both norm outputs, down) AND every
//!      phase-1 slab-partials segment must be bit-identical (0 ulp).
//!      The fused partials must match the single-plan
//!      `gemv_m1_f16f32` outputs exactly — that is the p1_multi
//!      contract (the p1_o phase-1 included).
//!   2. layer_fused_bit_exact_vs_split — the same reference: the S1-10
//!      persistent kernel (one launch: stage 0 gather + rms0 [li==0,
//!      embed row 0 = the x input row, identity gather] + stages 1..8)
//!      vs the split sequence on a fresh bufs pair: every output and
//!      every phase-1 partials segment bit-identical. The stage-8 norm
//!      output exercises the same rms code path as the stage-5 output
//!      (xn is one buffer in the engine, so the intermediate ffn-normed
//!      value is transient — the engine A/B gate covers the wiring).
//!   3. fused_determinism_double_run + layer_fused_determinism_double_run
//!      — two fused / layer-fused runs from identical inputs are
//!      bit-identical (no atomics / no nondeterminism).
//!   4. fused_engine_ab_bitwise + layer_fused_engine_ab_bitwise —
//!      engine-level: REINFER_FUSED on vs off and REINFER_LAYER_FUSED on
//!      vs off (both with REINFER_FUSED=on): 16 + 128 greedy decode
//!      tokens: per-step logits bit-identical, identical greedy text;
//!      the fused / layer-fused engine is deterministic across two
//!      separate loads (bit-level).
//!
//! Run (real machine; nvcc 13.2 — the sm_120a JIT rule):
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc \
//! REINFER_MODEL_DIR=/home/dora/.reinfer/models/Qwen/Qwen3-0.6B \
//! cargo test -p reinfer-cuda --features cuda --test fused_decode -- \
//!     --ignored --test-threads=1 --nocapture
//! ```

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // test assertions panic on failure
#![allow(clippy::print_stdout)] // smoke output

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
    use reinfer_cuda::decode::DecodeKernels;
    use reinfer_cuda::diff::DiffKernels;
    use reinfer_cuda::engine::{DecodeGemmPlans, DenseKernels, Engine, LayerGemmPlans};
    use reinfer_cuda::fused::{FusedDecodeKernels, FusedGeom};
    use reinfer_cuda::gemm::{GemmPlan, Jgemm};
    use reinfer_cuda::jit::{CtxGuard, JLib, KernelFn, launch_rows};
    use reinfer_cuda::layer_fused::{LayerFusedKernels, LayerFusedSpec};
    use reinfer_cuda::{CudaContext, CudaEvent, CudaStream};
    use reinfer_jit::compile::{compile_cubin, gencode_flags};
    use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
    use reinfer_tokenizer::Tokenizer;
    use std::ffi::c_void;

    // Qwen3-0.6B shapes.
    const H: usize = 1024;
    const NQK: usize = 2048; // q_heads(16) x d(128)
    const KVK: usize = 1024; // kv_heads(8) x d(128)
    const FFN: usize = 3072;
    const D: usize = 128;
    // Flash-decode geometry for the layer-level gates: one page of
    // BLOCK_LEN slots; the current step writes slot KV_LEN - 1 (the
    // earlier slots carry random pre-fill, identical on both paths).
    const BLOCK_LEN: usize = 256;
    const KV_LEN: usize = 17;
    const EPS: f32 = 1e-6;
    const ETA: f32 = 1e6;
    const POS: u32 = 42;

    /// Qwen3 q-k rope scale, computed exactly as the engine does.
    fn qscale() -> f32 {
        1.0 / (D as f32).sqrt()
    }

    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    /// Random f16 bit pattern (finite, dense value range).
    fn rand_f16_bits(seed: &mut u64) -> u16 {
        let mant = (xorshift(seed) as u16) & 0x3ff;
        let exp = ((xorshift(seed) as u16) % 0x1e) & 0xf;
        (exp << 10) | mant
    }

    fn upl(dev: u32, bytes: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(bytes.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), hb.as_ptr() as *mut u8, bytes.len());
        }
        let db = DeviceBuffer::alloc(DeviceId::new(dev), bytes.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), bytes.len(), None).unwrap();
        db
    }

    fn downl(dev: u32, n_bytes: usize) -> DeviceBuffer {
        DeviceBuffer::alloc(DeviceId::new(dev), n_bytes).unwrap()
    }

    fn read_u16(buf: &DeviceBuffer, n: usize) -> Vec<u16> {
        let hb = HostBuffer::alloc(n * 2).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(buf), n * 2, None).unwrap();
        unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const u16, n).to_vec() }
    }

    fn read_f32(buf: &DeviceBuffer, n: usize) -> Vec<f32> {
        let hb = HostBuffer::alloc(n * 4).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(buf), n * 4, None).unwrap();
        unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, n).to_vec() }
    }

    /// Device-pointer read of n f32 (the fused partials segments sit inside
    /// the unit's private buffer — only the geometry pointers are public).
    fn read_f32_ptr(dev: u32, ptr: *const f32, n: usize) -> Vec<f32> {
        let hb = HostBuffer::alloc(n * 4).unwrap();
        let _guard = CtxGuard::set_current(dev).unwrap();
        // SAFETY: test-only; the guard makes the device current; `ptr`
        // comes from the fused geometry (a live DeviceBuffer).
        let err = unsafe {
            cudarc::driver::sys::cuMemcpyDtoH_v2(
                hb.as_ptr() as *mut c_void,
                ptr as cudarc::driver::sys::CUdeviceptr,
                n * 4,
            )
        };
        assert_eq!(
            err,
            cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS,
            "cuMemcpyDtoH_v2: {err:?}"
        );
        unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, n).to_vec() }
    }

    fn bytes_u16(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// Standard JitCache load of gemm_m1.cu (cache key identical to
    /// Jgemm's, so the cubin is a cache hit) — the split-reference
    /// single-plan phase kernels.
    fn load_gemm_m1(arch: &str, cache_dir: Option<std::path::PathBuf>) -> JLib {
        let tc = probe_toolchain_for_arch(arch).unwrap();
        let src = KernelSource {
            name: "gemm_m1",
            src: include_str!("../kernels/gemm_m1.cu"),
            headers: vec![],
            flags: gencode_flags(arch).unwrap(),
            arch: arch.to_string(),
            toolchain_ver: tc.ver_line.clone(),
        };
        let cache = JitCache::open(cache_dir).unwrap();
        let key = JitKey::new(&src, &tc);
        let (_, cubin_path) = cache.build_once(&key, &src, || compile_cubin(&src, &tc)).unwrap();
        JLib::from_bytes(std::fs::read(cubin_path).unwrap()).unwrap()
    }

    /// Split-reference two-phase gemv (single-plan kernels): writes the
    /// phase-1 partials and the phase-2 output.
    fn split_gemv(
        phase1: KernelFn,
        phase2: KernelFn,
        stream: &CudaStream,
        dev: u32,
        a: *const u16,
        b: *const u16,
        partials: *mut f32,
        c: *mut f32,
        n: i32,
        k: i32,
        ncols: u32,
        nslabs: u32,
    ) {
        let a_v = a;
        let b_v = b;
        let p_v = partials;
        let n_v = n;
        let k_v = k;
        let ns_v = nslabs as i32;
        let mut args1: [*mut c_void; 6] = [
            (&a_v as *const *const u16) as *mut c_void,
            (&b_v as *const *const u16) as *mut c_void,
            (&p_v as *const *mut f32) as *mut c_void,
            (&n_v as *const i32) as *mut c_void,
            (&k_v as *const i32) as *mut c_void,
            (&ns_v as *const i32) as *mut c_void,
        ];
        unsafe { launch_rows(phase1, stream, dev, ncols * nslabs, 256, args1.as_mut_ptr()) }
            .unwrap();
        let p2_v = partials;
        let c_v = c;
        let n2_v = n;
        let ns2_v = nslabs as i32;
        let mut args2: [*mut c_void; 4] = [
            (&p2_v as *const *mut f32) as *mut c_void,
            (&c_v as *const *mut f32) as *mut c_void,
            (&n2_v as *const i32) as *mut c_void,
            (&ns2_v as *const i32) as *mut c_void,
        ];
        unsafe { launch_rows(phase2, stream, dev, ncols, 256, args2.as_mut_ptr()) }.unwrap();
    }

    /// One layer's buffers: the inputs (x, xn, attn, weights) and the
    /// f16/f32 output buffers for both paths.
    struct LayerBufs {
        xn: DeviceBuffer,   // the q/k/v/gate/up projection input (layer input)
        attn: DeviceBuffer, // the o-projection input (engine: attention output)
        // Embed table row 0 = the x input row: the S1-10 stage-0 gather
        // (li==0, token 0) copies it over x — an identity copy, so the
        // layer kernel's stage 0 runs without disturbing the residual
        // stream input.
        embed: DeviceBuffer,
        x_s: DeviceBuffer, // split residual stream
        x_f: DeviceBuffer, // fused residual stream
        xn_ffn_s: DeviceBuffer,
        xn_ffn_f: DeviceBuffer,
        xn_attn_s: DeviceBuffer,
        xn_attn_f: DeviceBuffer,
        down_s: DeviceBuffer,
        down_f: DeviceBuffer,
        q_w: DeviceBuffer,
        k_w: DeviceBuffer,
        v_w: DeviceBuffer,
        o_w: DeviceBuffer,
        g_w: DeviceBuffer,
        u_w: DeviceBuffer,
        d_w: DeviceBuffer,
        q_norm: DeviceBuffer,
        k_norm: DeviceBuffer,
        ffn_norm: DeviceBuffer,
        attn_norm: DeviceBuffer,
        c_q: DeviceBuffer,
        c_k: DeviceBuffer,
        c_v: DeviceBuffer,
        c_o: DeviceBuffer,
        c_g: DeviceBuffer,
        c_u: DeviceBuffer,
        c_d: DeviceBuffer,
        q16: DeviceBuffer,
        k16: DeviceBuffer,
        v16: DeviceBuffer,
        q16_f: DeviceBuffer,
        k16_f: DeviceBuffer,
        v16_f: DeviceBuffer,
        // kv caches (identical random pre-fill; the flash kernels write
        // the current step's slot) + the per-step kv_len/page-table
        // buffers; attn_f is the fused flash's attention output. kv_f is
        // TWO pages (both prefilled with the SAME random page) so the
        // layer kernel's li=1 run reads/writes the second page — the
        // engine's per-layer kv region.
        kv_s: DeviceBuffer,
        kv_f: DeviceBuffer,
        // Two-page identity page table ([0, 1]): the li=1 layer kernel
        // reads pages[li] = 1 as its base page.
        pages2: DeviceBuffer,
        lens: DeviceBuffer,
        pages: DeviceBuffer,
        attn_f: DeviceBuffer,
        // split single-plan phase-1 partials (per plan)
        p_q: DeviceBuffer,
        p_k: DeviceBuffer,
        p_v: DeviceBuffer,
        p_o: DeviceBuffer,
        p_g: DeviceBuffer,
        p_u: DeviceBuffer,
        p_d: DeviceBuffer,
    }

    /// Slab geometry, identical to `Jgemm::shape` (S1-9b: block target
    /// 192 for very tall plans 2n < k — the down projection — else 96).
    fn gemv_shape(n: usize, k: usize) -> (u32, u32) {
        let ncols = (n as u32).div_ceil(256).max(1);
        let target = if 2 * n < k { 192u32 } else { 96u32 };
        let nslabs = target.div_ceil(ncols).clamp(1, (k as u32 / 32).max(1));
        (ncols, nslabs)
    }

    fn layer_bufs(dev: u32, seed0: u64) -> LayerBufs {
        let mut seed = seed0;
        let r16 = |n: usize, s: &mut u64| (0..n).map(|_| rand_f16_bits(s)).collect::<Vec<u16>>();
        let x = r16(H, &mut seed);
        let xn = r16(H, &mut seed);
        let attn = r16(NQK, &mut seed);
        let q_w = r16(H * NQK, &mut seed);
        let k_w = r16(H * KVK, &mut seed);
        let v_w = r16(H * KVK, &mut seed);
        let o_w = r16(NQK * H, &mut seed);
        let g_w = r16(H * FFN, &mut seed);
        let u_w = r16(H * FFN, &mut seed);
        let d_w = r16(FFN * H, &mut seed);
        let q_norm = r16(D, &mut seed);
        let k_norm = r16(D, &mut seed);
        let ffn_norm = r16(H, &mut seed);
        let attn_norm = r16(H, &mut seed);
        // Random kv-cache pre-fill (identical on both paths; the flash
        // kernels overwrite slot KV_LEN - 1 with the current step's k/v).
        // kv_f holds TWO identical pages: the li=1 layer-kernel run reads
        // page 1 with the same pre-fill the split reference reads on
        // page 0.
        let kv_fill = r16(BLOCK_LEN * KVK * 2, &mut seed);
        let kv_fill2 = [&kv_fill[..], &kv_fill[..]].concat();
        // The fused and split paths need separate input copies: the
        // residual stream x is mutated in place by both paths.
        let d_xn = upl(dev, &bytes_u16(&xn));
        let d_x_s = upl(dev, &bytes_u16(&x));
        let d_x_f = upl(dev, &bytes_u16(&x));
        let d_attn = upl(dev, &bytes_u16(&attn));
        let mk = |b: &[u8]| upl(dev, b);
        let c32 = |n: usize| downl(dev, n * 4);
        let a16 = |n: usize| downl(dev, n * 2);
        let part = |n: usize, k: usize| {
            let (_, ns) = gemv_shape(n, k);
            downl(dev, n * ns as usize * 4)
        };
        LayerBufs {
            xn: d_xn,
            attn: d_attn,
            embed: upl(dev, &bytes_u16(&x)),
            x_s: d_x_s,
            x_f: d_x_f,
            xn_ffn_s: a16(H),
            xn_ffn_f: a16(H),
            xn_attn_s: a16(H),
            xn_attn_f: a16(H),
            down_s: a16(FFN),
            down_f: a16(FFN),
            q_w: mk(&bytes_u16(&q_w)),
            k_w: mk(&bytes_u16(&k_w)),
            v_w: mk(&bytes_u16(&v_w)),
            o_w: mk(&bytes_u16(&o_w)),
            g_w: mk(&bytes_u16(&g_w)),
            u_w: mk(&bytes_u16(&u_w)),
            d_w: mk(&bytes_u16(&d_w)),
            q_norm: mk(&bytes_u16(&q_norm)),
            k_norm: mk(&bytes_u16(&k_norm)),
            ffn_norm: mk(&bytes_u16(&ffn_norm)),
            attn_norm: mk(&bytes_u16(&attn_norm)),
            c_q: c32(NQK),
            c_k: c32(KVK),
            c_v: c32(KVK),
            c_o: c32(H),
            c_g: c32(FFN),
            c_u: c32(FFN),
            c_d: c32(H),
            q16: a16(NQK),
            k16: a16(KVK),
            v16: a16(KVK),
            q16_f: a16(NQK),
            k16_f: a16(KVK),
            v16_f: a16(KVK),
            kv_s: upl(dev, &bytes_u16(&kv_fill)),
            kv_f: upl(dev, &bytes_u16(&kv_fill2)),
            lens: upl(dev, &(KV_LEN as u32).to_le_bytes()),
            pages: upl(dev, &0u32.to_le_bytes()),
            pages2: upl(dev, &[0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
            attn_f: a16(NQK),
            p_q: part(NQK, H),
            p_k: part(KVK, H),
            p_v: part(KVK, H),
            p_o: part(H, NQK),
            p_g: part(FFN, H),
            p_u: part(FFN, H),
            p_d: part(H, FFN),
        }
    }

    /// Fabricated one-layer decode plans (the real plan layout: a/b from
    /// the same buffers the engine uses, c = the f32 gemm outputs).
    fn fabricate_plans(b: &LayerBufs) -> DecodeGemmPlans {
        let a16p = |buf: &DeviceBuffer| buf.as_ptr() as *const u16;
        let c32p = |buf: &DeviceBuffer| buf.as_ptr() as *mut f32;
        let plan = |a: &DeviceBuffer, b: &DeviceBuffer, c: &DeviceBuffer, n: usize, k: usize| {
            GemmPlan::row_major_f16(a16p(a), a16p(b), c32p(c), 1, n, k)
        };
        let q = plan(&b.xn, &b.q_w, &b.c_q, NQK, H);
        let k = plan(&b.xn, &b.k_w, &b.c_k, KVK, H);
        let v = plan(&b.xn, &b.v_w, &b.c_v, KVK, H);
        let o = plan(&b.attn, &b.o_w, &b.c_o, H, NQK);
        let gate = plan(&b.xn, &b.g_w, &b.c_g, FFN, H);
        let up = plan(&b.xn, &b.u_w, &b.c_u, FFN, H);
        let down = plan(&b.down_f, &b.d_w, &b.c_d, H, FFN);
        DecodeGemmPlans {
            layers: vec![LayerGemmPlans { q, k, v, o, gate, up, down }],
            // n >= 192*256 so nslabs_lm == 1 (the build_plans invariant;
            // the real lm_head has nslabs = 1 by the same formula).
            lm_head: plan(&b.xn, &b.d_w, &b.c_d, 60000, H),
        }
    }

    /// Two-layer fabricated plans (identical shapes): the li=1 layer-
    /// kernel run needs its own 7-row block at table + 7 and a two-layer
    /// geometry (n_layers == 2 for the stage_ts slots). `b1` provides the
    /// layer-0 rows (unused by the li=1 launch, needed for the geometry),
    /// `b2` the layer-1 rows the kernel actually runs.
    fn fabricate_plans2(b1: &LayerBufs, b2: &LayerBufs) -> DecodeGemmPlans {
        let a16p = |buf: &DeviceBuffer| buf.as_ptr() as *const u16;
        let c32p = |buf: &DeviceBuffer| buf.as_ptr() as *mut f32;
        let plan = |a: &DeviceBuffer, b: &DeviceBuffer, c: &DeviceBuffer, n: usize, k: usize| {
            GemmPlan::row_major_f16(a16p(a), a16p(b), c32p(c), 1, n, k)
        };
        let g1 = LayerGemmPlans {
            q: plan(&b1.xn, &b1.q_w, &b1.c_q, NQK, H),
            k: plan(&b1.xn, &b1.k_w, &b1.c_k, KVK, H),
            v: plan(&b1.xn, &b1.v_w, &b1.c_v, KVK, H),
            o: plan(&b1.attn, &b1.o_w, &b1.c_o, H, NQK),
            gate: plan(&b1.xn, &b1.g_w, &b1.c_g, FFN, H),
            up: plan(&b1.xn, &b1.u_w, &b1.c_u, FFN, H),
            down: plan(&b1.down_f, &b1.d_w, &b1.c_d, H, FFN),
        };
        let g2 = LayerGemmPlans {
            q: plan(&b2.xn, &b2.q_w, &b2.c_q, NQK, H),
            k: plan(&b2.xn, &b2.k_w, &b2.c_k, KVK, H),
            v: plan(&b2.xn, &b2.v_w, &b2.c_v, KVK, H),
            o: plan(&b2.attn, &b2.o_w, &b2.c_o, H, NQK),
            gate: plan(&b2.xn, &b2.g_w, &b2.c_g, FFN, H),
            up: plan(&b2.xn, &b2.u_w, &b2.c_u, FFN, H),
            down: plan(&b2.down_f, &b2.d_w, &b2.c_d, H, FFN),
        };
        DecodeGemmPlans { layers: vec![g1, g2], lm_head: plan(&b2.xn, &b2.d_w, &b2.c_d, 60000, H) }
    }

    /// Full fused layer run (the 8 fused launches, mirroring the engine's
    /// pipeline: p1_qkv at layer start, the fused flash (kv write +
    /// attention), p1_o, p2_o, then p1_gu / p2_gu_d / p2_d — the points
    /// where the split path reads attn (post-flash), xn (ffn-normed) and
    /// down (the swiglu output)) on `bufs` (fresh input copies), with the
    /// fused unit built over `fabricate_plans`.
    fn run_fused(
        fused: &mut FusedDecodeKernels,
        jgemm: &Jgemm,
        ctx: &CudaContext,
        decode: &DecodeKernels,
        stream: &CudaStream,
        dev: u32,
        b: &LayerBufs,
    ) {
        // Plan table built over THIS buffer set — the table rows hold
        // device pointers into `b`, so a run on another bufs pair must
        // rebuild the table (the gate-2 determinism pair is separate).
        fused.build_plans(ctx.device_id(), jgemm, &fabricate_plans(b)).unwrap();
        let g = fused.geom();
        fused.launch_p1(stream, g.tables, 3, g.grid_qkv[0]).unwrap();
        fused
            .launch_p2_qkv(
                stream,
                g.pq,
                g.pk,
                g.pv,
                b.q16_f.as_ptr() as *mut u16,
                b.k16_f.as_ptr() as *mut u16,
                b.v16_f.as_ptr() as *mut u16,
                b.q_norm.as_ptr() as *const u16,
                b.k_norm.as_ptr() as *const u16,
                NQK as u32,
                KVK as u32,
                KVK as u32,
                g.nslabs_q,
                g.nslabs_k,
                g.nslabs_v,
                D as u32,
                (D / 2) as u32,
                POS,
                ETA,
                qscale(),
                1.0,
                EPS,
                1, // head_norm on
                g.grid_qkv_p2,
            )
            .unwrap();
        // fused flash: kv write of the current slot + flash attention
        // (writes b.attn_f)
        decode
            .launch_decode_step_gqa_flash_fused(
                dev,
                b.q16_f.as_ptr() as *const u16,
                b.pages.as_ptr() as *const u32,
                b.kv_f.as_ptr() as *const u16,
                b.lens.as_ptr() as *const u32,
                b.attn_f.as_ptr() as *mut u16,
                1,                // b
                16,               // qh
                D as u32,         // d
                BLOCK_LEN as u32, // block_len
                2,                // kv_ratio
                8,                // kv_heads
                BLOCK_LEN as u32, // max_kv (one page)
                1,                // total_pages
                1,                // identity page table
                b.k16_f.as_ptr() as *const u16,
                b.v16_f.as_ptr() as *const u16,
            )
            .unwrap();
        // o phase-1 — its own node: reads the FULL attention row (all
        // heads), which the flash blocks write only per head; stream
        // ordering makes it race-free. Writes the o slab partials into
        // g.po, byte-identically to the split gemv_m1_f16f32. The plan
        // row's a is b.attn (the split flash's out — run_split ran
        // first on the same bufs, and the attn assertion below pins
        // b.attn == b.attn_f, so the o phase-1 reads input bits
        // identical to the engine's fused path, where the flash out IS
        // the o plan input).
        fused.launch_p1(stream, unsafe { g.tables.add(3) }, 1, g.grid_o[0]).unwrap();
        fused
            .launch_p2_add_rms(
                stream,
                g.po,
                b.x_f.as_ptr() as *mut u16,
                b.xn_ffn_f.as_ptr() as *mut u16,
                b.ffn_norm.as_ptr() as *const u16,
                H as u32,
                g.nslabs_o,
                EPS,
            )
            .unwrap();
        // gate/up phase-1 — reads xn (p2_o's ffn-normed x), the split
        // path's read point
        fused.launch_p1(stream, unsafe { g.tables.add(4) }, 2, g.grid_gu[0]).unwrap();
        // merged gate/up phase-2 + cast-SiLU-GLU + down phase-1 (rows g,
        // u, d at table.add(4)); writes down_f and the down partials,
        // block-local, identical arithmetic to the split p2_gu + p1_d.
        fused.launch_p2_gu_d(stream, unsafe { g.tables.add(4) }, g.grid_gu_p2).unwrap();
        fused
            .launch_p2_add_rms(
                stream,
                g.pd,
                b.x_f.as_ptr() as *mut u16,
                b.xn_attn_f.as_ptr() as *mut u16,
                b.attn_norm.as_ptr() as *const u16,
                H as u32,
                g.nslabs_d,
                EPS,
            )
            .unwrap();
    }

    /// Full split layer run on `bufs`: 7 two-phase gemvs + 3 casts + 2
    /// head norms + 2 ropes + fused_add_rms + fused_cast_swiglu +
    /// add_cast + rms_norm — the exact split sequence. Shapes come from
    /// `jgemm.shape` so the buffers line up with the fused layout.
    fn run_split(
        lib: &JLib,
        dense: &DenseKernels,
        diff: &DiffKernels,
        decode: &DecodeKernels,
        jgemm: &Jgemm,
        stream: &CudaStream,
        dev: u32,
        b: &LayerBufs,
    ) {
        let phase1 = lib.kernel("gemv_m1_f16f32").unwrap();
        let phase2 = lib.kernel("gemv_m1_f16f32_reduce").unwrap();
        let gemv = |a: &DeviceBuffer,
                    w: &DeviceBuffer,
                    part: &DeviceBuffer,
                    c: &DeviceBuffer,
                    n: i32,
                    k: i32| {
            let (ncols, nslabs) = jgemm.shape(n, k);
            split_gemv(
                phase1,
                phase2,
                stream,
                dev,
                a.as_ptr() as *const u16,
                w.as_ptr() as *const u16,
                part.as_ptr() as *mut f32,
                c.as_ptr() as *mut f32,
                n,
                k,
                ncols,
                nslabs,
            );
        };
        // q/k/v projections + casts
        gemv(&b.xn, &b.q_w, &b.p_q, &b.c_q, NQK as i32, H as i32);
        diff.launch_cast_f32_f16(
            dev,
            stream,
            b.c_q.as_ptr() as *const f32,
            b.q16.as_ptr() as *mut u16,
            NQK as u32,
        )
        .unwrap();
        gemv(&b.xn, &b.k_w, &b.p_k, &b.c_k, KVK as i32, H as i32);
        diff.launch_cast_f32_f16(
            dev,
            stream,
            b.c_k.as_ptr() as *const f32,
            b.k16.as_ptr() as *mut u16,
            KVK as u32,
        )
        .unwrap();
        gemv(&b.xn, &b.v_w, &b.p_v, &b.c_v, KVK as i32, H as i32);
        diff.launch_cast_f32_f16(
            dev,
            stream,
            b.c_v.as_ptr() as *const f32,
            b.v16.as_ptr() as *mut u16,
            KVK as u32,
        )
        .unwrap();
        // q/k head norms (in place) + ropes (q scaled)
        dense
            .launch_rms_heads(
                dev,
                stream,
                b.q16.as_ptr() as *const u16,
                b.q16.as_ptr() as *mut u16,
                b.q_norm.as_ptr() as *const u16,
                16,
                D as u32,
                EPS,
            )
            .unwrap();
        dense
            .launch_rms_heads(
                dev,
                stream,
                b.k16.as_ptr() as *const u16,
                b.k16.as_ptr() as *mut u16,
                b.k_norm.as_ptr() as *const u16,
                8,
                D as u32,
                EPS,
            )
            .unwrap();
        dense
            .launch_rope_heads(
                dev,
                stream,
                b.q16.as_ptr() as *mut u16,
                16,
                (D / 2) as u32,
                POS,
                ETA,
                qscale(),
            )
            .unwrap();
        dense
            .launch_rope_heads(
                dev,
                stream,
                b.k16.as_ptr() as *mut u16,
                8,
                (D / 2) as u32,
                POS,
                ETA,
                1.0,
            )
            .unwrap();
        // kv write of the current slot + flash decode attention (writes
        // b.attn — the o projection input, exactly the engine's split
        // sequence)
        dense
            .launch_kv_write(
                dev,
                stream,
                b.k16.as_ptr() as *const u16,
                b.v16.as_ptr() as *const u16,
                b.kv_s.as_ptr() as *mut u16,
                0,                   // phys (one page)
                (KV_LEN - 1) as u32, // off
                BLOCK_LEN as u32,
                8, // kv_heads
                D as u32,
                1, // total_pages
            )
            .unwrap();
        decode
            .launch_decode_step_gqa_flash(
                dev,
                b.q16.as_ptr() as *const u16,
                b.pages.as_ptr() as *const u32,
                b.kv_s.as_ptr() as *const u16,
                b.lens.as_ptr() as *const u32,
                b.attn.as_ptr() as *mut u16,
                1,                // b
                16,               // qh
                D as u32,         // d
                BLOCK_LEN as u32, // block_len
                2,                // kv_ratio
                8,                // kv_heads
                BLOCK_LEN as u32, // max_kv (one page)
                1,                // total_pages
                1,                // identity page table
            )
            .unwrap();
        // o projection + fused residual add + ffn rms
        gemv(&b.attn, &b.o_w, &b.p_o, &b.c_o, H as i32, NQK as i32);
        dense
            .launch_fused_add_rms(
                dev,
                stream,
                b.x_s.as_ptr() as *mut u16,
                b.c_o.as_ptr() as *const f32,
                b.xn_ffn_s.as_ptr() as *mut u16,
                b.ffn_norm.as_ptr() as *const u16,
                H as u32,
                EPS,
            )
            .unwrap();
        // gate/up + fused cast-swiglu
        gemv(&b.xn, &b.g_w, &b.p_g, &b.c_g, FFN as i32, H as i32);
        gemv(&b.xn, &b.u_w, &b.p_u, &b.c_u, FFN as i32, H as i32);
        dense
            .launch_fused_cast_swiglu(
                dev,
                stream,
                b.c_g.as_ptr() as *const f32,
                b.c_u.as_ptr() as *const f32,
                b.down_s.as_ptr() as *mut u16,
                FFN as u32,
            )
            .unwrap();
        // down + add_cast + rms_norm (next attn norm)
        gemv(&b.down_s, &b.d_w, &b.p_d, &b.c_d, H as i32, FFN as i32);
        dense
            .launch_add_cast(
                dev,
                stream,
                b.x_s.as_ptr() as *mut u16,
                b.c_d.as_ptr() as *const f32,
                H as u32,
            )
            .unwrap();
        dense
            .launch_rms_norm(
                dev,
                stream,
                b.x_s.as_ptr() as *const u16,
                b.xn_attn_s.as_ptr() as *mut u16,
                b.attn_norm.as_ptr() as *const u16,
                H as u32,
                EPS,
            )
            .unwrap();
    }

    /// One layer through the S1-10 persistent kernel: stages 0..8 in a
    /// single launch over the plan table of the fused geometry `g` (must
    /// be built over THIS bufs pair — the kernel reads the plan rows and
    /// writes the shared partials segments). The kernel const addresses
    /// `b`: x = b.x_f (the fresh residual stream — the split reference
    /// mutated b.x_s, not b.x_f), xn = b.xn_attn_f (stage 5's ffn-normed
    /// value is transient — one xn buffer, as in the engine), the
    /// q/k/v/attn outputs, the kv cache (b.kv_f), and embed row 0 = the
    /// initial x (li==0 gather: identity copy, token 0). wnext = the
    /// attn-norm weight (stage 8 and the rms0 stage), wffn = the ffn-norm
    /// weight (stage 5), li = 0 (exercises the gather + rms0 branch; its
    /// xn output is transient, overwritten by stages 5/8 — the engine A/B
    /// gate covers the rms0 → stage-1 wiring).
    fn run_layer_fused(
        lf: &mut LayerFusedKernels,
        ctx: &CudaContext,
        stream: &CudaStream,
        b: &LayerBufs,
        g: &FusedGeom,
    ) {
        let spec = LayerFusedSpec {
            h: H,
            nqk: NQK,
            kvk: KVK,
            ffn: FFN,
            d: D,
            q_heads: 16,
            kv_heads: 8,
            block_len: BLOCK_LEN,
            max_kv: BLOCK_LEN,
            total_pages: 1,
            pp: 1,
            eta: ETA,
            eps: EPS,
            head_norm: true,
            embed: b.embed.as_ptr() as *const u16,
            x: b.x_f.as_ptr() as *mut u16,
            xn: b.xn_attn_f.as_ptr() as *mut u16,
            q16: b.q16_f.as_ptr() as *mut u16,
            k16: b.k16_f.as_ptr() as *mut u16,
            v16: b.v16_f.as_ptr() as *mut u16,
            attn: b.attn_f.as_ptr() as *mut u16,
            kv: b.kv_f.as_ptr() as *mut u16,
            lens: b.lens.as_ptr() as *const u32,
            pages: b.pages.as_ptr() as *const u32,
            wnorm0: b.attn_norm.as_ptr() as *const u16,
        };
        lf.build(ctx.device_id(), g, &spec).unwrap();
        lf.launch(
            stream,
            g.tables,                           // layer 0 (li * 7 rows)
            b.attn_norm.as_ptr() as *const u16, // wnext (stage 8 / rms0)
            b.ffn_norm.as_ptr() as *const u16,  // wffn (stage 5)
            b.q_norm.as_ptr() as *const u16,    // wq (stage 2 head norm)
            b.k_norm.as_ptr() as *const u16,    // wk (stage 2 head norm)
            0,                                  // li (stage 0 on)
            1,                                  // n_layers (stage_ts slots)
            POS,                                // rope pos
            0,                                  // token (embed row 0)
        )
        .unwrap();
    }

    /// Bit-exact compare of two f16 buffers.
    fn assert_u16_eq(a: &[u16], b: &[u16], tag: &str) {
        assert_eq!(a.len(), b.len(), "{tag}: len");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "{tag}: bit mismatch at [{i}]: split {x:04x} fused {y:04x}");
        }
    }

    /// Bit-exact compare of two f32 buffers.
    fn assert_f32_eq(a: &[f32], b: &[f32], tag: &str) {
        assert_eq!(a.len(), b.len(), "{tag}: len");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{tag}: bit mismatch at [{i}]: split {x} fused {y}"
            );
        }
    }

    /// FFN kernel micro-bench: the gate/up phase-1
    /// (`gemv_m1_f16f32_multi`, 2 plans) and the merged gate/up phase-2 +
    /// cast-SiLU-GLU + down phase-1 (`gemv_p2_gu_p1d_swiglu`) in
    /// isolation — back-to-back launches timed with cudaEvents, mean
    /// over N iterations (warmup discarded). GB/s is computed over the
    /// weight bytes each kernel moves: gate+up = 2 x FFN*H*2,
    /// down = FFN*H*2. The point is single-kernel A/B measurement for
    /// the S1-9b FFN tuning loop, independent of the engine's launch
    /// interleaving (the engine's ffn_gu/ffn_d segments include
    /// per-launch overheads this bench does not).
    #[test]
    #[ignore = "gpu.yml: l3-fused-decode / microbench"]
    fn ffn_kernels_microbench() {
        let (ctx, stream, arch, cache) = setup();
        let dev = ctx.device_id().index();
        let jgemm = Jgemm::new(dev, &arch, Some(cache.clone())).unwrap();
        let gemm_lib = load_gemm_m1(&arch, Some(cache.clone()));
        let mut fused = FusedDecodeKernels::new(
            dev,
            &arch,
            Some(cache),
            gemm_lib.kernel("gemv_m1_f16f32_reduce").unwrap(),
        )
        .unwrap();
        let b = layer_bufs(dev, 0xB17E_5EEDu64);
        fused.build_plans(ctx.device_id(), &jgemm, &fabricate_plans(&b)).unwrap();
        let g = fused.geom();

        // Weight bytes per launch — the bandwidth bundle each kernel
        // moves (the residual-stream A vectors are negligible).
        const GU_BYTES: f64 = 2.0 * (FFN * H * 2) as f64;
        const D_BYTES: f64 = (FFN * H * 2) as f64;

        // Warmup: settle clocks/allocations before the timed loop.
        for _ in 0..3 {
            fused.launch_p1(&stream, unsafe { g.tables.add(4) }, 2, g.grid_gu[0]).unwrap();
            fused.launch_p2_gu_d(&stream, unsafe { g.tables.add(4) }, g.grid_gu_p2).unwrap();
            fused
                .launch_p2_add_rms(
                    &stream,
                    g.pd,
                    b.x_f.as_ptr() as *mut u16,
                    b.xn_attn_f.as_ptr() as *mut u16,
                    b.attn_norm.as_ptr() as *const u16,
                    H as u32,
                    g.nslabs_d,
                    EPS,
                )
                .unwrap();
        }
        stream.synchronize().unwrap();

        const N: usize = 10;
        let mut t_gu = [0.0f64; N];
        let mut t_d = [0.0f64; N];
        let mut t_rms = [0.0f64; N];
        let ev0 = CudaEvent::new(ctx.device_id()).unwrap();
        let ev1 = CudaEvent::new(ctx.device_id()).unwrap();
        let ev2 = CudaEvent::new(ctx.device_id()).unwrap();
        let ev3 = CudaEvent::new(ctx.device_id()).unwrap();
        for i in 0..N {
            ev0.record(&stream).unwrap();
            fused.launch_p1(&stream, unsafe { g.tables.add(4) }, 2, g.grid_gu[0]).unwrap();
            ev1.record(&stream).unwrap();
            fused.launch_p2_gu_d(&stream, unsafe { g.tables.add(4) }, g.grid_gu_p2).unwrap();
            ev2.record(&stream).unwrap();
            fused
                .launch_p2_add_rms(
                    &stream,
                    g.pd,
                    b.x_f.as_ptr() as *mut u16,
                    b.xn_attn_f.as_ptr() as *mut u16,
                    b.attn_norm.as_ptr() as *const u16,
                    H as u32,
                    g.nslabs_d,
                    EPS,
                )
                .unwrap();
            ev3.record(&stream).unwrap();
            stream.synchronize().unwrap();
            t_gu[i] = ev0.elapsed_ms(&ev1).unwrap() as f64;
            t_d[i] = ev1.elapsed_ms(&ev2).unwrap() as f64;
            t_rms[i] = ev2.elapsed_ms(&ev3).unwrap() as f64;
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let mg = mean(&t_gu);
        let md = mean(&t_d);
        let mr = mean(&t_rms);
        let wnd = |v: &[f64]| {
            let (a, b) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
            (a * 1e3, b * 1e3)
        };
        let (mn_g, mx_g) = wnd(&t_gu);
        let (mn_d, mx_d) = wnd(&t_d);
        let (mn_r, mx_r) = wnd(&t_rms);
        // A/B: the same kernel with nslabs=24 (the o-style count) — pins
        // the per-element cost of the partial-sum loop.
        let mut t_r24 = [0.0f64; N];
        for i in 0..N {
            ev2.record(&stream).unwrap();
            fused
                .launch_p2_add_rms(
                    &stream,
                    g.pd,
                    b.x_f.as_ptr() as *mut u16,
                    b.xn_attn_f.as_ptr() as *mut u16,
                    b.attn_norm.as_ptr() as *const u16,
                    H as u32,
                    24,
                    EPS,
                )
                .unwrap();
            ev3.record(&stream).unwrap();
            stream.synchronize().unwrap();
            t_r24[i] = ev2.elapsed_ms(&ev3).unwrap() as f64;
        }
        let (mn_r2, mx_r2) = wnd(&t_r24);
        println!(
            "ffn microbench (n={N}, grid_gu={}, grid_gu_p2={}, slabs g/u/d={}/{}/{}, \
             window min..max us):\n  p1_gu     mean {:7.3} us [{:.2}..{:.2}]  {:6.0} GB/s\n  \
             p2_gu_d   mean {:7.3} us [{:.2}..{:.2}]  {:6.0} GB/s\n  \
             p2_add_rms mean {:6.3} us [{:.2}..{:.2}]   (nslabs=24: {:6.3} us [{:.2}..{:.2}])",
            g.grid_gu[0],
            g.grid_gu_p2,
            g.nslabs_g,
            g.nslabs_g,
            g.nslabs_d,
            mg * 1e3,
            mn_g,
            mx_g,
            GU_BYTES / (mg * 1e-3) / 1e9,
            md * 1e3,
            mn_d,
            mx_d,
            D_BYTES / (md * 1e-3) / 1e9,
            mr * 1e3,
            mn_r,
            mx_r,
            mean(&t_r24) * 1e3,
            mn_r2,
            mx_r2,
        );
    }

    fn setup() -> (CudaContext, CudaStream, String, std::path::PathBuf) {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let stream = CudaStream::new(ctx.device_id()).unwrap();
        let arch = reinfer_cuda::arch::resolve_arch().unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-fused-decode");
        let _ = std::fs::remove_dir_all(&cache);
        (ctx, stream, arch, cache)
    }

    /// Gate 1: the full fused layer vs the full split layer — every
    /// output and every phase-1 partials segment bit-identical.
    #[test]
    #[ignore = "gpu.yml: l3-fused-decode / kernel-level"]
    fn fused_layer_bit_exact_vs_split() {
        let (ctx, stream, arch, cache) = setup();
        let dev = ctx.device_id().index();
        let dense = DenseKernels::new(&arch, Some(cache.clone())).unwrap();
        let diff = DiffKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let decode = DecodeKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let jgemm = Jgemm::new(dev, &arch, Some(cache.clone())).unwrap();
        let gemm_lib = load_gemm_m1(&arch, Some(cache.clone()));
        let mut fused = FusedDecodeKernels::new(
            dev,
            &arch,
            Some(cache),
            gemm_lib.kernel("gemv_m1_f16f32_reduce").unwrap(),
        )
        .unwrap();

        let b = layer_bufs(dev, 0x51E9u64);

        run_split(&gemm_lib, &dense, &diff, &decode, &jgemm, &stream, dev, &b);
        run_fused(&mut fused, &jgemm, &ctx, &decode, &stream, dev, &b);
        stream.synchronize().unwrap();

        // q/k/v after rope
        assert_u16_eq(&read_u16(&b.q16, NQK), &read_u16(&b.q16_f, NQK), "q16");
        assert_u16_eq(&read_u16(&b.k16, KVK), &read_u16(&b.k16_f, KVK), "k16");
        assert_u16_eq(&read_u16(&b.v16, KVK), &read_u16(&b.v16_f, KVK), "v16");
        // attention output (the fused flash vs the split flash — the o
        // projection input)
        assert_u16_eq(&read_u16(&b.attn, NQK), &read_u16(&b.attn_f, NQK), "attn");
        // ffn norm output (o residual + ffn rms)
        assert_u16_eq(&read_u16(&b.xn_ffn_s, H), &read_u16(&b.xn_ffn_f, H), "xn_ffn");
        // down (gate/up reductions + swiglu)
        assert_u16_eq(&read_u16(&b.down_s, FFN), &read_u16(&b.down_f, FFN), "down");
        // residual stream x (after both adds) and the attn norm output
        assert_u16_eq(&read_u16(&b.x_s, H), &read_u16(&b.x_f, H), "x");
        assert_u16_eq(&read_u16(&b.xn_attn_s, H), &read_u16(&b.xn_attn_f, H), "xn_attn");
        // phase-1 partials: the multi kernel's segments vs the single-plan
        // kernel's outputs — the p1_multi contract.
        let g = fused.geom();
        assert_f32_eq(
            &read_f32(&b.p_q, NQK * g.nslabs_q as usize),
            &read_f32_ptr(dev, g.pq, NQK * g.nslabs_q as usize),
            "partials q",
        );
        assert_f32_eq(
            &read_f32(&b.p_k, KVK * g.nslabs_k as usize),
            &read_f32_ptr(dev, g.pk, KVK * g.nslabs_k as usize),
            "partials k",
        );
        assert_f32_eq(
            &read_f32(&b.p_v, KVK * g.nslabs_v as usize),
            &read_f32_ptr(dev, g.pv, KVK * g.nslabs_v as usize),
            "partials v",
        );
        assert_f32_eq(
            &read_f32(&b.p_o, H * g.nslabs_o as usize),
            &read_f32_ptr(dev, g.po, H * g.nslabs_o as usize),
            "partials o",
        );
        assert_f32_eq(
            &read_f32(&b.p_g, FFN * g.nslabs_g as usize),
            &read_f32_ptr(dev, g.pg, FFN * g.nslabs_g as usize),
            "partials gate",
        );
        assert_f32_eq(
            &read_f32(&b.p_u, FFN * g.nslabs_g as usize),
            &read_f32_ptr(dev, g.pu, FFN * g.nslabs_g as usize),
            "partials up",
        );
        assert_f32_eq(
            &read_f32(&b.p_d, H * g.nslabs_d as usize),
            &read_f32_ptr(dev, g.pd, H * g.nslabs_d as usize),
            "partials down",
        );
        println!(
            "fused layer: bit-exact vs split on q/k/v, attn, x, xn_ffn, xn_attn, down and all 7 phase-1 partials segments"
        );
    }

    /// Gate 2 (S1-10): the whole layer through the single persistent
    /// kernel vs the split sequence — every output and every phase-1
    /// partials segment bit-identical (the p1_multi contract inside one
    /// kernel, with the S1-9 arithmetic preserved stage by stage).
    #[test]
    #[ignore = "gpu.yml: l3-fused-decode / kernel-level"]
    fn layer_fused_bit_exact_vs_split() {
        let (ctx, stream, arch, cache) = setup();
        let dev = ctx.device_id().index();
        let dense = DenseKernels::new(&arch, Some(cache.clone())).unwrap();
        let diff = DiffKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let decode = DecodeKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let jgemm = Jgemm::new(dev, &arch, Some(cache.clone())).unwrap();
        let gemm_lib = load_gemm_m1(&arch, Some(cache.clone()));
        let mut fused = FusedDecodeKernels::new(
            dev,
            &arch,
            Some(cache.clone()),
            gemm_lib.kernel("gemv_m1_f16f32_reduce").unwrap(),
        )
        .unwrap();
        let mut lf = LayerFusedKernels::new(dev, &arch, Some(cache)).unwrap();

        // The layer arm runs on a FRESH bufs pair (same seed — identical
        // buffers): the split reference mutates x_s / writes the o plan's
        // input (b.attn = the split flash output, which the layer
        // kernel's own stage 3 must reproduce bit-exactly) and the
        // reference partials; the persistent kernel writes its own f16
        // outputs into b.*_f and the shared partials segments.
        let bl = layer_bufs(dev, 0x51E9u64);
        run_split(&gemm_lib, &dense, &diff, &decode, &jgemm, &stream, dev, &bl);
        fused.build_plans(ctx.device_id(), &jgemm, &fabricate_plans(&bl)).unwrap();
        let g = fused.geom();
        run_layer_fused(&mut lf, &ctx, &stream, &bl, g);
        stream.synchronize().unwrap();

        // q/k/v after head norm + rope
        assert_u16_eq(&read_u16(&bl.q16, NQK), &read_u16(&bl.q16_f, NQK), "layer q16");
        assert_u16_eq(&read_u16(&bl.k16, KVK), &read_u16(&bl.k16_f, KVK), "layer k16");
        assert_u16_eq(&read_u16(&bl.v16, KVK), &read_u16(&bl.v16_f, KVK), "layer v16");
        // flash attention output (the layer kernel's stage 3 vs the split
        // flash — also the o plan input the stage-4 phase-1 read)
        assert_u16_eq(&read_u16(&bl.attn, NQK), &read_u16(&bl.attn_f, NQK), "layer attn");
        // down (gate/up reductions + swiglu, stage 7)
        assert_u16_eq(&read_u16(&bl.down_s, FFN), &read_u16(&bl.down_f, FFN), "layer down");
        // residual stream x (both adds; the layer kernel's x_f input was
        // the fresh initial x — the stage-0 gather is an identity copy)
        assert_u16_eq(&read_u16(&bl.x_s, H), &read_u16(&bl.x_f, H), "layer x");
        // the attn-norm output (stage 8 — the same rms code path as
        // stage 5's ffn-normed value, which is transient in the engine's
        // single-xn layout and covered by the engine A/B gate)
        assert_u16_eq(&read_u16(&bl.xn_attn_s, H), &read_u16(&bl.xn_attn_f, H), "layer xn_attn");
        // phase-1 partials: the persistent kernel's segments vs the
        // single-plan kernel's outputs — the p1_multi contract held
        // inside the layer kernel.
        assert_f32_eq(
            &read_f32(&bl.p_q, NQK * g.nslabs_q as usize),
            &read_f32_ptr(dev, g.pq, NQK * g.nslabs_q as usize),
            "layer partials q",
        );
        assert_f32_eq(
            &read_f32(&bl.p_k, KVK * g.nslabs_k as usize),
            &read_f32_ptr(dev, g.pk, KVK * g.nslabs_k as usize),
            "layer partials k",
        );
        assert_f32_eq(
            &read_f32(&bl.p_v, KVK * g.nslabs_v as usize),
            &read_f32_ptr(dev, g.pv, KVK * g.nslabs_v as usize),
            "layer partials v",
        );
        assert_f32_eq(
            &read_f32(&bl.p_o, H * g.nslabs_o as usize),
            &read_f32_ptr(dev, g.po, H * g.nslabs_o as usize),
            "layer partials o",
        );
        assert_f32_eq(
            &read_f32(&bl.p_g, FFN * g.nslabs_g as usize),
            &read_f32_ptr(dev, g.pg, FFN * g.nslabs_g as usize),
            "layer partials gate",
        );
        assert_f32_eq(
            &read_f32(&bl.p_u, FFN * g.nslabs_g as usize),
            &read_f32_ptr(dev, g.pu, FFN * g.nslabs_g as usize),
            "layer partials up",
        );
        assert_f32_eq(
            &read_f32(&bl.p_d, H * g.nslabs_d as usize),
            &read_f32_ptr(dev, g.pd, H * g.nslabs_d as usize),
            "layer partials down",
        );
        println!(
            "layer-fused kernel: bit-exact vs split on q/k/v, attn, x, xn_attn, down and all 7 phase-1 partials segments (one launch, stages 0..8)"
        );
    }

    /// Gate 1b (S1-10): the persistent kernel's li=1 run — the 2-layer
    /// geometry: the kernel's `pages + li*pp` lands on the SECOND page,
    /// the stage-3 kv write goes into layer 1's region, wnext/wffn/wq/wk
    /// are layer-1 weights, x/xn carry the layer-1 state (no gather).
    /// The kv prefill deliberately differs between the two pages (page 1
    /// == the split reference's prefill, page 0 = a different random
    /// page) so any page-offset slip (reading or writing the wrong page)
    /// fails the comparison instead of being masked by identical pages.
    /// Every output and every phase-1 partials segment must be
    /// bit-identical to the split sequence on the same buffers.
    #[test]
    #[ignore = "gpu.yml: l3-fused-decode / kernel-level"]
    fn layer_fused_li1_bit_exact_vs_split() {
        let (ctx, stream, arch, cache) = setup();
        let dev = ctx.device_id().index();
        let dense = DenseKernels::new(&arch, Some(cache.clone())).unwrap();
        let diff = DiffKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let decode = DecodeKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let jgemm = Jgemm::new(dev, &arch, Some(cache.clone())).unwrap();
        let gemm_lib = load_gemm_m1(&arch, Some(cache.clone()));
        let mut fused = FusedDecodeKernels::new(
            dev,
            &arch,
            Some(cache.clone()),
            gemm_lib.kernel("gemv_m1_f16f32_reduce").unwrap(),
        )
        .unwrap();
        let mut lf = LayerFusedKernels::new(dev, &arch, Some(cache)).unwrap();

        // b1 = the layer-0 rows of the two-layer plan table (geometry
        // filler — the li=1 launch reads rows 7..13), b2 = the layer-1
        // buffers the kernel actually runs. Same seed: identical buffers.
        let b1 = layer_bufs(dev, 0x51E9u64);
        let b2 = layer_bufs(dev, 0x51E9u64);
        run_split(&gemm_lib, &dense, &diff, &decode, &jgemm, &stream, dev, &b2);
        fused.build_plans(ctx.device_id(), &jgemm, &fabricate_plans2(&b1, &b2)).unwrap();
        let g = fused.geom();

        // The layer-1 kv region: a dedicated 2-page buffer — page 1 =
        // the split reference's prefill (read back from kv_s; the split
        // only mutated slot KV_LEN-1, which the layer kernel overwrites
        // with the same k/v anyway), page 0 = a different random page.
        // The layer kernel must touch page 1 only. Layout is the engine's
        // KvStore layout: [all pages' K][all pages' V] — the kernel's
        // v_base = total_pages*block_len*per_tok starts the V region.
        let per_tok_kv = KVK; // kv_heads * d
        let kv_s = read_u16(&b2.kv_s, BLOCK_LEN * per_tok_kv * 2);
        let mut seed = 0x51E9u64 ^ 0x0BAD_5EEDu64;
        let page_a = (0..(BLOCK_LEN * per_tok_kv * 2))
            .map(|_| rand_f16_bits(&mut seed))
            .collect::<Vec<u16>>();
        let kv2: Vec<u16> = [
            &page_a[0..BLOCK_LEN * per_tok_kv], // K0
            &kv_s[0..BLOCK_LEN * per_tok_kv],   // K1 (kv_fill K)
            &page_a[BLOCK_LEN * per_tok_kv..],  // V0
            &kv_s[BLOCK_LEN * per_tok_kv..],    // V1 (kv_fill V)
        ]
        .concat();
        let d_kv2 = upl(dev, &bytes_u16(&kv2));

        let spec = LayerFusedSpec {
            h: H,
            nqk: NQK,
            kvk: KVK,
            ffn: FFN,
            d: D,
            q_heads: 16,
            kv_heads: 8,
            block_len: BLOCK_LEN,
            max_kv: BLOCK_LEN,
            total_pages: 2,
            pp: 1,
            eta: ETA,
            eps: EPS,
            head_norm: true,
            embed: std::ptr::null(), // no gather at li > 0
            x: b2.x_f.as_ptr() as *mut u16,
            xn: b2.xn_attn_f.as_ptr() as *mut u16,
            q16: b2.q16_f.as_ptr() as *mut u16,
            k16: b2.k16_f.as_ptr() as *mut u16,
            v16: b2.v16_f.as_ptr() as *mut u16,
            attn: b2.attn_f.as_ptr() as *mut u16,
            kv: d_kv2.as_ptr() as *mut u16,
            lens: b2.lens.as_ptr() as *const u32,
            pages: b2.pages2.as_ptr() as *const u32,
            wnorm0: b2.attn_norm.as_ptr() as *const u16,
        };
        lf.build(ctx.device_id(), g, &spec).unwrap();
        lf.launch(
            &stream,
            unsafe { g.tables.add(7) },          // layer-1 rows (li * 7)
            b2.attn_norm.as_ptr() as *const u16, // wnext (stage 8)
            b2.ffn_norm.as_ptr() as *const u16,  // wffn (stage 5)
            b2.q_norm.as_ptr() as *const u16,    // wq (stage 2 head norm)
            b2.k_norm.as_ptr() as *const u16,    // wk (stage 2 head norm)
            1,                                   // li (layer 1 — no gather)
            2,                                   // n_layers (stage_ts slots)
            POS,                                 // rope pos
            0,                                   // token (unused at li > 0)
        )
        .unwrap();
        stream.synchronize().unwrap();

        // Same output surface as the li=0 gate: q/k/v after head norm +
        // rope, the flash attention output, down, the residual stream x
        // (the split's layer-1 input was the same initial x — b2.x_f was
        // untouched by run_split), the attn-norm output, and the 7
        // phase-1 partials segments (shared segments — same base, no
        // offset).
        assert_u16_eq(&read_u16(&b2.q16, NQK), &read_u16(&b2.q16_f, NQK), "li1 q16");
        assert_u16_eq(&read_u16(&b2.k16, KVK), &read_u16(&b2.k16_f, KVK), "li1 k16");
        assert_u16_eq(&read_u16(&b2.v16, KVK), &read_u16(&b2.v16_f, KVK), "li1 v16");
        assert_u16_eq(&read_u16(&b2.attn, NQK), &read_u16(&b2.attn_f, NQK), "li1 attn");
        assert_u16_eq(&read_u16(&b2.down_s, FFN), &read_u16(&b2.down_f, FFN), "li1 down");
        assert_u16_eq(&read_u16(&b2.x_s, H), &read_u16(&b2.x_f, H), "li1 x");
        assert_u16_eq(&read_u16(&b2.xn_attn_s, H), &read_u16(&b2.xn_attn_f, H), "li1 xn_attn");
        assert_f32_eq(
            &read_f32(&b2.p_q, NQK * g.nslabs_q as usize),
            &read_f32_ptr(dev, g.pq, NQK * g.nslabs_q as usize),
            "li1 partials q",
        );
        assert_f32_eq(
            &read_f32(&b2.p_k, KVK * g.nslabs_k as usize),
            &read_f32_ptr(dev, g.pk, KVK * g.nslabs_k as usize),
            "li1 partials k",
        );
        assert_f32_eq(
            &read_f32(&b2.p_v, KVK * g.nslabs_v as usize),
            &read_f32_ptr(dev, g.pv, KVK * g.nslabs_v as usize),
            "li1 partials v",
        );
        assert_f32_eq(
            &read_f32(&b2.p_o, H * g.nslabs_o as usize),
            &read_f32_ptr(dev, g.po, H * g.nslabs_o as usize),
            "li1 partials o",
        );
        assert_f32_eq(
            &read_f32(&b2.p_g, FFN * g.nslabs_g as usize),
            &read_f32_ptr(dev, g.pg, FFN * g.nslabs_g as usize),
            "li1 partials gate",
        );
        assert_f32_eq(
            &read_f32(&b2.p_u, FFN * g.nslabs_g as usize),
            &read_f32_ptr(dev, g.pu, FFN * g.nslabs_g as usize),
            "li1 partials up",
        );
        assert_f32_eq(
            &read_f32(&b2.p_d, H * g.nslabs_d as usize),
            &read_f32_ptr(dev, g.pd, H * g.nslabs_d as usize),
            "li1 partials down",
        );
        // The stage-3 kv write: physical page 1 (li=1, pages2 = [0, 1])
        // must hold the same k16/v16 the split's kv_write_row stored at
        // page 0 — same slot, same values.
        let kv2_r = read_u16(&d_kv2, 2 * BLOCK_LEN * per_tok_kv * 2);
        let slot1 = (BLOCK_LEN + KV_LEN - 1) * per_tok_kv; // phys 1, off KV_LEN-1
        let v_base1 = 2 * BLOCK_LEN * per_tok_kv;
        let slot0 = (KV_LEN - 1) * per_tok_kv; // phys 0
        let v_base0 = BLOCK_LEN * per_tok_kv;
        assert_u16_eq(
            &kv2_r[slot1..slot1 + per_tok_kv],
            &kv_s[slot0..slot0 + per_tok_kv],
            "li1 kv K write",
        );
        assert_u16_eq(
            &kv2_r[v_base1 + slot1..v_base1 + slot1 + per_tok_kv],
            &kv_s[v_base0 + slot0..v_base0 + slot0 + per_tok_kv],
            "li1 kv V write",
        );
        // And page 0 must NOT hold the new k/v — a page-offset write bug
        // would land at page-0 slot KV_LEN-1 (slot0), which still holds
        // the different page-A prefill when the write went to page 1.
        assert_ne!(
            &kv2_r[slot0..slot0 + per_tok_kv],
            &kv_s[slot0..slot0 + per_tok_kv],
            "li1 kv write must land in page 1, not page 0"
        );
        println!(
            "layer-fused li=1 kernel: bit-exact vs split on q/k/v, attn, x, xn_attn, down, all 7 phase-1 partials segments and the page-1 kv write"
        );
    }

    /// Gate 2: two fused layer runs from identical inputs are
    /// bit-identical (determinism).
    #[test]
    #[ignore = "gpu.yml: l3-fused-decode / determinism"]
    fn fused_determinism_double_run() {
        let (ctx, stream, arch, cache) = setup();
        let dev = ctx.device_id().index();
        let decode = DecodeKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let jgemm = Jgemm::new(dev, &arch, Some(cache.clone())).unwrap();
        let gemm_lib = load_gemm_m1(&arch, Some(cache.clone()));
        let mut fused = FusedDecodeKernels::new(
            dev,
            &arch,
            Some(cache),
            gemm_lib.kernel("gemv_m1_f16f32_reduce").unwrap(),
        )
        .unwrap();

        let b1 = layer_bufs(dev, 0xD37u64);
        let b2 = layer_bufs(dev, 0xD37u64);
        run_fused(&mut fused, &jgemm, &ctx, &decode, &stream, dev, &b1);
        run_fused(&mut fused, &jgemm, &ctx, &decode, &stream, dev, &b2);
        stream.synchronize().unwrap();

        assert_u16_eq(&read_u16(&b1.attn_f, NQK), &read_u16(&b2.attn_f, NQK), "det attn");
        assert_u16_eq(&read_u16(&b1.q16_f, NQK), &read_u16(&b2.q16_f, NQK), "det q16");
        assert_u16_eq(&read_u16(&b1.k16_f, KVK), &read_u16(&b2.k16_f, KVK), "det k16");
        assert_u16_eq(&read_u16(&b1.v16_f, KVK), &read_u16(&b2.v16_f, KVK), "det v16");
        assert_u16_eq(&read_u16(&b1.xn_ffn_f, H), &read_u16(&b2.xn_ffn_f, H), "det xn_ffn");
        assert_u16_eq(&read_u16(&b1.down_f, FFN), &read_u16(&b2.down_f, FFN), "det down");
        assert_u16_eq(&read_u16(&b1.x_f, H), &read_u16(&b2.x_f, H), "det x");
        assert_u16_eq(&read_u16(&b1.xn_attn_f, H), &read_u16(&b2.xn_attn_f, H), "det xn_attn");
        println!("fused layer: two runs bit-identical (deterministic)");
    }

    /// Gate 3 (S1-10): two persistent-kernel layer runs from identical
    /// inputs are bit-identical (determinism — the grid barriers must
    /// not leak run-order effects).
    #[test]
    #[ignore = "gpu.yml: l3-fused-decode / determinism"]
    fn layer_fused_determinism_double_run() {
        let (ctx, stream, arch, cache) = setup();
        let dev = ctx.device_id().index();
        let dense = DenseKernels::new(&arch, Some(cache.clone())).unwrap();
        let diff = DiffKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let decode = DecodeKernels::new(&arch, Some(cache.clone()), stream.clone()).unwrap();
        let jgemm = Jgemm::new(dev, &arch, Some(cache.clone())).unwrap();
        let gemm_lib = load_gemm_m1(&arch, Some(cache.clone()));
        let mut fused = FusedDecodeKernels::new(
            dev,
            &arch,
            Some(cache.clone()),
            gemm_lib.kernel("gemv_m1_f16f32_reduce").unwrap(),
        )
        .unwrap();
        let mut lf = LayerFusedKernels::new(dev, &arch, Some(cache)).unwrap();

        let b1 = layer_bufs(dev, 0xD37u64);
        let b2 = layer_bufs(dev, 0xD37u64);
        // Per pair: the split reference (initializes b.attn — the o
        // plan's input — and the reference partials), then the layer
        // kernel over that pair's plan table (the plan rows and the
        // kernel const hold pointers into the pair's buffers).
        for b in [&b1, &b2] {
            run_split(&gemm_lib, &dense, &diff, &decode, &jgemm, &stream, dev, b);
            fused.build_plans(ctx.device_id(), &jgemm, &fabricate_plans(b)).unwrap();
            let g = fused.geom();
            run_layer_fused(&mut lf, &ctx, &stream, b, g);
        }
        stream.synchronize().unwrap();

        assert_u16_eq(&read_u16(&b1.q16_f, NQK), &read_u16(&b2.q16_f, NQK), "det layer q16");
        assert_u16_eq(&read_u16(&b1.k16_f, KVK), &read_u16(&b2.k16_f, KVK), "det layer k16");
        assert_u16_eq(&read_u16(&b1.v16_f, KVK), &read_u16(&b2.v16_f, KVK), "det layer v16");
        assert_u16_eq(&read_u16(&b1.attn_f, NQK), &read_u16(&b2.attn_f, NQK), "det layer attn");
        assert_u16_eq(&read_u16(&b1.down_f, FFN), &read_u16(&b2.down_f, FFN), "det layer down");
        assert_u16_eq(&read_u16(&b1.x_f, H), &read_u16(&b2.x_f, H), "det layer x");
        assert_u16_eq(
            &read_u16(&b1.xn_attn_f, H),
            &read_u16(&b2.xn_attn_f, H),
            "det layer xn_attn",
        );
        println!("layer-fused: two runs bit-identical (deterministic)");
    }

    // -----------------------------------------------------------------------
    // Gate 3: engine A/B — REINFER_FUSED on vs off.
    // -----------------------------------------------------------------------

    const FUSED_ENV: &str = "REINFER_FUSED";
    const LAYER_FUSED_ENV: &str = "REINFER_LAYER_FUSED";

    fn model_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR"))
    }

    /// Load the engine with REINFER_FUSED=`fused_value` and
    /// REINFER_LAYER_FUSED=`layer_value` (both read once at load; the
    /// process env is ours — tests must run single-threaded).
    fn load_pair(fused_value: &str, layer_value: &str) -> Engine {
        // SAFETY (test-only): --test-threads=1 keeps the env mutation
        // single-threaded; the value is read once at Engine::load.
        unsafe {
            std::env::set_var(FUSED_ENV, fused_value);
            std::env::set_var(LAYER_FUSED_ENV, layer_value);
        }
        let ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).expect("stream");
        let _ = stream.synchronize().expect("sync");
        Engine::load(
            dev,
            &reinfer_cuda::arch::resolve_arch().expect("arch"),
            Some(std::env::temp_dir().join("reinfer-jit-fused-engine")),
            &model_dir(),
            4096,
        )
        .expect("engine load")
    }

    /// `load_pair` with REINFER_LAYER_FUSED=off (the S1-9 fused path —
    /// the default is on, so the off arm must be explicit).
    fn load(fused_value: &str) -> Engine {
        load_pair(fused_value, "off")
    }

    fn tokenizer() -> Tokenizer {
        let dir = model_dir();
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("tokenizer_config.json")).expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        Tokenizer::from_hf_json(&tok, &cfg).expect("hf tokenizer")
    }

    /// Dense per-token prefill of the prompt (graph_engine.rs convention)
    /// + `n` greedy (t=0) decode steps: per-step logits and the token
    /// stream (the decode logits only).
    fn run(eng: &mut Engine, ids: &[u32], n: usize) -> (Vec<Vec<f32>>, Vec<u32>) {
        for (i, &t) in ids.iter().enumerate() {
            eng.step(t, i, i + 1).unwrap();
        }
        let mut logits = Vec::with_capacity(n);
        let mut toks = Vec::with_capacity(n);
        let mut cur = *ids.last().unwrap();
        let mut pos = ids.len();
        for _ in 0..n {
            let lg = eng.step(cur, pos, pos + 1).unwrap();
            let next = argmax_first(&lg);
            logits.push(lg);
            toks.push(next);
            cur = next;
            pos += 1;
        }
        (logits, toks)
    }

    fn argmax_first(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, l) in logits.iter().enumerate().skip(1) {
            if l > &logits[best] {
                best = i;
            }
        }
        best as u32
    }

    fn assert_bitwise(a: &[Vec<f32>], b: &[Vec<f32>], tag: &str) {
        assert_eq!(a.len(), b.len(), "{tag}: step count mismatch");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.len(), y.len(), "{tag}: step {i}: logits len");
            for (j, (v, w)) in x.iter().zip(y.iter()).enumerate() {
                assert_eq!(
                    v.to_bits(),
                    w.to_bits(),
                    "{tag}: step {i}: logits[{j}] bit mismatch {v} vs {w}"
                );
            }
        }
    }

    /// Fused vs split engine: 16 + 128 greedy decode tokens, per-step
    /// logits bit-identical and identical text; the fused engine is
    /// deterministic across two separate loads (bit-level).
    #[test]
    #[ignore = "gpu.yml: l3-fused-decode / engine-ab"]
    fn fused_engine_ab_bitwise() {
        let tok = tokenizer();
        let ids = tok.encode("Hello", false).expect("encode");
        assert!(!ids.is_empty());

        // A: fused — two separate loads must produce bit-identical steps.
        let mut eng_a1 = load("on");
        let (l1, t1) = run(&mut eng_a1, &ids, 16);
        drop(eng_a1);
        let mut eng_a2 = load("on");
        let (l2, t2) = run(&mut eng_a2, &ids, 16);
        assert_eq!(t1, t2, "fused determinism: identical 16-token text");
        assert_bitwise(&l1, &l2, "fused determinism");
        drop(eng_a2);

        // B: split reference arm (REINFER_FUSED=off).
        let mut eng_off = load("off");
        let (l_off, t_off) = run(&mut eng_off, &ids, 128);
        drop(eng_off);
        let mut eng_on = load("on");
        let (l_on, t_on) = run(&mut eng_on, &ids, 128);
        drop(eng_on);

        assert_eq!(t_on, t_off, "fused on/off must produce identical text");
        assert_bitwise(&l_on, &l_off, "fused on/off");
        println!(
            "fused A/B: {} decode steps bit-identical; text {:?}",
            t_on.len(),
            tok.decode_all(&t_on)
        );
    }

    /// Gate 4 (S1-10): engine A/B — REINFER_LAYER_FUSED=on (the single
    /// per-layer persistent kernel) vs off (the S1-9 eight-kernel fused
    /// path), both with REINFER_FUSED=on: 16 + 128 greedy decode tokens,
    /// per-step logits bit-identical and identical text; the layer-fused
    /// engine is deterministic across two separate loads (bit-level).
    #[test]
    #[ignore = "gpu.yml: l3-fused-decode / engine-ab"]
    fn layer_fused_engine_ab_bitwise() {
        let tok = tokenizer();
        let ids = tok.encode("Hello", false).expect("encode");
        assert!(!ids.is_empty());

        // A: layer-fused — two separate loads must produce bit-identical
        // steps.
        let mut eng_a1 = load_pair("on", "on");
        let (l1, t1) = run(&mut eng_a1, &ids, 16);
        drop(eng_a1);
        let mut eng_a2 = load_pair("on", "on");
        let (l2, t2) = run(&mut eng_a2, &ids, 16);
        assert_eq!(t1, t2, "layer-fused determinism: identical 16-token text");
        assert_bitwise(&l1, &l2, "layer-fused determinism");
        drop(eng_a2);

        // B: the S1-9 fused reference arm (REINFER_LAYER_FUSED=off).
        let mut eng_off = load_pair("on", "off");
        let (l_off, t_off) = run(&mut eng_off, &ids, 128);
        drop(eng_off);
        let mut eng_on = load_pair("on", "on");
        let (l_on, t_on) = run(&mut eng_on, &ids, 128);
        drop(eng_on);

        assert_eq!(t_on, t_off, "layer on/off must produce identical text");
        assert_bitwise(&l_on, &l_off, "layer on/off");
        println!(
            "layer-fused A/B: {} decode steps bit-identical; text {:?}",
            t_on.len(),
            tok.decode_all(&t_on)
        );
    }
}
