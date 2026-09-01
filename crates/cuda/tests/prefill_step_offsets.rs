//! 016 T1a-test: `Engine::step(tok, pos, pos+1)` 全窗语义固化（v1 命中路径
//! "copy 前缀 + 逐 token step" 的地基——r2 修订：无新引擎 API、无内核改动）。
//!
//! Gates:
//! ① `kv_write_page_boundaries` — KV 写入跨页边界寻址正确：pos ∈
//!    {0, 31, 32, 63, 64} 分别经 step 跳写，直接读引擎 kv_store 的
//!    k_ptr/v_ptr 页内容，断言每个 pos 的字节落在物理页 `li*pp + pos/32`、
//!    槽位 `pos%32`（与"逐步从 0 写"的同位置字节位级一致——寻址错位
//!    （页号/槽位/层偏移任何一处）必不匹配）。
//! ② `decode_window_full_range_bitwise` — 解码步注意力读 [0, pos+1]：
//!    同一 tiny prompt（≤64 token）三条路径——(a) 逐步全写；(b) 前缀
//!    逐步写 + D2D 复制前缀页 + 跳写 pos=N（v1 命中路径形态）；(c) 逐步
//!    写前缀 + 跳写 pos=N——在 pos=N 处 logits 位级一致（同内核同输入），
//!    池中已写槽位逐位相同，且此后 greedy（t=0）8 token 续文全等。
//! ③ `decode_window_reads_copied_early_pages` — 全窗读正向证据：把池中
//!    页 0（远前缀 [0, 32)）内容替换为另一 token 序列的 KV（D2D），
//!    pos=N 处 logits 必须变化（若解码只读近窗/末页，logits 会位级不变
//!    ——那将是对"注意力全窗读"的意外发现）。
//!
//! Run (RTX 5090 / sm_120a JIT env):
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc \
//! REINFER_MODEL_DIR=/home/dora/.reinfer/models/Qwen/Qwen3-0.6B \
//! cargo test -p reinfer-cuda --features cuda --test prefill_step_offsets -- \
//!     --ignored --test-threads=1 --nocapture
//! ```

#![cfg(feature = "cuda")]
#![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
#![allow(clippy::print_stdout)] // 记录档输出

mod gpu {
    use reinfer_core::DeviceId;
    use reinfer_cuda::CudaContext;
    use reinfer_cuda::buffer::HostBuffer;
    use reinfer_cuda::engine::{Engine, argmax_first};
    use reinfer_tokenizer::Tokenizer;
    use std::path::PathBuf;

    /// Decode-step KV page size — mirrors engine.rs `BLOCK_LEN` (32).
    const BLOCK_LEN: usize = 32;

    fn model_dir() -> PathBuf {
        PathBuf::from(std::env::var("REINFER_MODEL_DIR").expect("REINFER_MODEL_DIR"))
    }

    fn tokenizer() -> Tokenizer {
        let tok: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir().join("tokenizer.json")).expect("tokenizer.json"),
        )
        .expect("tokenizer json");
        let tcfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(model_dir().join("tokenizer_config.json"))
                .expect("tokenizer_config.json"),
        )
        .expect("tokcfg json");
        Tokenizer::from_hf_json(&tok, &tcfg).expect("hf tokenizer")
    }

    fn load(dev: u32, max_kv: usize) -> Engine {
        let arch = reinfer_cuda::arch::resolve_arch().expect("arch");
        Engine::load(
            DeviceId::new(dev),
            &arch,
            Some(std::env::temp_dir().join("reinfer-jit-prefill-offsets")),
            &model_dir(),
            max_kv,
        )
        .expect("engine load")
    }

    /// Long-enough prompt for the 66-token (page-boundary) and 34-token
    /// (decode-window) acceptance windows.
    const PROMPT: &str = "The quick brown fox jumps over the lazy dog near \
        the river while the autumn leaves drift slowly down and settle on \
        the quiet ground and the wind carries the first cold breath of \
        evening across the empty meadow and beyond the distant hills the \
        silver moon rises slowly over the sleeping valleys while the stars \
        begin to blink one by one in the dark blue sky and the whole night \
        settles softly down upon the land";

    fn bitwise(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    // ---------------- engine KV page readback (k_ptr / v_ptr) ----------------

    /// 层 `li` 位置 `pos` 的 K/V 元素（页布局序：((li*pp + pos/32)*32 +
    /// pos%32)*per_tok + kh*head_dim + di；K 区在前，V 区在
    /// total_pages*32*per_tok 元素之后——与 kv_write 内核同址）。
    fn kv_pos(eng: &Engine, li: usize, pos: usize) -> (Vec<u16>, Vec<u16>) {
        let cfg = eng.config();
        let kv = eng.kv_store();
        let per_tok = cfg.kv_heads * cfg.head_dim;
        let pp = kv.total_pages / cfg.n_layer;
        let (p, o) = (pos / BLOCK_LEN, pos % BLOCK_LEN);
        let base = ((li * pp + p) * BLOCK_LEN + o) * per_tok;
        let k_region = kv.total_pages * BLOCK_LEN * per_tok; // elements per region
        let bytes = per_tok * 2;
        let read = |off_elems: usize| -> Vec<u16> {
            let hb = HostBuffer::alloc(bytes).unwrap();
            let rc = unsafe {
                reinfer_cuda::_cudarc::runtime::sys::cudaMemcpy(
                    hb.as_ptr() as *mut std::ffi::c_void,
                    (kv.data.as_ptr() as *const u8).add(off_elems * 2) as *const std::ffi::c_void,
                    bytes,
                    reinfer_cuda::_cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            rc.result().unwrap();
            unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const u16, per_tok).to_vec() }
        };
        (read(base), read(k_region + base))
    }

    /// 位置 `pos` 的 K/V 逐位比较（全层；返回首个不匹配的 (层, 元素)）。
    fn kv_pos_eq(a: &Engine, b: &Engine, pos: usize) -> Option<(usize, usize)> {
        for li in 0..a.config().n_layer {
            let (ka, va) = kv_pos(a, li, pos);
            let (kb, vb) = kv_pos(b, li, pos);
            let n = ka.len();
            for (i, (x, y)) in ka.iter().zip(&kb).enumerate() {
                if x != y {
                    return Some((li, i));
                }
            }
            for (i, (x, y)) in va.iter().zip(&vb).enumerate() {
                if x != y {
                    return Some((li, n + i));
                }
            }
        }
        None
    }

    /// 全层位置 [pos0, pos1) 中每个已写位置逐位比较（同 `kv_pos_eq` 语义，
    /// 返回首个不匹配的 (位置, 层, 元素)）。
    fn kv_pos_range_eq(
        a: &Engine,
        b: &Engine,
        pos0: usize,
        pos1: usize,
    ) -> Option<(usize, usize, usize)> {
        for pos in pos0..pos1 {
            if let Some((li, i)) = kv_pos_eq(a, b, pos) {
                return Some((pos, li, i));
            }
        }
        None
    }

    /// D2D 复制 `src` 引擎池中位置 [pos0, pos1) 覆盖的页（K 区 + V 区，
    /// 全层）到 `dst` 引擎池（v1 命中路径的前缀复制模拟——层步长 pp
    /// 编址，与 copy_prefix_to_engine 同布局）。调用前两侧引擎的流须
    /// 已同步（step 内部同步；本函数以同步 cudaMemcpy 执行）。
    fn copy_kv_pos_range(src: &Engine, dst: &Engine, pos0: usize, pos1: usize) {
        let sc = src.config();
        let kv = src.kv_store();
        let dkv = dst.kv_store();
        assert_eq!(kv.total_pages, dkv.total_pages, "KvStore geometry mismatch");
        let per_tok = sc.kv_heads * sc.head_dim;
        let pp = kv.total_pages / sc.n_layer;
        let (p0, p1) = (pos0 / BLOCK_LEN, pos1.div_ceil(BLOCK_LEN));
        let k_region = kv.total_pages * BLOCK_LEN * per_tok; // elements per region
        let n_elems = (p1 - p0) * BLOCK_LEN * per_tok;
        let bytes = n_elems * 2;
        for li in 0..sc.n_layer {
            let off_bytes = ((li * pp + p0) * BLOCK_LEN * per_tok) * 2;
            for region in 0..2 {
                let rc = unsafe {
                    reinfer_cuda::_cudarc::runtime::sys::cudaMemcpy(
                        (dkv.data.as_ptr() as *mut u8).add(off_bytes + region * k_region * 2)
                            as *mut std::ffi::c_void,
                        (kv.data.as_ptr() as *const u8).add(off_bytes + region * k_region * 2)
                            as *const std::ffi::c_void,
                        bytes,
                        reinfer_cuda::_cudarc::runtime::sys::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    )
                };
                rc.result().unwrap();
            }
        }
    }

    // ============ ① KV 写入跨页边界寻址正确 ============

    /// 跳写 pos ∈ {31, 32, 63, 64}（页内首/尾 31/63、页边界两侧 31→32
    /// 与 63→64、跨两层页 64→页 2）：直接读回 kv_store 的 k_ptr/v_ptr
    /// 页内容，断言每个 pos 的字节落在物理页 `li*pp + pos/32`、槽位
    /// `pos%32`——与"逐步从 0 写"的同位置字节**全层**位级一致。
    ///
    /// 关键约束：跳写前前缀 [0, pos) 必须完整（逐步写满），否则上层
    /// 层间依赖会让层 ≥1 的 KV 值不同（层 0 注意力读了垃圾槽 → 输出
    /// 不同 → 上层输入不同 → 上层 KV 不同）——这本身正是 v1 命中路径
    /// "整前缀 copy、不可部分 copy" 的证据。
    #[test]
    #[ignore = "gpu.yml: prefill-step-offsets"]
    fn kv_write_page_boundaries() {
        let _ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let ids = tokenizer().encode(PROMPT, false).expect("encode");
        assert!(ids.len() >= 66, "prompt must encode to >= 66 tokens (got {})", ids.len());
        // 参考：逐步从 0 写全窗 [0, 65)。
        let mut eng_ref = load(0, 256);
        for (i, &t) in ids[..65].iter().enumerate() {
            eng_ref.step(t, i, i + 1).expect("ref step");
        }
        // 跳写引擎：前缀 [0, 63) 逐步写满（pos 0..62——跳写 63 时窗
        // [0, 64) 中除当前 token 外必须无缺口，否则层 0 注意力读了垃圾
        // 槽 → 上层 KV 不同，与寻址无关），然后每个目标 pos 以单次
        // step(tok, pos, pos+1) 直接跳写（31、32 的前缀已在逐步范围内；
        // 63 先于 64，保证 64 的前缀 [0, 64) 完整）。每个跳写步的前向
        // 输入与 ref 的对应步完全相同（同池内容 + 同当前 token），因此
        // 全部 n_layer 层的 KV 都必须位级一致——否则写入落在了错误的
        // 物理页/槽位。
        let mut eng_spot = load(0, 256);
        for (i, &t) in ids[..63].iter().enumerate() {
            eng_spot.step(t, i, i + 1).expect("spot prefix step");
        }
        for &pos in &[31usize, 32, 63, 64] {
            eng_spot.step(ids[pos], pos, pos + 1).expect("spot jump step");
        }
        for pos in [31usize, 32, 63, 64] {
            let mm = kv_pos_eq(&eng_spot, &eng_ref, pos);
            assert!(
                mm.is_none(),
                "pos {pos}: spot write landed on a different physical slot \
                 than the full-window reference (first mismatch: layer {}, elem {})",
                mm.map(|m| m.0).unwrap_or(0),
                mm.map(|m| m.1).unwrap_or(0)
            );
        }
        println!(
            "kv_write_page_boundaries: jump pos {{31,32,63,64}} -> pages li*pp + pos/32, \
             slot pos%32 bit-identical (all {} layers) to the full-window write",
            eng_ref.config().n_layer
        );
    }

    // ============ ② 解码步注意力读 [0, pos+1]（位级） ============

    /// 三条路径在 pos=N 处 logits 位级一致：(a) 逐步全写；(b) 前缀逐步
    /// 写 + D2D 复制前缀页 + 跳写 N（v1 命中路径形态）；(c) 逐步写前缀
    /// + 跳写 N。池中已写槽位逐位相同；此后 greedy 8 token 续文全等。
    /// N = 33（跨页边界：页 1 槽 1）。
    #[test]
    #[ignore = "gpu.yml: prefill-step-offsets"]
    fn decode_window_full_range_bitwise() {
        let _ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let ids = tokenizer().encode(PROMPT, false).expect("encode");
        assert!(ids.len() >= 66, "prompt must encode to >= 66 tokens (got {})", ids.len());
        let n = 33usize;
        // (a) 逐步全写 [0, n+1)。
        let mut full = load(0, 256);
        for (i, &t) in ids[..=n].iter().enumerate() {
            full.step(t, i, i + 1).expect("full step");
        }
        let lg_full = full.step(ids[n], n, n + 1).expect("full decode step");
        // (b) v1 命中形态：前缀 [0, n) 由另一引擎逐步写产生，D2D 复制
        // 进 hit 引擎池，再跳写 pos=n。
        let mut pre = load(0, 256);
        for (i, &t) in ids[..n].iter().enumerate() {
            pre.step(t, i, i + 1).expect("prefix step");
        }
        let mut hit = load(0, 256);
        copy_kv_pos_range(&pre, &hit, 0, n);
        hit.step(ids[n], n, n + 1).expect("hit jump step");
        let lg_hit = hit.step(ids[n], n, n + 1).expect("hit decode step");
        // (c) 逐步写前缀 + 跳写 pos=n（同引擎内直接跳写）。
        let mut win = load(0, 256);
        for (i, &t) in ids[..n].iter().enumerate() {
            win.step(t, i, i + 1).expect("win prefix step");
        }
        win.step(ids[n], n, n + 1).expect("win jump step");
        let lg_win = win.step(ids[n], n, n + 1).expect("win decode step");

        // 池：已写槽位 [0, n+1) 全部逐位相同（覆盖页 0 槽 0..31、页 1
        // 槽 0..1——跨页边界）。
        for (name, eng) in [("hit", &hit), ("win", &win)] {
            let mm = kv_pos_range_eq(&full, eng, 0, n + 1);
            assert!(
                mm.is_none(),
                "{name}: KV divergence vs full-window write at pos {} layer {} elem {}",
                mm.map(|m| m.0).unwrap_or(0),
                mm.map(|m| m.1).unwrap_or(0),
                mm.map(|m| m.2).unwrap_or(0)
            );
        }
        // logits：位级一致。
        assert!(
            bitwise(&lg_full, &lg_hit),
            "hit path diverged from full-window decode bitwise (max |diff| {})",
            max_diff(&lg_full, &lg_hit)
        );
        assert!(
            bitwise(&lg_full, &lg_win),
            "win path diverged from full-window decode bitwise (max |diff| {})",
            max_diff(&lg_full, &lg_win)
        );
        // greedy（t=0）续 token：三路全等。
        let mut texts = [vec![lg_full], vec![lg_hit], vec![lg_win]];
        for k in 0..8usize {
            let pos = n + 1 + k;
            for (e, lg) in texts.iter_mut().enumerate() {
                let t = argmax_first(&lg[0]);
                lg[0] = match e {
                    0 => full.step(t, pos, pos + 1).expect("full greedy step"),
                    1 => hit.step(t, pos, pos + 1).expect("hit greedy step"),
                    _ => win.step(t, pos, pos + 1).expect("win greedy step"),
                };
            }
            let (a, b, c) = (&texts[0][0], &texts[1][0], &texts[2][0]);
            assert!(
                bitwise(a, b) && bitwise(a, c),
                "greedy step {k} (pos {pos}) diverged across paths (full/hit/win)"
            );
        }
        println!(
            "decode_window_full_range_bitwise: full / hit(copy+step) / win(jump) at pos={n} \
             — pool slots + logits bit-identical, 8 greedy tokens identical"
        );
    }

    // ============ ③ 全窗读正向证据：远前缀页参与注意力 ============

    /// 把池中页 0（远前缀 [0, 32)）内容替换为**另一 token 序列**的 KV：
    /// pos=N 处 logits 必须变化（位级）。若解码注意力只读近窗/末页，
    /// 替换后 logits 会位级不变——即"注意力非全窗读"的意外发现。
    #[test]
    #[ignore = "gpu.yml: prefill-step-offsets"]
    fn decode_window_reads_copied_early_pages() {
        let _ctx = CudaContext::init(DeviceId::new(0)).expect("ctx");
        let ids = tokenizer().encode(PROMPT, false).expect("encode");
        assert!(ids.len() >= 66, "prompt must encode to >= 66 tokens (got {})", ids.len());
        let n = 33usize;
        // 基线：逐步全写 [0, n+1)。
        let mut full = load(0, 256);
        for (i, &t) in ids[..=n].iter().enumerate() {
            full.step(t, i, i + 1).expect("full step");
        }
        let lg_full = full.step(ids[n], n, n + 1).expect("full decode step");
        // 篡改源：另一 token 序列（取主 PROMPT 后段 ids[34..66]——与
        // ids[..32] 不同序列，保证页 0 内容不同）的页 0 KV。
        let other = ids[34..66].to_vec();
        assert!(other.len() >= 32, "need >= 32 tokens for the tamper page");
        let same = other[..32].iter().zip(&ids[..32]).all(|(a, b)| a == b);
        assert!(!same, "other prompt collides with the main prompt head");
        let mut alt = load(0, 256);
        for (i, &t) in other[..32].iter().enumerate() {
            alt.step(t, i, i + 1).expect("alt step");
        }
        // 篡改引擎：逐步写 ids[0..=32]（页 1 槽 0 真实），再把页 0 覆盖
        // 为 other 序列的页 0 → 池 = [页0: 不同内容] + [页1: 真实 ids[32]]。
        let mut tam = load(0, 256);
        for (i, &t) in ids[..=32].iter().enumerate() {
            tam.step(t, i, i + 1).expect("tam step");
        }
        copy_kv_pos_range(&alt, &tam, 0, 32);
        // 篡改生效验证：页 0 内容确实不同。
        assert!(
            kv_pos_range_eq(&full, &tam, 0, 32).is_some(),
            "tamper copy did not change page 0 content"
        );
        // 页 1 槽 0（pos 32）仍与 full 位级相同（篡改只动页 0）。
        assert!(kv_pos_eq(&full, &tam, 32).is_none(), "tamper copy clobbered page 1 (pos 32)");
        let lg_tam = tam.step(ids[n], n, n + 1).expect("tam decode step");
        let diff = max_diff(&lg_full, &lg_tam);
        assert!(
            !bitwise(&lg_full, &lg_tam),
            "decode at pos={n} did not move when the early page [0, 32) content \
             changed (window may not span the full [0, n+1) range) — max|diff| {diff:e}"
        );
        println!(
            "decode_window_reads_copied_early_pages: page-0 tamper changed pos={n} \
             logits (max|diff| {diff:e}) — attention spans the copied early pages"
        );
    }
}
