//! Request-scoped KV segment pool (spec 005 S2-C: the KV pool budget layer).
//!
//! The CUDA engine's KV store is one contiguous device buffer of `total`
//! physical pages laid out **layer-major**: layer `li` owns physical pages
//! `[li*pp, (li+1)*pp)` with `pp = ceil(max_kv / block_len)` (engine.rs S1-2,
//! static identity page table — single-request semantics). Serving multiple
//! concurrent requests requires **segment allocation**: a contiguous run of
//! pages granted to one request.
//!
//! Recommended request-major mapping inside a segment of
//! `n_layer * pp_req` pages (compatible with the layer-major device layout):
//!
//! ```text
//! segment [b, b + n_layer*pp_req)
//!   layer li pages  = [b + li*pp_req, b + (li+1)*pp_req)
//!   page table (li) = b + li*pp_req + j          (j in 0..pp_req)
//! ```
//!
//! with `pp_req = ceil(req_len / block_len)`. **NOTE (2026-09-01)**: the
//! *current* scheduler allocates every request segment at the **full window**
//! size (`n_layer × ceil(max_kv/block_len)`, see bin/reinfer sched_loop
//! `alloc_segment` → `window_pages()`), so the layer stride in practice is
//! the full-window `pp` — every segment carries layered "tail holes"
//! (pp − pp_req unused pages per layer). This module's suggestion below is
//! the prospective per-request-len mapping (a future refinement); do not
//! implement against `pp_req` while consumers compute layer offsets with
//! the full-window `pp` (the identity table `li*pp + j` contract).
//!
//! The per-request page tables
//! replace the engine's static identity table; `kv_write` then receives
//! `phys = seg_base + li*pp_req + lp` per (request, layer) — the wiring is
//! the scheduler-executor wave's job (this crate only hands out segments).
//!
//! ## Refcount
//!
//! Every physical page carries a reference count (0 = free). `alloc` sets it
//! to 1; [`KvSegmentPool::ref_`] increments it (prefix sharing — P3-01);
//! [`KvSegmentPool::free`] / [`KvSegmentPool::unref`] drop one reference per
//! page and return the pages that reach 0 to the free list (vLLM
//! `BlockAllocator::free` semantics). Pages shared by several requests stay
//! live until every holder releases them.
//!
//! ## Free list and determinism
//!
//! The free list holds coalesced free **runs**, most recently released first.
//! Allocation is first-fit from the front of the list; a run larger than the
//! request is split and its remainder re-inserted at the front (recency).
//! Freeing re-inserts runs at the front and coalesces with any adjacent run,
//! so no two list entries are ever adjacent.
//!
//! **Determinism contract**: the pool must be mutated by a single thread
//! (the scheduler's event loop — plan D1). Given an identical sequence of
//! `alloc`/`free`/`ref_`/`unref` calls the pool returns identical segments
//! (no RNG, no interior mutation, no hash-map iteration). The caller decides
//! the call order; the pool only guarantees the mapping call-sequence →
//! result.
//!
//! ## Conservation invariants
//!
//! `used + available == total`, `used == allocs - frees` (page symmetry),
//! free runs are disjoint, in-bounds and never adjacent (checked by
//! [`KvSegmentPool::assert_conserved`] — the 014 T8 three-in-one criterion).

use crate::pool::{BlockLen, MemoryError};

/// A contiguous run of physical pages in the KV pool.
///
/// `n_pages == 0` denotes the empty segment: `free`/`ref_`/`unref` treat it
/// as a no-op, `alloc` refuses to create one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSegment {
    /// First physical page of the segment.
    pub base_page: usize,
    /// Number of contiguous pages.
    pub n_pages: usize,
}

impl KvSegment {
    /// One past the last page (`base_page + n_pages`).
    pub const fn end(&self) -> usize {
        self.base_page + self.n_pages
    }

    /// Physical page indices covered by this segment (ascending).
    pub const fn pages(&self) -> std::ops::Range<usize> {
        self.base_page..self.end()
    }

    /// Empty segment (no pages).
    pub const fn is_empty(&self) -> bool {
        self.n_pages == 0
    }

    /// Whether the segment lies entirely inside a pool of `total` pages.
    pub const fn in_bounds(&self, total: usize) -> bool {
        self.end() <= total
    }
}

/// A free run in the segment free list (invariant: never adjacent to
/// another run — `insert_run` coalesces).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FreeRun {
    base: usize,
    len: usize,
}

/// KV page pool with per-request segment allocation and per-page refcount.
///
/// See the module docs for the layout, refcount, determinism and
/// conservation contracts.
#[derive(Debug)]
pub struct KvSegmentPool {
    /// Token slots per page (mirrors [`PagePool`](crate::pool::PagePool)).
    block_slots: usize,
    /// Physical pages in the pool.
    total: usize,
    /// Per-page reference count; 0 = free.
    refs: Vec<u32>,
    /// Free runs, most recently released first.
    free: Vec<FreeRun>,
    /// Cumulative pages allocated since construction.
    allocs: usize,
    /// Cumulative pages released (refcount dropped to 0).
    frees: usize,
    /// Cumulative segment-level allocation / free calls.
    seg_allocs: usize,
    seg_frees: usize,
}

/// Runtime pool statistics (the scheduler's `KvPoolStats` — plan.md
/// interface contract `SchedulePolicy::rank(queue, pool: &KvPoolStats)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvPoolStats {
    /// Physical pages in the pool.
    pub total_pages: usize,
    /// Pages with refcount > 0 (live KV).
    pub used_pages: usize,
    /// Free pages.
    pub available_pages: usize,
    /// Number of free runs — external-fragmentation proxy. 0 when fully
    /// used, 1 when the free space is contiguous.
    pub fragments: usize,
    /// Largest contiguous free run, in pages. Segment allocation needs a
    /// single run >= the request, so this can be well below
    /// `available_pages` under fragmentation.
    pub max_free_run: usize,
    /// Cumulative pages allocated since construction.
    pub allocs_pages: usize,
    /// Cumulative pages released (refcount dropped to 0).
    pub frees_pages: usize,
    /// Cumulative segment-level allocations.
    pub allocs_segments: usize,
    /// Cumulative segment-level releases.
    pub frees_segments: usize,
}

impl KvSegmentPool {
    /// New pool of `total` pages, all free (one coalesced run `[0, total)`).
    pub fn new(block: BlockLen, total: usize) -> Self {
        Self {
            block_slots: block.slots(),
            total,
            refs: vec![0; total],
            free: if total == 0 { Vec::new() } else { vec![FreeRun { base: 0, len: total }] },
            allocs: 0,
            frees: 0,
            seg_allocs: 0,
            seg_frees: 0,
        }
    }

    /// Token slots per page (16/32).
    pub fn block_slots(&self) -> usize {
        self.block_slots
    }

    /// Physical pages in the pool.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Free pages.
    pub fn free_count(&self) -> usize {
        self.free.iter().map(|r| r.len).sum()
    }

    /// Pages with refcount > 0.
    pub fn in_use(&self) -> usize {
        self.refs.iter().filter(|&&r| r > 0).count()
    }

    /// Number of free runs (external-fragmentation proxy).
    pub fn fragments(&self) -> usize {
        self.free.len()
    }

    /// Largest contiguous free run, in pages.
    pub fn max_free_run(&self) -> usize {
        self.free.iter().map(|r| r.len).max().unwrap_or(0)
    }

    /// Snapshot of all runtime statistics.
    pub fn stats(&self) -> KvPoolStats {
        KvPoolStats {
            total_pages: self.total,
            used_pages: self.in_use(),
            available_pages: self.free_count(),
            fragments: self.fragments(),
            max_free_run: self.max_free_run(),
            allocs_pages: self.allocs,
            frees_pages: self.frees,
            allocs_segments: self.seg_allocs,
            frees_segments: self.seg_frees,
        }
    }

    /// Conservation assertions (014 T8 three-in-one: conservation formula,
    /// alloc/free symmetry, pool total fixed) plus free-list integrity:
    /// runs in bounds, disjoint, never adjacent, covering exactly the free
    /// pages.
    pub fn assert_conserved(&self) {
        assert_eq!(self.in_use() + self.free_count(), self.total, "page conservation");
        assert_eq!(
            self.in_use(),
            self.allocs - self.frees,
            "used == allocs - frees (alloc/free symmetry)"
        );
        let refs_sum: usize = self.refs.iter().map(|&r| r as usize).sum();
        assert_eq!(refs_sum, self.in_use(), "refcount sum");
        let mut covered = 0usize;
        for r in &self.free {
            assert!(r.base + r.len <= self.total, "free run out of bounds");
            covered += r.len;
        }
        assert_eq!(covered, self.free_count(), "free runs cover the free pages");
        for (i, a) in self.free.iter().enumerate() {
            for b in self.free.iter().skip(i + 1) {
                assert!(
                    a.base >= b.base + b.len || b.base >= a.base + a.len,
                    "overlapping free runs"
                );
                assert!(
                    a.base + a.len != b.base && b.base + b.len != a.base,
                    "adjacent free runs not coalesced"
                );
            }
        }
    }

    /// Allocate a contiguous segment of `n_pages` pages (refcount 1 each).
    ///
    /// First-fit over the recency-ordered free list; oversized runs are
    /// split and the remainder re-inserted at the front. Fails with
    /// [`MemoryError::OutOfPages`] when no single run is large enough — the
    /// caller should consult [`KvSegmentPool::max_free_run`] (contiguity,
    /// not the free total, is the binding constraint) or preempt per D7.
    pub fn alloc(&mut self, n_pages: usize) -> Result<KvSegment, MemoryError> {
        if n_pages == 0 {
            return Err(MemoryError::InvalidLen { n: 0 });
        }
        let Some(idx) = self.free.iter().position(|r| r.len >= n_pages) else {
            return Err(MemoryError::OutOfPages { need: n_pages, free: self.free_count() });
        };
        let run = self.free.remove(idx);
        let seg = KvSegment { base_page: run.base, n_pages };
        if run.len > n_pages {
            // Remainder back to the front — most recently touched run first.
            // Cannot be adjacent to any other run (its parent was maximal).
            self.free.insert(0, FreeRun { base: run.base + n_pages, len: run.len - n_pages });
        }
        for r in &mut self.refs[seg.pages()] {
            debug_assert_eq!(*r, 0, "alloc over a live page");
            *r = 1;
        }
        self.allocs += n_pages;
        self.seg_allocs += 1;
        Ok(seg)
    }

    /// Allocate a contiguous segment of `n_pages` pages at the **end** of a
    /// free run (first-fit from the back of the free list).
    ///
    /// The remainder (the run's head) is re-inserted at the front, so a
    /// pool built as `alloc_from_end(anchor)` followed by front-first
    /// `alloc` calls keeps the anchor at the top and hands out segments
    /// from the bottom — the max segment extent stays pinned at the pool
    /// total for as long as the anchor lives.
    ///
    /// The scheduler-executor uses this for its **anchor window** (005
    /// S2-D): the pool's top window is allocated once at construction and
    /// never freed, so the batch kernels' `pool_pages` (the max segment
    /// extent of the current batch, from which the V-region base is
    /// derived) always equals the pool total and the V region is stable.
    /// Same refcount/recency bookkeeping as [`KvSegmentPool::alloc`]; a
    /// pure function of the call sequence (determinism contract holds).
    pub fn alloc_from_end(&mut self, n_pages: usize) -> Result<KvSegment, MemoryError> {
        if n_pages == 0 {
            return Err(MemoryError::InvalidLen { n: 0 });
        }
        // First run (from the back) that fits — the anchor prefers the very
        // end of the pool.
        let Some((idx, _)) = self.free.iter().enumerate().rev().find(|(_, r)| r.len >= n_pages)
        else {
            return Err(MemoryError::OutOfPages { need: n_pages, free: self.free_count() });
        };
        let run = self.free.remove(idx);
        let seg = KvSegment { base_page: run.base + run.len - n_pages, n_pages };
        if run.len > n_pages {
            // Head remainder back to the front (recency). Cannot be
            // adjacent to any other run (its parent was maximal).
            self.free.insert(0, FreeRun { base: run.base, len: run.len - n_pages });
        }
        for r in &mut self.refs[seg.pages()] {
            debug_assert_eq!(*r, 0, "alloc over a live page");
            *r = 1;
        }
        self.allocs += n_pages;
        self.seg_allocs += 1;
        Ok(seg)
    }

    /// Drop one reference from every page of `seg`; pages reaching 0 return
    /// to the free list (vLLM `BlockAllocator::free` semantics — the common
    /// path when the segment is not shared).
    pub fn free(&mut self, seg: KvSegment) {
        self.drop_refs(seg);
    }

    /// Alias of [`KvSegmentPool::free`] — drop one reference (explicit
    /// name for the refcounted/prefix-sharing path).
    pub fn unref(&mut self, seg: KvSegment) {
        self.drop_refs(seg);
    }

    /// Add one reference to every page of `seg` (prefix sharing). Panics on
    /// pages that are not currently live.
    pub fn ref_(&mut self, seg: KvSegment) {
        if seg.is_empty() {
            return;
        }
        assert!(seg.in_bounds(self.total), "segment out of bounds: {seg:?}");
        for r in &mut self.refs[seg.pages()] {
            assert!(*r >= 1, "ref on a free page");
            assert!(*r < u32::MAX, "refcount overflow");
            *r += 1;
        }
    }

    /// Shared implementation of `free`/`unref`.
    fn drop_refs(&mut self, seg: KvSegment) {
        if seg.is_empty() {
            return;
        }
        assert!(seg.in_bounds(self.total), "segment out of bounds: {seg:?}");
        let mut zeroed: Vec<usize> = Vec::new();
        for p in seg.pages() {
            assert!(self.refs[p] >= 1, "double free / unref of a free page {p}");
            self.refs[p] -= 1;
            if self.refs[p] == 0 {
                zeroed.push(p);
            }
        }
        if zeroed.is_empty() {
            return;
        }
        self.frees += zeroed.len();
        self.seg_frees += 1;
        // Build maximal runs from the ascending zeroed pages...
        let mut runs: Vec<FreeRun> = Vec::new();
        for p in zeroed {
            match runs.last_mut() {
                Some(r) if r.base + r.len == p => r.len += 1,
                _ => runs.push(FreeRun { base: p, len: 1 }),
            }
        }
        // ...and re-insert them front-first (recency), coalescing.
        for run in runs {
            self.insert_run(run);
        }
    }

    /// Insert a free run at the front of the list, coalescing with any
    /// adjacent run (at most two exist per the non-adjacency invariant;
    /// re-scans after each merge to handle chains).
    fn insert_run(&mut self, mut run: FreeRun) {
        let mut i = 0;
        while i < self.free.len() {
            let other = self.free[i];
            if other.base + other.len == run.base {
                run = FreeRun { base: other.base, len: run.len + other.len };
                self.free.remove(i);
                continue;
            }
            if run.base + run.len == other.base {
                run = FreeRun { base: run.base, len: run.len + other.len };
                self.free.remove(i);
                continue;
            }
            i += 1;
        }
        self.free.insert(0, run);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn pool(total: usize) -> KvSegmentPool {
        KvSegmentPool::new(BlockLen::B32, total)
    }

    #[test]
    fn alloc_free_roundtrip() {
        let mut p = pool(8);
        let s = p.alloc(3).unwrap();
        assert_eq!(s, KvSegment { base_page: 0, n_pages: 3 });
        assert_eq!(p.in_use(), 3);
        assert_eq!(p.free_count(), 5);
        assert_eq!(p.fragments(), 1, "split remainder stays one run");
        p.assert_conserved();
        p.free(s);
        assert_eq!(p.in_use(), 0);
        assert_eq!(p.free_count(), 8);
        assert_eq!(p.fragments(), 1, "coalesced back into one run");
        assert_eq!(p.max_free_run(), 8);
        p.assert_conserved();
        let st = p.stats();
        assert_eq!(st.allocs_pages, 3);
        assert_eq!(st.frees_pages, 3);
        assert_eq!(st.allocs_segments, 1);
        assert_eq!(st.frees_segments, 1);
    }

    #[test]
    fn most_recently_freed_run_reused_first() {
        let mut p = pool(8);
        let a = p.alloc(1).unwrap(); // [0,1)
        let b = p.alloc(1).unwrap(); // [1,2)
        let _c = p.alloc(1).unwrap(); // [2,3)
        p.free(a);
        p.free(b);
        // [0,1)+[1,2) coalesce into the front run [0,2); first-fit takes it
        // before the older [3,8) remainder.
        let nxt = p.alloc(1).unwrap();
        assert_eq!(nxt, KvSegment { base_page: 0, n_pages: 1 });
        assert_eq!(p.fragments(), 2);
        p.assert_conserved();
    }

    #[test]
    fn split_keeps_remainder_reusable() {
        let mut p = pool(8);
        let big = p.alloc(6).unwrap();
        assert_eq!(big, KvSegment { base_page: 0, n_pages: 6 });
        assert_eq!(p.free_count(), 2);
        let small = p.alloc(2).unwrap();
        assert_eq!(small, KvSegment { base_page: 6, n_pages: 2 });
        assert_eq!(p.in_use(), 8);
        p.assert_conserved();
    }

    #[test]
    fn coalesces_adjacent_frees() {
        let mut p = pool(8);
        let a = p.alloc(2).unwrap(); // [0,2)
        let b = p.alloc(2).unwrap(); // [2,4)
        let _c = p.alloc(2).unwrap(); // [4,6)
        p.free(a);
        p.free(b);
        // [0,4) is one run; [6,8) stays free from the split.
        assert_eq!(p.fragments(), 2);
        assert_eq!(p.max_free_run(), 4);
        p.assert_conserved();
    }

    #[test]
    fn refcount_sharing_keeps_pages_live() {
        let mut p = pool(4);
        let s = p.alloc(2).unwrap();
        p.ref_(s);
        assert_eq!(p.in_use(), 2);
        p.free(s); // one of two refs dropped — pages stay live
        assert_eq!(p.in_use(), 2);
        assert_eq!(p.free_count(), 2);
        let st = p.stats();
        assert_eq!(st.frees_pages, 0, "no release while refcount > 0");
        p.unref(s); // last ref — released
        assert_eq!(p.in_use(), 0);
        assert_eq!(p.free_count(), 4);
        assert_eq!(p.fragments(), 1);
        p.assert_conserved();
    }

    #[test]
    fn refcount_uneven_partial_release() {
        // Pages of one segment may carry different refcounts when only a
        // sub-range was shared: releasing the whole segment frees only the
        // pages whose count drops to 0.
        let mut p = pool(6);
        let full = p.alloc(3).unwrap(); // [0,3) refs 1
        let shared = KvSegment { base_page: 1, n_pages: 1 };
        p.ref_(shared); // page 1 refs 2
        p.free(full); // pages 0,2 freed; page 1 keeps ref 1
        assert_eq!(p.in_use(), 1);
        assert_eq!(p.free_count(), 5);
        assert_eq!(p.fragments(), 2, "pages 0 and 2 form two runs");
        p.assert_conserved();
        p.free(shared);
        assert_eq!(p.in_use(), 0);
        assert_eq!(p.free_count(), 6);
        assert_eq!(p.fragments(), 1, "all three runs coalesce");
        p.assert_conserved();
    }

    #[test]
    fn alloc_from_end_takes_the_pool_top() {
        let mut p = pool(8);
        let top = p.alloc_from_end(2).unwrap();
        assert_eq!(top, KvSegment { base_page: 6, n_pages: 2 }, "anchor at the very top");
        assert_eq!(p.free_count(), 6);
        assert_eq!(p.fragments(), 1);
        assert_eq!(p.max_free_run(), 6);
        p.assert_conserved();
        // Front-first allocations never extend past the anchor.
        let a = p.alloc(2).unwrap();
        assert_eq!(a, KvSegment { base_page: 0, n_pages: 2 });
        let b = p.alloc(2).unwrap();
        assert_eq!(b, KvSegment { base_page: 2, n_pages: 2 });
        let c = p.alloc(2).unwrap();
        assert_eq!(c, KvSegment { base_page: 4, n_pages: 2 });
        p.assert_conserved();
        // Freeing users reuses their runs (front), never the anchor.
        p.free(a);
        let d = p.alloc(1).unwrap();
        assert_eq!(d, KvSegment { base_page: 0, n_pages: 1 }, "reuses the freed front run");
        p.assert_conserved();
        // Release everything; the anchor release coalesces the whole pool.
        p.free(d);
        p.free(b);
        p.free(c);
        p.free(top);
        assert_eq!(p.free_count(), 8);
        assert_eq!(p.fragments(), 1, "anchor release coalesces the whole pool");
        p.assert_conserved();
    }

    #[test]
    fn alloc_from_end_prefers_the_last_run() {
        // With multiple free runs, from-end takes the back-most run that
        // fits (first-fit scanning backwards).
        let mut p = pool(16);
        let a = p.alloc(3).unwrap(); // [0,3)
        let _b = p.alloc(3).unwrap(); // [3,6)
        p.free(a); // runs [0,3), [6,16)
        let end = p.alloc_from_end(4).unwrap();
        assert_eq!(end, KvSegment { base_page: 12, n_pages: 4 }, "back-most fitting run");
        p.assert_conserved();
    }

    #[test]
    fn alloc_from_end_exact_run_and_errors() {
        let mut p = pool(8);
        assert_eq!(p.alloc_from_end(0), Err(MemoryError::InvalidLen { n: 0 }));
        assert_eq!(p.alloc_from_end(9), Err(MemoryError::OutOfPages { need: 9, free: 8 }));
        let top = p.alloc_from_end(4).unwrap();
        assert_eq!(top, KvSegment { base_page: 4, n_pages: 4 });
        let head = p.alloc(4).unwrap();
        assert_eq!(head, KvSegment { base_page: 0, n_pages: 4 });
        p.assert_conserved();
        assert_eq!(p.alloc_from_end(1), Err(MemoryError::OutOfPages { need: 1, free: 0 }));
        // Any fitting run is usable from the end (front-first-fit scanning
        // from the back of the list — not a strict "top-of-pool only").
        p.free(head);
        let again = p.alloc_from_end(4).unwrap();
        assert_eq!(again, KvSegment { base_page: 0, n_pages: 4 }, "only run left, taken whole");
        p.assert_conserved();
    }

    #[test]
    fn alloc_zero_and_fragmentation_errors() {
        let mut p = pool(4);
        assert_eq!(p.alloc(0), Err(MemoryError::InvalidLen { n: 0 }));
        assert_eq!(p.alloc(5), Err(MemoryError::OutOfPages { need: 5, free: 4 }));
        p.alloc(4).unwrap();
        assert_eq!(p.alloc(1), Err(MemoryError::OutOfPages { need: 1, free: 0 }));
        // Fragmentation: total free is enough but no single run is.
        let mut q = pool(6);
        let a = q.alloc(2).unwrap(); // [0,2)
        let _b = q.alloc(2).unwrap(); // [2,4) rem [4,6)
        q.free(a); // runs [0,2), [4,6)
        assert_eq!(q.free_count(), 4);
        assert_eq!(q.max_free_run(), 2);
        assert_eq!(q.alloc(3), Err(MemoryError::OutOfPages { need: 3, free: 4 }));
    }

    #[test]
    fn empty_segment_ops_are_noops() {
        let mut p = pool(4);
        let empty = KvSegment { base_page: 0, n_pages: 0 };
        p.free(empty);
        p.unref(empty);
        p.ref_(empty);
        assert_eq!(p.stats().allocs_segments, 0);
        assert_eq!(p.free_count(), 4);
    }

    #[test]
    fn deterministic_given_call_sequence() {
        // Identical call sequences on fresh pools yield identical segments
        // and stats — pins the determinism contract.
        let ops = |p: &mut KvSegmentPool| -> Vec<KvSegment> {
            let a = p.alloc(3).unwrap();
            let b = p.alloc(2).unwrap();
            p.free(a);
            let c = p.alloc(1).unwrap();
            let d = p.alloc(4).unwrap();
            p.free(b);
            p.free(d);
            let e = p.alloc(5).unwrap();
            p.free(c);
            p.free(e);
            vec![a, b, c, d, e]
        };
        let mut p1 = pool(16);
        let mut p2 = pool(16);
        assert_eq!(ops(&mut p1), ops(&mut p2));
        assert_eq!(p1.stats(), p2.stats());
        assert_eq!(p1.free_count(), 16);
        assert_eq!(p1.in_use(), 0);
        assert_eq!(p1.fragments(), 1);
        p1.assert_conserved();
    }

    #[test]
    fn churn_conserves_under_deterministic_lcg() {
        // Fixed-seed LCG churn: interleaved alloc/free with conservation
        // asserted after every step (no RNG — fully deterministic).
        let mut p = pool(64);
        let mut live: Vec<KvSegment> = Vec::new();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        for _ in 0..2_000 {
            let r = next() % 100;
            if r < 45 || live.is_empty() {
                let n = 1 + (next() % 7) as usize;
                if let Ok(seg) = p.alloc(n) {
                    live.push(seg);
                }
            } else {
                let idx = (next() as usize) % live.len();
                let seg = live.swap_remove(idx);
                p.free(seg);
            }
            p.assert_conserved();
        }
        for seg in live {
            p.free(seg);
        }
        assert_eq!(p.in_use(), 0);
        assert_eq!(p.free_count(), 64);
        assert_eq!(p.fragments(), 1);
        p.assert_conserved();
    }

    #[test]
    fn empty_pool() {
        let mut p = KvSegmentPool::new(BlockLen::B16, 0);
        assert_eq!(p.free_count(), 0);
        assert_eq!(p.alloc(1), Err(MemoryError::OutOfPages { need: 1, free: 0 }));
        assert_eq!(p.fragments(), 0);
        assert_eq!(p.max_free_run(), 0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn free_out_of_bounds_panics() {
        let mut p = pool(4);
        p.free(KvSegment { base_page: 3, n_pages: 2 });
    }

    #[test]
    #[should_panic(expected = "double free")]
    fn double_free_panics() {
        let mut p = pool(4);
        let s = p.alloc(2).unwrap();
        p.free(s);
        p.free(s);
    }

    #[test]
    #[should_panic(expected = "ref on a free page")]
    fn ref_freed_page_panics() {
        let mut p = pool(4);
        let s = p.alloc(1).unwrap();
        p.free(s);
        p.ref_(s);
    }
}
