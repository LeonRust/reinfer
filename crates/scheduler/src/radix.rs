//! Token-prefix KV cache front-end (spec 016 P3-01 v1 — tasks T0a/T0b).
//!
//! `TokenRadixCache` is the pure-CPU decision layer of the prefix cache: it
//! records which prompt prefixes are cached, at which physical pages, in
//! which recency order. It holds no GPU state and no pool state — the caller
//! (scheduler-executor, T1b/T2) owns the KV pool and executes the refcount
//! transitions this cache reports:
//!
//! - `insert` returns the evicted entries as a callback list — the caller
//!   must `pool.unref` each (pages return to the pool);
//! - on `insert` errors the cache has not been mutated, so a caller that
//!   already `pool.ref_`-ed the candidate run must give that reference back;
//! - `lookup` is a pure read (`&self`) — spec D3a: the hit path must not
//!   mutate scheduler state.
//!
//! ## Entry model (spec D1/D4)
//!
//! An entry is `(aligned token key, base_page, L pages, recency stamp)`:
//!
//! - the **key** is the request prompt (token ids); only its **page-aligned
//!   prefix** `key[..L*block]` with `L = floor(key_len / block)` is stored in
//!   the trie — tokens beyond the aligned prefix are not cached (v1 has no
//!   partial pages, spec Non-Goals);
//! - `base_page` + `L` name the cached prefix run in the KV pool
//!   (`prefix_run = { base, L }`, D1);
//! - `lookup` returns the **longest page-aligned hit**: an entry whose
//!   aligned prefix matches the prompt, `Hit { base_page, pages, key_len }`
//!   with `pages = key_len / block`. A hit may come from an entry whose own
//!   key is *longer* than the prompt — the entry's first `pages` pages are
//!   still the exact KV of the shorter prompt's prefix (bit-identical by
//!   construction), so a 96-token prompt can hit a 128-token entry with
//!   `key_len = 96`, `pages = 3`.
//!
//! ## Recency and eviction
//!
//! Recency is a monotonic stamp assigned at insert; [`TokenRadixCache::touch`]
//! refreshes it for a re-used entry (the caller signals re-use at refill,
//! spec D2). Eviction takes the **oldest** entries until the budget
//! (`max_pages` = `REINFER_PREFIX_CACHE_PAGES`; default = pool total × 10%,
//! ≥ 1 — wired at T2) fits the new entry. `lookup` never refreshes recency:
//! with a pure read there is no hit-path mutation (D3a).
//!
//! ## Determinism contract (spec T0)
//!
//! Single-threaded (D1), no RNG, no hash-map iteration order (children are
//! `BTreeMap`, stamps are strictly increasing): the same (insert, lookup,
//! touch, evict) sequence on fresh caches yields bit-identical entry tables,
//! eviction traces and pool return sequences — pinned by the
//! SchedDeterminism-style double-run test.
//!
//! ## Refill sequence (spec D2) — caller contract
//!
//! ```text
//! refill_hook(seg, prompt):                      // at the release guard
//!   if L = floor(prompt.len / block) < MIN_BLOCKS: pool.free(seg)
//!   elif lookup(prompt).key_len == L*block:      // same aligned prefix cached
//!       cache.touch(prompt); pool.free(seg)
//!   else:
//!       pool.ref_(seg[..L]); pool.free(seg)      // prefix ref 2 -> 1 survives
//!       cache.insert(prompt, seg.base, L)        // evictions -> pool.unref
//! ```
//!
//! The `ref_` before `free` keeps the aligned prefix pages alive through the
//! release (never transiently zero); the entry then owns their single
//! reference. On `insert` error the caller unrefs the run it just ref-ed.

use std::collections::BTreeMap;

/// Default tokens per page — the page-aligned match granularity.
pub const DEFAULT_BLOCK: usize = 32;

/// Minimum prefix pages for admission (spec D2: prefixes shorter than two
/// blocks are not cached; the refill sequence no-ops straight to
/// `pool.free(seg)`).
pub const MIN_BLOCKS: usize = 2;

/// A page-aligned cache hit (spec T0: longest page-aligned match).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixHit {
    /// First physical page of the matched prefix run (in the KV pool).
    pub base_page: u32,
    /// Pages covered by the hit: `L = floor(key_len / block)`.
    pub pages: u32,
    /// Aligned prefix tokens matched: `pages * block` (≤ the prompt length).
    pub key_len: usize,
}

/// An entry evicted by `insert`/`clear` — the caller must `pool.unref` it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evicted {
    /// First physical page of the evicted run.
    pub base_page: u32,
    /// Pages of the evicted run.
    pub pages: u32,
}

/// Deterministic snapshot of one cached entry (entry-table observability and
/// the SchedDeterminism-style identity check).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryView {
    /// Aligned prefix tokens that form the entry key.
    pub key: Vec<u32>,
    /// First physical page of the cached run.
    pub base_page: u32,
    /// Cached pages (`key.len() / block`).
    pub pages: u32,
}

/// `insert` validation errors. All errors are raised before any mutation, so
/// a rejected insert leaves the cache untouched (and the caller returns the
/// `ref_` it took on the candidate run).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadixError {
    /// The key is empty — nothing to cache.
    EmptyKey,
    /// `pages == 0` — an entry must own at least one page.
    ZeroPages,
    /// `pages * block > key.len()` — the entry would cover tokens the key
    /// does not provide.
    PagesBeyondKey {
        /// Provided key length in tokens.
        key_len: usize,
        /// Requested pages.
        pages: u32,
    },
    /// `max_pages == 0` — the cache admits nothing.
    CacheDisabled,
    /// A single entry cannot exceed the budget (no other entry can be
    /// evicted to make room for it).
    EntryExceedsBudget {
        /// Requested pages.
        pages: u32,
        /// The page budget.
        max_pages: u64,
    },
}

/// One trie node: token-keyed children plus an optional entry. Entries are
/// attached only at page-aligned depths (`depth == pages * block`).
#[derive(Debug, Default)]
struct TrieNode {
    /// Child nodes keyed by the next token (ordered — determinism).
    children: BTreeMap<u32, TrieNode>,
    /// Entry attached at this node's depth, if any.
    entry: Option<Entry>,
    /// Number of entries in this subtree, including the node's own.
    subtree_entries: usize,
}

/// A cached run with its recency stamp.
#[derive(Debug)]
struct Entry {
    /// First physical page of the run.
    base_page: u32,
    /// Pages of the run.
    pages: u32,
    /// Recency stamp: strictly increasing over insert/touch calls.
    stamp: u64,
}

/// Token-prefix KV cache front-end (see the module docs for the contracts).
#[derive(Debug)]
pub struct TokenRadixCache {
    /// Tokens per page (match-alignment granularity).
    block: usize,
    /// Page budget (`REINFER_PREFIX_CACHE_PAGES`; default pool × 10%, ≥ 1).
    max_pages: u64,
    /// Pages currently owned by cached entries.
    cached_pages: u64,
    /// Number of cached entries.
    n_entries: usize,
    /// Monotonic recency counter (assigned to the next insert/touch).
    stamp: u64,
    /// Trie root (depth 0, never holds an entry).
    root: TrieNode,
}

impl TokenRadixCache {
    /// New cache with the default block size and the given page budget.
    pub fn new(max_pages: u64) -> Self {
        Self::new_with_block(max_pages, DEFAULT_BLOCK)
    }

    /// New cache with a custom block size (tests use small blocks; the
    /// engine always uses the default). `block` must be ≥ 1.
    pub fn new_with_block(max_pages: u64, block: usize) -> Self {
        assert!(block >= 1, "block size must be positive, got {block}");
        Self {
            block,
            max_pages,
            cached_pages: 0,
            n_entries: 0,
            stamp: 0,
            root: TrieNode::default(),
        }
    }

    /// Tokens per page.
    pub fn block(&self) -> usize {
        self.block
    }

    /// Page budget.
    pub fn max_pages(&self) -> u64 {
        self.max_pages
    }

    /// Pages currently owned by cached entries.
    pub fn pages(&self) -> u64 {
        self.cached_pages
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.n_entries
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.n_entries == 0
    }

    /// Longest page-aligned hit for `ids` (pure read — no state mutation).
    ///
    /// Walks the trie along the prompt; the hit is the deepest aligned depth
    /// whose subtree still contains an entry. The returned `key_len` is the
    /// aligned token count (`pages * block`) — never a partial page. `None`
    /// for an empty prompt or when no aligned prefix is cached.
    pub fn lookup(&self, ids: &[u32]) -> Option<PrefixHit> {
        let (d, path) = self.deepest_hit_path(ids)?;
        let entry = self.entry_at(&path)?;
        Some(PrefixHit { base_page: entry.base_page, pages: (d / self.block) as u32, key_len: d })
    }

    /// Refresh the recency of the entry that would serve `key` (the caller
    /// signals a re-use at refill, spec D2). Returns false when no entry
    /// covers the key's aligned prefix — a pure no-op then.
    pub fn touch(&mut self, key: &[u32]) -> bool {
        let Some((_, path)) = self.deepest_hit_path(key) else { return false };
        let mut node = &mut self.root;
        for &tok in &path {
            let Some(next) = node.children.get_mut(&tok) else { return false };
            node = next;
        }
        let Some(entry) = node.entry.as_mut() else { return false };
        self.stamp += 1;
        entry.stamp = self.stamp;
        true
    }

    /// Admit `(key, base_page, pages)` as a cached entry.
    ///
    /// The aligned prefix `key[..pages*block]` is the entry key; a longer
    /// `key` is truncated (its tail is not cached — no partial pages). If the
    /// aligned key is already cached the insert is a no-op that only refreshes
    /// recency (single entry per aligned prefix, spec T2).
    ///
    /// When the budget would be exceeded, the oldest entries are evicted
    /// first (LRU) and returned — the caller must `pool.unref` each. Every
    /// error leaves the cache untouched.
    ///
    /// Caller contract: the run `(base_page, pages)` must already carry the
    /// cache's reference (spec D2: `pool.ref_` before `pool.free`).
    pub fn insert(
        &mut self,
        key: &[u32],
        base_page: u32,
        pages: u32,
    ) -> Result<Vec<Evicted>, RadixError> {
        if key.is_empty() {
            return Err(RadixError::EmptyKey);
        }
        if pages == 0 {
            return Err(RadixError::ZeroPages);
        }
        let aligned = (pages as usize) * self.block;
        if aligned > key.len() {
            return Err(RadixError::PagesBeyondKey { key_len: key.len(), pages });
        }
        if self.max_pages == 0 {
            return Err(RadixError::CacheDisabled);
        }
        if pages as u64 > self.max_pages {
            return Err(RadixError::EntryExceedsBudget { pages, max_pages: self.max_pages });
        }
        let key = &key[..aligned];
        // Create the path (shared prefixes are single nodes — structural
        // sharing); an entry already at the end means the aligned key exists.
        {
            let mut node = &mut self.root;
            for &tok in key {
                node = node.children.entry(tok).or_default();
            }
            if let Some(entry) = node.entry.as_mut() {
                self.stamp += 1;
                entry.stamp = self.stamp;
                return Ok(Vec::new());
            }
        }
        // Evict the oldest entries until the budget fits (the validation
        // above guarantees `pages <= max_pages`, so the loop always ends).
        let mut evicted: Vec<Evicted> = Vec::new();
        let mut surplus =
            self.cached_pages.saturating_add(pages as u64).saturating_sub(self.max_pages);
        if surplus > 0 {
            let oldest = self.eviction_candidates();
            for (path, _, candidate) in oldest {
                if surplus == 0 {
                    break;
                }
                let removed = self.remove_entry_at(&path).expect("candidate entry present");
                debug_assert_eq!(
                    (removed.base_page, removed.pages),
                    (candidate.base_page, candidate.pages),
                    "eviction candidate and removal must agree"
                );
                surplus = surplus.saturating_sub(removed.pages as u64);
                evicted.push(removed);
            }
            debug_assert_eq!(surplus, 0, "budget must fit after evicting the oldest");
        }
        // Admit the entry and account its subtree membership.
        {
            let mut node = &mut self.root;
            node.subtree_entries += 1;
            for &tok in key {
                node = node.children.get_mut(&tok).expect("path created above");
                node.subtree_entries += 1;
            }
            self.stamp += 1;
            node.entry = Some(Entry { base_page, pages, stamp: self.stamp });
        }
        self.cached_pages += pages as u64;
        self.n_entries += 1;
        Ok(evicted)
    }

    /// Evict every entry (oldest first is not required — `clear` is total).
    /// The returned list is the caller's `pool.unref` callback list; the
    /// cache is empty afterwards.
    pub fn clear(&mut self) -> Vec<Evicted> {
        let all: Vec<Evicted> = self
            .collect_entries()
            .into_iter()
            .map(|(_, e)| Evicted { base_page: e.base_page, pages: e.pages })
            .collect();
        self.root = TrieNode::default();
        self.cached_pages = 0;
        self.n_entries = 0;
        all
    }

    /// Deterministic snapshot of the entry table, sorted by key (ascending
    /// token order). Two runs of the same operation sequence produce
    /// identical tables (SchedDeterminism-style identity check).
    pub fn entry_table(&self) -> Vec<EntryView> {
        self.collect_entries()
            .into_iter()
            .map(|(key, e)| EntryView { key, base_page: e.base_page, pages: e.pages })
            .collect()
    }

    /// Internal invariants: entry depth == pages × block, pages sum ==
    /// `pages()`, entry count == `len()`, stamps unique, budget respected.
    pub fn assert_consistent(&self) {
        let all = self.collect_entries();
        let mut pages_sum: u64 = 0;
        let mut stamps: std::collections::BTreeSet<u64> = std::collections::BTreeSet::new();
        for (path, e) in &all {
            assert_eq!(
                path.len(),
                e.pages as usize * self.block,
                "entry depth must equal pages * block"
            );
            assert!(e.pages >= 1, "entries own at least one page");
            assert!(e.pages as u64 <= self.max_pages, "entry exceeds the budget");
            pages_sum += e.pages as u64;
            assert!(stamps.insert(e.stamp), "recency stamps must be unique");
        }
        assert_eq!(pages_sum, self.cached_pages, "cached pages == sum of entry pages");
        assert_eq!(all.len(), self.n_entries, "entry count");
        assert_eq!(self.root.subtree_entries, self.n_entries, "root subtree count == entry count");
        assert!(self.cached_pages <= self.max_pages, "cached pages within the budget");
    }

    /// `(aligned matched depth, path to the deepest matching entry)`: walk 1
    /// finds the deepest aligned depth whose subtree contains an entry; walk
    /// 2 picks the deepest entry in that subtree (ties broken by the
    /// lexicographically smallest key — deterministic).
    fn deepest_hit_path(&self, ids: &[u32]) -> Option<(usize, Vec<u32>)> {
        // Walk 1: descend along the prompt, remembering the deepest aligned
        // depth that still covers an entry.
        let mut node = &self.root;
        let mut best_depth: Option<usize> = None;
        for (i, &tok) in ids.iter().enumerate() {
            let depth = i + 1;
            let Some(child) = node.children.get(&tok) else { break };
            node = child;
            if depth % self.block == 0 && node.subtree_entries > 0 {
                best_depth = Some(depth);
            }
        }
        let d = best_depth?;
        // Walk 2: reach the subtree root at depth `d` (it exists — walk 1
        // descended through it) and find its deepest entry.
        let mut node = &self.root;
        for &tok in &ids[..d] {
            node = node.children.get(&tok).expect("walk-1 path exists");
        }
        let mut best: Option<(usize, Vec<u32>)> = None;
        let mut work: Vec<(&TrieNode, Vec<u32>)> = vec![(node, ids[..d].to_vec())];
        while let Some((n, path)) = work.pop() {
            if n.entry.is_some() {
                let better = match &best {
                    Some((bd, bp)) => path.len() > *bd || (path.len() == *bd && path < *bp),
                    None => true,
                };
                if better {
                    best = Some((path.len(), path.clone()));
                }
            }
            for (&tok, child) in &n.children {
                let mut p = path.clone();
                p.push(tok);
                work.push((child, p));
            }
        }
        best.map(|(_, path)| (d, path))
    }

    /// The node at the end of `path` (immutable).
    fn entry_at(&self, path: &[u32]) -> Option<&Entry> {
        let mut node = &self.root;
        for &tok in path {
            node = node.children.get(&tok)?;
        }
        node.entry.as_ref()
    }

    /// All entries as `(key path, entry)` sorted by key (ascending) —
    /// deterministic iteration for eviction and the entry table.
    fn collect_entries(&self) -> Vec<(Vec<u32>, &Entry)> {
        let mut out: Vec<(Vec<u32>, &Entry)> = Vec::new();
        let mut work: Vec<(&TrieNode, Vec<u32>)> = vec![(&self.root, Vec::new())];
        while let Some((node, path)) = work.pop() {
            for (&tok, child) in &node.children {
                let mut p = path.clone();
                p.push(tok);
                work.push((child, p));
            }
            if let Some(e) = &node.entry {
                out.push((path, e));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// All entries ordered oldest-first (unique stamps — a total order).
    fn eviction_candidates(&self) -> Vec<(Vec<u32>, u64, Evicted)> {
        let mut out: Vec<(Vec<u32>, u64, Evicted)> = self
            .collect_entries()
            .into_iter()
            .map(|(path, e)| (path, e.stamp, Evicted { base_page: e.base_page, pages: e.pages }))
            .collect();
        out.sort_by_key(|&(_, stamp, _)| stamp);
        out
    }

    /// Remove the entry at `path` and prune the now-empty tail of its chain
    /// (nodes that carry no other entry and no branch). Decrements the
    /// subtree accounting along the surviving path.
    fn remove_entry_at(&mut self, path: &[u32]) -> Option<Evicted> {
        let k = path.len();
        // Immutable walk: node shapes along the path + the entry payload.
        struct Shape {
            has_entry: bool,
            children: usize,
        }
        let mut shapes: Vec<Shape> = Vec::with_capacity(k);
        let mut node = &self.root;
        for &tok in path {
            let child = node.children.get(&tok)?;
            shapes.push(Shape { has_entry: node.entry.is_some(), children: node.children.len() });
            node = child;
        }
        let entry = node.entry.as_ref()?;
        let evicted = Evicted { base_page: entry.base_page, pages: entry.pages };
        // Trim point: the first path node whose whole subtree is the entry
        // being removed (no other entry, single-child chain to the end).
        let mut removable_next = node.children.is_empty();
        let mut trim_from: Option<usize> = if removable_next { Some(k) } else { None };
        for j in (1..k).rev() {
            let s = &shapes[j];
            let removable = !s.has_entry && s.children == 1 && removable_next;
            if removable {
                trim_from = Some(j);
                removable_next = true;
            } else {
                break;
            }
        }
        match trim_from {
            Some(j) => {
                // The subtree at path[j] is the entry chain alone — drop it
                // wholesale; decrement the surviving path path[..j-1].
                let mut node = &mut self.root;
                node.subtree_entries -= 1;
                for &tok in &path[..j - 1] {
                    node = node.children.get_mut(&tok).expect("path node");
                    node.subtree_entries -= 1;
                }
                node.children.remove(&path[j - 1]).expect("subtree to drop");
            }
            None => {
                // The entry node has other content (children or ancestors
                // with entries): clear the entry in place, decrement the
                // whole path.
                let mut node = &mut self.root;
                node.subtree_entries -= 1;
                for &tok in path {
                    node = node.children.get_mut(&tok).expect("path node");
                    node.subtree_entries -= 1;
                }
                let removed = node.entry.take().expect("entry at path end");
                debug_assert_eq!(
                    (removed.base_page, removed.pages),
                    (evicted.base_page, evicted.pages),
                    "removal must match the walk-1 payload"
                );
            }
        }
        self.cached_pages -= evicted.pages as u64;
        self.n_entries -= 1;
        Some(evicted)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// Distinct prompt keys for the scripts below.
    fn key(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    /// Spec 016 D2 refill sequence over a mock pool: returns the evictions
    /// the caller had to unref (the trace the determinism test compares).
    fn refill(
        pool: &mut MockPool,
        cache: &mut TokenRadixCache,
        prompt: &[u32],
        seg: (usize, usize),
    ) -> Vec<Evicted> {
        let block = cache.block();
        let pages = prompt.len() / block; // L = floor(prompt_len / block)
        if pages < MIN_BLOCKS {
            pool.free(seg.0, seg.1); // shorter than two blocks: not cached
            return Vec::new();
        }
        let aligned = pages * block;
        if let Some(hit) = cache.lookup(prompt)
            && hit.key_len == aligned
        {
            // Same aligned prefix already cached: keep the single entry,
            // refresh its recency, release the segment untouched.
            assert!(cache.touch(prompt), "full hit must touch");
            pool.free(seg.0, seg.1);
            return Vec::new();
        }
        // Keep the aligned prefix alive through the release (ref 2 -> 1),
        // return the rest, then admit the entry.
        pool.ref_(seg.0, pages);
        pool.free(seg.0, seg.1);
        match cache.insert(prompt, seg.0 as u32, pages as u32) {
            Ok(evicted) => {
                for e in &evicted {
                    pool.unref(e.base_page as usize, e.pages as usize);
                }
                evicted
            }
            Err(_) => {
                // The cache refused the entry (budget): give the prefix back.
                pool.unref(seg.0, pages);
                Vec::new()
            }
        }
    }

    // ------------------------------------------------------------------
    // T0a — pure cache unit tests
    // ------------------------------------------------------------------

    #[test]
    fn empty_cache_lookup_and_touch_noop() {
        let cache = TokenRadixCache::new(8);
        assert_eq!(cache.lookup(&[]), None, "empty prompt");
        assert_eq!(cache.lookup(&[1, 2, 3, 4]), None, "nothing cached");
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.pages(), 0);
        assert!(cache.is_empty());
        let mut cache = TokenRadixCache::new(8);
        assert!(!cache.touch(&[]), "empty key never touches");
        assert!(!cache.touch(&[1, 2, 3, 4]), "missing key never touches");
        assert_eq!(cache.lookup(&[]), None);
    }

    #[test]
    fn insert_rejects_invalid_inputs() {
        let mut cache = TokenRadixCache::new_with_block(8, 2);
        assert_eq!(cache.insert(&[], 0, 1), Err(RadixError::EmptyKey));
        assert_eq!(cache.insert(&[1, 2, 3], 0, 0), Err(RadixError::ZeroPages));
        // pages * block (4) exceeds the 3-token key.
        assert_eq!(
            cache.insert(&[1, 2, 3], 0, 2),
            Err(RadixError::PagesBeyondKey { key_len: 3, pages: 2 })
        );
        // A disabled cache admits nothing.
        let mut disabled = TokenRadixCache::new_with_block(0, 2);
        assert_eq!(disabled.insert(&[1, 2, 3, 4], 0, 2), Err(RadixError::CacheDisabled));
        // A single entry cannot exceed the budget (8 aligned tokens fit the
        // 8-token key, so this is the budget check, not a key-length error).
        let mut tight = TokenRadixCache::new_with_block(3, 2);
        assert_eq!(
            tight.insert(&[1, 2, 3, 4, 5, 6, 7, 8], 0, 4),
            Err(RadixError::EntryExceedsBudget { pages: 4, max_pages: 3 })
        );
        // Rejected inserts never mutate.
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.pages(), 0);
        cache.assert_consistent();
    }

    #[test]
    fn serial_chain_inserts_and_longest_hit() {
        let mut cache = TokenRadixCache::new_with_block(16, 2);
        assert_eq!(cache.insert(&[0, 1, 2, 3], 10, 2), Ok(vec![]));
        assert_eq!(cache.insert(&[0, 1, 2, 3, 4, 5], 20, 3), Ok(vec![]));
        cache.assert_consistent();
        // The longest hit wins for the longer prompt.
        assert_eq!(
            cache.lookup(&[0, 1, 2, 3, 4, 5]),
            Some(PrefixHit { base_page: 20, pages: 3, key_len: 6 })
        );
        // The shorter prompt hits the *deeper* entry's run: its first 2
        // pages are the exact KV of the 4-token prefix (bit-identical).
        assert_eq!(
            cache.lookup(&[0, 1, 2, 3]),
            Some(PrefixHit { base_page: 20, pages: 2, key_len: 4 })
        );
        assert_eq!(cache.lookup(&[0, 1]), Some(PrefixHit { base_page: 20, pages: 1, key_len: 2 }));
        assert_eq!(cache.pages(), 5);
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());
    }

    #[test]
    fn branching_keys_hit_their_own_entries() {
        let mut cache = TokenRadixCache::new_with_block(16, 2);
        cache.insert(&[0, 1, 2, 3], 10, 2).unwrap();
        cache.insert(&[0, 1, 9, 9], 20, 2).unwrap();
        assert_eq!(
            cache.lookup(&[0, 1, 2, 3]),
            Some(PrefixHit { base_page: 10, pages: 2, key_len: 4 })
        );
        assert_eq!(
            cache.lookup(&[0, 1, 9, 9]),
            Some(PrefixHit { base_page: 20, pages: 2, key_len: 4 })
        );
        // The shared 1-page prefix is servable; both branches cover it —
        // the deepest (tie) is resolved deterministically (smaller key).
        assert_eq!(cache.lookup(&[0, 1]), Some(PrefixHit { base_page: 10, pages: 1, key_len: 2 }));
        cache.assert_consistent();
    }

    #[test]
    fn no_partial_page_hits() {
        let mut cache = TokenRadixCache::new_with_block(16, 2);
        cache.insert(&[0, 1, 2, 3], 10, 2).unwrap();
        cache.insert(&[0, 1, 2, 4], 20, 2).unwrap();
        // Divergence inside the second page: each prompt hits its own entry.
        assert_eq!(
            cache.lookup(&[0, 1, 2, 3]),
            Some(PrefixHit { base_page: 10, pages: 2, key_len: 4 })
        );
        assert_eq!(
            cache.lookup(&[0, 1, 2, 4]),
            Some(PrefixHit { base_page: 20, pages: 2, key_len: 4 })
        );
        // A 3-token prompt can only use the aligned 2-token prefix — the
        // partial third token is never served (v1 Non-Goal: no split pages).
        assert_eq!(
            cache.lookup(&[0, 1, 2]),
            Some(PrefixHit { base_page: 10, pages: 1, key_len: 2 })
        );
        cache.assert_consistent();
    }

    #[test]
    fn duplicate_insert_is_noop_and_touches() {
        let mut cache = TokenRadixCache::new_with_block(4, 2);
        cache.insert(&[0, 1, 2, 3], 10, 2).unwrap();
        // Same aligned key, different run: no second entry, recency refreshed.
        assert_eq!(cache.insert(&[0, 1, 2, 3], 99, 2), Ok(vec![]));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.pages(), 2);
        assert_eq!(cache.entry_table()[0].base_page, 10, "first entry kept");
        // The refresh is observable in the eviction order: b inserted after
        // a, a re-used afterwards, then c arrives under budget pressure.
        let mut cache = TokenRadixCache::new_with_block(4, 2);
        cache.insert(&[0, 1, 2, 3], 10, 2).unwrap();
        cache.insert(&[5, 6, 7, 8], 20, 2).unwrap();
        assert!(cache.touch(&[0, 1, 2, 3]), "re-use of a");
        // Without the touch the oldest would be a; with it, b goes first.
        assert_eq!(
            cache.insert(&[9, 9, 9, 9], 30, 2),
            Ok(vec![Evicted { base_page: 20, pages: 2 }])
        );
        assert_eq!(cache.pages(), 4);
        cache.assert_consistent();
    }

    #[test]
    fn lru_touch_refreshes_recency() {
        // Three entries under a 4-page budget: the least recently used one
        // (b — never touched) is evicted when c arrives.
        let mut cache = TokenRadixCache::new_with_block(4, 2);
        cache.insert(&[0, 1, 2, 3], 10, 2).unwrap();
        cache.insert(&[5, 6, 7, 8], 20, 2).unwrap();
        assert!(cache.touch(&[0, 1, 2, 3]));
        assert_eq!(
            cache.insert(&[9, 9, 9, 9], 30, 2),
            Ok(vec![Evicted { base_page: 20, pages: 2 }])
        );
        assert_eq!(cache.lookup(&[0, 1, 2, 3]).map(|h| h.base_page), Some(10));
        assert_eq!(cache.lookup(&[9, 9, 9, 9]).map(|h| h.base_page), Some(30));
        assert_eq!(cache.lookup(&[5, 6, 7, 8]), None, "evicted entry gone");
        cache.assert_consistent();
    }

    #[test]
    fn budget_exact_eviction() {
        // Budget 3: a(2) + b(2) overshoots by exactly 1 -> a evicted first.
        let mut cache = TokenRadixCache::new_with_block(3, 2);
        assert_eq!(cache.insert(&[0, 1, 2, 3], 10, 2), Ok(vec![]));
        assert_eq!(
            cache.insert(&[5, 6, 7, 8], 20, 2),
            Ok(vec![Evicted { base_page: 10, pages: 2 }])
        );
        assert_eq!(cache.pages(), 2);
        // A 1-page entry fits without eviction: budget exactly full.
        assert_eq!(cache.insert(&[9, 9], 30, 1), Ok(vec![]));
        assert_eq!(cache.pages(), 3);
        // The next insertion evicts the oldest (b) to make room.
        assert_eq!(cache.insert(&[1, 1], 40, 1), Ok(vec![Evicted { base_page: 20, pages: 2 }]));
        assert_eq!(cache.pages(), 2);
        cache.assert_consistent();
    }

    #[test]
    fn default_block_32_alignment() {
        let mut cache = TokenRadixCache::new(8);
        let key100: Vec<u32> = key(0..100);
        // 100 tokens -> 3 pages (96 aligned); the 4-token tail is not cached.
        assert_eq!(cache.insert(&key100, 10, 3), Ok(vec![]));
        assert_eq!(cache.lookup(&key100), Some(PrefixHit { base_page: 10, pages: 3, key_len: 96 }));
        // A 96-token prompt hits the same entry (same aligned prefix).
        assert_eq!(
            cache.lookup(&key100[..96]),
            Some(PrefixHit { base_page: 10, pages: 3, key_len: 96 })
        );
        // Re-inserting the aligned prefix is a duplicate (key truncated).
        assert_eq!(cache.insert(&key100[..96], 20, 3), Ok(vec![]));
        assert_eq!(cache.len(), 1);
        // Requesting a fourth page overruns the 100-token key.
        assert_eq!(
            cache.insert(&key100, 11, 4),
            Err(RadixError::PagesBeyondKey { key_len: 100, pages: 4 })
        );
        cache.assert_consistent();
    }

    #[test]
    fn evicted_key_misses_when_unrelated() {
        let mut cache = TokenRadixCache::new_with_block(2, 2);
        cache.insert(&[0, 1, 2, 3], 10, 2).unwrap();
        // The unrelated key has no aligned prefix in common: a full eviction
        // of a leaves nothing to serve its tokens.
        assert_eq!(
            cache.insert(&[7, 8, 9, 10], 20, 2),
            Ok(vec![Evicted { base_page: 10, pages: 2 }])
        );
        assert_eq!(cache.lookup(&[0, 1, 2, 3]), None);
        assert_eq!(
            cache.lookup(&[7, 8, 9, 10]),
            Some(PrefixHit { base_page: 20, pages: 2, key_len: 4 })
        );
        cache.assert_consistent();
    }

    #[test]
    fn nested_entry_eviction_keeps_the_parent_prefix() {
        // Budget 4: b (3 pages) extends a (2 pages); a is the oldest and is
        // evicted, but its tokens remain servable through b's run.
        let mut cache = TokenRadixCache::new_with_block(4, 2);
        cache.insert(&[0, 1, 2, 3], 10, 2).unwrap();
        assert_eq!(
            cache.insert(&[0, 1, 2, 3, 4, 5], 20, 3),
            Ok(vec![Evicted { base_page: 10, pages: 2 }])
        );
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.pages(), 3);
        assert_eq!(
            cache.lookup(&[0, 1, 2, 3]),
            Some(PrefixHit { base_page: 20, pages: 2, key_len: 4 })
        );
        assert_eq!(cache.clear(), vec![Evicted { base_page: 20, pages: 3 }]);
        assert!(cache.is_empty());
        assert_eq!(cache.lookup(&[0, 1, 2, 3, 4, 5]), None);
        cache.assert_consistent();
    }

    #[test]
    fn entry_table_sorted_by_key() {
        let mut cache = TokenRadixCache::new_with_block(16, 2);
        cache.insert(&[9, 9, 9, 9], 30, 2).unwrap();
        cache.insert(&[0, 1, 2, 3], 10, 2).unwrap();
        cache.insert(&[0, 1, 2, 3, 4, 5], 20, 3).unwrap();
        let table = cache.entry_table();
        let keys: Vec<&Vec<u32>> = table.iter().map(|v| &v.key).collect();
        assert_eq!(keys, vec![&vec![0, 1, 2, 3], &vec![0, 1, 2, 3, 4, 5], &vec![9, 9, 9, 9]]);
        assert_eq!(table[1], EntryView { key: vec![0, 1, 2, 3, 4, 5], base_page: 20, pages: 3 });
    }

    // ------------------------------------------------------------------
    // T0b — pool reflection tests (mock pool with ref_/free/unref)
    // ------------------------------------------------------------------

    /// Test-only mirror of `KvSegmentPool` (crates/memory/src/segment.rs):
    /// per-page refcounts (`ref_` +1, `free`/`unref` -1) with coalesced free
    /// runs, most recently released first.
    #[derive(Debug)]
    struct MockPool {
        total: usize,
        refs: Vec<u32>,
        free: Vec<(usize, usize)>,
    }

    impl MockPool {
        fn new(total: usize) -> Self {
            let free = if total == 0 { Vec::new() } else { vec![(0, total)] };
            Self { total, refs: vec![0; total], free }
        }

        /// First-fit allocation of `n` pages; every page gets refcount 1.
        fn alloc(&mut self, n: usize) -> Option<(usize, usize)> {
            if n == 0 {
                return None;
            }
            let idx = self.free.iter().position(|&(_, len)| len >= n)?;
            let (base, len) = self.free.remove(idx);
            let seg = (base, n);
            if len > n {
                self.free.insert(0, (base + n, len - n));
            }
            for r in &mut self.refs[base..base + n] {
                assert_eq!(*r, 0, "alloc over a live page");
                *r = 1;
            }
            Some(seg)
        }

        /// +1 on every page of the run (the cache's hold on its prefix).
        fn ref_(&mut self, base: usize, n: usize) {
            if n == 0 {
                return;
            }
            assert!(base + n <= self.total, "ref out of bounds");
            for r in &mut self.refs[base..base + n] {
                assert!(*r >= 1, "ref on a free page");
                *r += 1;
            }
        }

        /// -1 on every page; pages reaching 0 return to the free list.
        fn free(&mut self, base: usize, n: usize) {
            self.drop_refs(base, n);
        }

        /// Alias of `free` — the explicit name for the eviction path.
        fn unref(&mut self, base: usize, n: usize) {
            self.drop_refs(base, n);
        }

        fn drop_refs(&mut self, base: usize, n: usize) {
            if n == 0 {
                return;
            }
            assert!(base + n <= self.total, "out of bounds");
            let mut zeroed: Vec<usize> = Vec::new();
            for p in base..base + n {
                assert!(self.refs[p] >= 1, "double free / unref of a free page {p}");
                self.refs[p] -= 1;
                if self.refs[p] == 0 {
                    zeroed.push(p);
                }
            }
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for p in zeroed {
                match runs.last_mut() {
                    Some((rb, rl)) if *rb + *rl == p => *rl += 1,
                    _ => runs.push((p, 1)),
                }
            }
            for run in runs {
                self.insert_run(run);
            }
        }

        /// Insert a run at the front, coalescing adjacent runs.
        fn insert_run(&mut self, mut run: (usize, usize)) {
            let mut i = 0;
            while i < self.free.len() {
                let other = self.free[i];
                if other.0 + other.1 == run.0 {
                    run = (other.0, run.1 + other.1);
                    self.free.remove(i);
                    continue;
                }
                if run.0 + run.1 == other.0 {
                    run = (run.0, run.1 + other.1);
                    self.free.remove(i);
                    continue;
                }
                i += 1;
            }
            self.free.insert(0, run);
        }

        fn in_use(&self) -> usize {
            self.refs.iter().filter(|&&r| r > 0).count()
        }

        fn free_count(&self) -> usize {
            self.free.iter().map(|&(_, l)| l).sum()
        }

        fn refcount(&self, base: usize, n: usize) -> Vec<u32> {
            self.refs[base..base + n].to_vec()
        }

        fn fragments_count(&self) -> usize {
            self.free.len()
        }

        /// Conservation invariants, mirroring `KvSegmentPool::assert_conserved`.
        fn assert_conserved(&self) {
            assert_eq!(self.in_use() + self.free_count(), self.total, "page conservation");
            let refs_sum: usize = self.refs.iter().map(|&r| r as usize).sum();
            assert_eq!(refs_sum, self.in_use(), "refcount sum == in-use pages");
            let mut covered = 0usize;
            for &(b, l) in &self.free {
                assert!(b + l <= self.total, "free run out of bounds");
                covered += l;
            }
            assert_eq!(covered, self.free_count(), "free runs cover the free pages");
            for (i, &(b1, l1)) in self.free.iter().enumerate() {
                for &(b2, l2) in &self.free[i + 1..] {
                    assert!(b1 >= b2 + l2 || b2 >= b1 + l1, "overlapping free runs");
                    assert!(b1 + l1 != b2 && b2 + l2 != b1, "adjacent free runs not coalesced");
                }
            }
        }
    }

    #[test]
    fn refill_free_leaves_prefix_ref1_and_suffix_freed() {
        // block = 2: a 9-token prompt spans ceil(9/2) = 5 pages; the cache
        // admits L = floor(9/2) = 4 aligned pages.
        let mut cache = TokenRadixCache::new_with_block(64, 2);
        let mut pool = MockPool::new(16);
        let prompt: Vec<u32> = key(0..9);
        let seg = pool.alloc(5).unwrap();
        assert_eq!(seg, (0, 5));
        let evicted = refill(&mut pool, &mut cache, &prompt, seg);
        assert!(evicted.is_empty());
        pool.assert_conserved();
        cache.assert_consistent();
        // Prefix pages are cache-owned (ref 1); the suffix page is back in
        // the pool (ref 0, coalesced with the untouched tail).
        assert_eq!(pool.refcount(0, 4), vec![1; 4], "prefix pages: cache-only ref");
        assert_eq!(pool.refcount(4, 1), vec![0], "suffix page returned to the pool");
        assert_eq!(pool.free_count(), 12);
        assert_eq!(pool.fragments_count(), 1, "suffix coalesces with the untouched tail");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.pages(), 4);
        // The cached aligned prefix is now servable.
        assert_eq!(cache.lookup(&prompt), Some(PrefixHit { base_page: 0, pages: 4, key_len: 8 }));
    }

    #[test]
    fn refill_same_key_keeps_a_single_entry() {
        let mut cache = TokenRadixCache::new_with_block(64, 2);
        let mut pool = MockPool::new(16);
        let prompt: Vec<u32> = key(0..4);
        // First request: else branch — ref_ + free + insert.
        let seg_a = pool.alloc(2).unwrap();
        refill(&mut pool, &mut cache, &prompt, seg_a);
        assert_eq!(cache.len(), 1);
        assert_eq!(pool.in_use(), 2);
        // Same prompt again: the aligned prefix is fully covered (D2 elif) —
        // the segment is released untouched, the tree keeps one entry.
        let seg_b = pool.alloc(2).unwrap();
        assert_eq!(seg_b, (2, 2), "freed suffix space is reused");
        let evicted = refill(&mut pool, &mut cache, &prompt, seg_b);
        assert!(evicted.is_empty());
        pool.assert_conserved();
        cache.assert_consistent();
        assert_eq!(cache.len(), 1, "radix tree keeps a single entry (spec T2)");
        assert_eq!(cache.entry_table()[0].base_page, 0);
        assert_eq!(pool.refcount(0, 2), vec![1; 2], "first request's run still cache-owned");
        assert_eq!(pool.refcount(2, 2), vec![0; 2], "second request's segment fully released");
        assert_eq!(pool.in_use(), 2);
        assert_eq!(pool.free_count(), 14);
    }

    #[test]
    fn refill_shorter_prompt_is_covered_by_a_deeper_entry() {
        // A 6-token entry is cached first; a 4-token refill shares its
        // aligned prefix — no second entry, nothing leaked.
        let mut cache = TokenRadixCache::new_with_block(64, 2);
        let mut pool = MockPool::new(16);
        let long: Vec<u32> = key(0..6);
        let seg_long = pool.alloc(3).unwrap();
        refill(&mut pool, &mut cache, &long, seg_long);
        let short: Vec<u32> = key(0..4);
        let seg_short = pool.alloc(2).unwrap();
        let evicted = refill(&mut pool, &mut cache, &short, seg_short);
        assert!(evicted.is_empty());
        pool.assert_conserved();
        cache.assert_consistent();
        assert_eq!(cache.len(), 1, "the deeper entry covers the shorter prompt");
        assert_eq!(pool.refcount(0, 3), vec![1; 3], "only the deeper run is cache-owned");
        assert_eq!(pool.refcount(3, 2), vec![0; 2], "the short segment was fully released");
        assert_eq!(pool.in_use(), 3);
    }

    #[test]
    fn refill_below_min_blocks_free_only() {
        let mut cache = TokenRadixCache::new_with_block(64, 2);
        let mut pool = MockPool::new(8);
        // 3 tokens -> L = 1 < MIN_BLOCKS: straight pool.free (spec D2).
        let prompt: Vec<u32> = key(0..3);
        let seg = pool.alloc(2).unwrap();
        let evicted = refill(&mut pool, &mut cache, &prompt, seg);
        assert!(evicted.is_empty());
        pool.assert_conserved();
        assert_eq!(pool.in_use(), 0, "everything returned");
        assert_eq!(pool.free_count(), 8);
        assert!(cache.is_empty());
        assert_eq!(cache.lookup(&prompt), None);
    }

    #[test]
    fn refill_oversized_entry_returns_the_prefix_ref() {
        // Budget 2 < 3 requested pages: the cache refuses, the caller gives
        // the ref_ back — no leak, nothing cached.
        let mut cache = TokenRadixCache::new_with_block(2, 2);
        let mut pool = MockPool::new(8);
        let prompt: Vec<u32> = key(0..6);
        let seg = pool.alloc(3).unwrap();
        let evicted = refill(&mut pool, &mut cache, &prompt, seg);
        assert!(evicted.is_empty());
        pool.assert_conserved();
        assert_eq!(pool.in_use(), 0, "refill returned everything");
        assert!(cache.is_empty());
    }

    #[test]
    fn clear_eject_returns_all_cached_pages() {
        let mut cache = TokenRadixCache::new_with_block(64, 2);
        let mut pool = MockPool::new(32);
        for i in 0..3u32 {
            let prompt: Vec<u32> = (0..6).map(|t| t + i * 100).collect();
            let seg = pool.alloc(3).unwrap();
            refill(&mut pool, &mut cache, &prompt, seg);
        }
        pool.assert_conserved();
        cache.assert_consistent();
        assert_eq!(pool.in_use(), 9, "three cache-owned runs");
        // Eject: every cached run is unref-ed back to the pool.
        let evicted = cache.clear();
        assert_eq!(evicted.len(), 3);
        for e in &evicted {
            pool.unref(e.base_page as usize, e.pages as usize);
        }
        pool.assert_conserved();
        assert_eq!(pool.in_use(), 0, "eject: all pages back to ref 0");
        assert_eq!(pool.free_count(), 32);
        assert_eq!(pool.fragments_count(), 1, "whole pool coalesced");
        assert!(cache.is_empty());
        assert_eq!(cache.pages(), 0);
    }

    #[test]
    fn eviction_pressure_unrefs_conservation() {
        // Budget 4 with 3-page entries: every new refill evicts the oldest
        // and returns its pages — conservation holds at every step.
        let mut cache = TokenRadixCache::new_with_block(4, 2);
        let mut pool = MockPool::new(16);
        let mut evicts: Vec<Vec<Evicted>> = Vec::new();
        for i in 0..4u32 {
            let prompt: Vec<u32> = (0..6).map(|t| t + i * 100).collect();
            let seg = pool.alloc(3).unwrap();
            evicts.push(refill(&mut pool, &mut cache, &prompt, seg));
            pool.assert_conserved();
            cache.assert_consistent();
        }
        // Each admission after the first evicted its predecessor exactly.
        // The pool reuses the just-freed run (recency first-fit), so the
        // bases alternate 0, 3, 0.
        assert_eq!(evicts[0], vec![]);
        assert_eq!(evicts[1], vec![Evicted { base_page: 0, pages: 3 }]);
        assert_eq!(evicts[2], vec![Evicted { base_page: 3, pages: 3 }]);
        assert_eq!(evicts[3], vec![Evicted { base_page: 0, pages: 3 }]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.pages(), 3);
        assert_eq!(pool.in_use(), 3, "only the newest run stays cache-owned");
        assert_eq!(pool.free_count(), 13);
        assert_eq!(
            cache.lookup(&(0..6).map(|t| t + 300).collect::<Vec<u32>>()),
            Some(PrefixHit { base_page: 3, pages: 3, key_len: 6 })
        );
        assert_eq!(cache.lookup(&(0..6u32).collect::<Vec<u32>>()), None, "evicted");
    }

    // ------------------------------------------------------------------
    // Determinism contract (SchedDeterminism style)
    // ------------------------------------------------------------------

    /// Everything observable in one scripted run: entry table, eviction
    /// traces, hit traces, pool snapshots, cached pages, entry count.
    type ScriptTraces = (
        Vec<EntryView>,
        Vec<Vec<Evicted>>,
        Vec<Option<PrefixHit>>,
        Vec<(usize, usize, usize)>,
        u64,
        usize,
    );

    /// One scripted (alloc, refill, lookup) run over fresh (cache, pool);
    /// returns every observable trace for the double-run identity check.
    fn run_scripted_sequence() -> ScriptTraces {
        let mut cache = TokenRadixCache::new_with_block(6, 2);
        let mut pool = MockPool::new(20);
        let key_a: Vec<u32> = key(0..4);
        let key_b: Vec<u32> = key(0..6);
        let key_c: Vec<u32> = vec![0, 1, 9, 9];
        let key_d: Vec<u32> = key(5..9);
        let mut evicts: Vec<Vec<Evicted>> = Vec::new();
        let mut hits: Vec<Option<PrefixHit>> = Vec::new();
        let mut snapshots: Vec<(usize, usize, usize)> = Vec::new();
        let snapshot = |pool: &MockPool| (pool.in_use(), pool.free_count(), pool.free.len());
        // 1. chain: a, then b extends it.
        let seg = pool.alloc(2).unwrap();
        evicts.push(refill(&mut pool, &mut cache, &key_a, seg));
        hits.push(cache.lookup(&key_a));
        let seg = pool.alloc(3).unwrap();
        evicts.push(refill(&mut pool, &mut cache, &key_b, seg));
        hits.push(cache.lookup(&key_b));
        snapshots.push(snapshot(&pool));
        // 2. re-use of a: covered by b -> elif, no new entry.
        let seg = pool.alloc(2).unwrap();
        evicts.push(refill(&mut pool, &mut cache, &key_a, seg));
        snapshots.push(snapshot(&pool));
        // 3. branch at token 2, budget pressure evicts the oldest (a).
        let seg = pool.alloc(2).unwrap();
        evicts.push(refill(&mut pool, &mut cache, &key_c, seg));
        hits.push(cache.lookup(&key_a)); // still servable through b
        snapshots.push(snapshot(&pool));
        // 4. unrelated key evicts b.
        let seg = pool.alloc(2).unwrap();
        evicts.push(refill(&mut pool, &mut cache, &key_d, seg));
        hits.push(cache.lookup(&key_b)); // shared 2-token prefix only
        snapshots.push(snapshot(&pool));
        // 5. a again: b is gone, so a's aligned prefix must be re-admitted.
        let seg = pool.alloc(2).unwrap();
        evicts.push(refill(&mut pool, &mut cache, &key_a, seg));
        hits.push(cache.lookup(&key_a));
        snapshots.push(snapshot(&pool));
        let table = cache.entry_table();
        let pages = cache.pages();
        let len = cache.len();
        (table, evicts, hits, snapshots, pages, len)
    }

    #[test]
    fn same_op_sequence_replays_identically() {
        // SchedDeterminism-style: two identical (alloc, refill, lookup,
        // evict) sequences on fresh state yield bit-identical traces.
        let a = run_scripted_sequence();
        let b = run_scripted_sequence();
        assert_eq!(a.0, b.0, "entry tables identical");
        assert_eq!(a.1, b.1, "eviction traces identical");
        assert_eq!(a.2, b.2, "hit traces identical");
        assert_eq!(a.3, b.3, "pool return sequences identical");
        assert_eq!(a.4, b.4, "cached pages identical");
        assert_eq!(a.5, b.5, "entry count identical");
        // Sanity on the scripted run: everything observable stayed
        // conservative and the interesting transitions did fire.
        let (table, evicts, hits, snapshots, pages, len) = a;
        assert_eq!(pages, 6, "budget exactly full");
        assert_eq!(len, 3);
        assert_eq!(evicts[3], vec![Evicted { base_page: 0, pages: 2 }], "a evicted at step 3");
        assert_eq!(evicts[4], vec![Evicted { base_page: 2, pages: 3 }], "b evicted at step 4");
        assert_eq!(
            hits[2],
            Some(PrefixHit { base_page: 2, pages: 2, key_len: 4 }),
            "a servable through b after its entry was evicted"
        );
        assert_eq!(
            hits[3],
            Some(PrefixHit { base_page: 5, pages: 1, key_len: 2 }),
            "after b is evicted only the shared 2-token prefix (c) remains"
        );
        assert_eq!(
            hits[4],
            Some(PrefixHit { base_page: 2, pages: 2, key_len: 4 }),
            "a re-admitted at the pool-reused base"
        );
        assert_eq!(snapshots[4].0, 6, "3 cache-owned runs of 2 pages each");
        assert_eq!(snapshots[4].0 + snapshots[4].1, 20, "pool conservation");
        assert_eq!(table.len(), 3);
    }

    #[test]
    fn eviction_and_lookup_churn_stays_consistent() {
        // Fixed-seed churn (deterministic — no RNG draws): interleaved
        // inserts, touches, lookups and budget pressure with both internal
        // invariants asserted after every step.
        let mut cache = TokenRadixCache::new_with_block(5, 2);
        let mut pool = MockPool::new(12);
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        for step in 0..300usize {
            let i = (next() % 5) as u32;
            let prompt: Vec<u32> = (0..6).map(|t| t + i * 10).collect();
            match next() % 3 {
                0 | 1 => {
                    if let Some(seg) = pool.alloc(3) {
                        refill(&mut pool, &mut cache, &prompt, seg);
                    }
                }
                _ => {
                    let _ = cache.lookup(&prompt);
                    let _ = cache.touch(&prompt);
                }
            }
            pool.assert_conserved();
            cache.assert_consistent();
            assert!(cache.pages() <= cache.max_pages(), "budget respected (step {step})");
        }
        let evicted = cache.clear();
        for e in &evicted {
            pool.unref(e.base_page as usize, e.pages as usize);
        }
        pool.assert_conserved();
        assert_eq!(pool.in_use(), 0, "churn ends fully returned");
        assert_eq!(pool.free_count(), 12);
        assert_eq!(pool.fragments_count(), 1, "whole pool coalesced");
        cache.assert_consistent();
    }
}
