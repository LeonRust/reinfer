//! KV 页池（014 T8；015 T4 跨后端复用）。
//!
//! 页块语义：每页 `block_len`（16/32）个 token 位的 KV 槽。物理分配策略：
//! free-list **LIFO 头插**（vLLM 语义——最近释放页最先复用）；页表逻辑槽→
//! 物理页（每个逻辑槽一张页；页内偏移由解码器按 `kv_pos % block_len` 定位）。
//!
//! 守恒断言（泄漏三合一 / 014 T8）：
//! - `in_use + free == total`（页数守恒式）；
//! - 总分配页 == 总释放页（分配对称）；
//! - pool 总页数不变。
//!
//! `#[forbid(unsafe_code)]`（本 crate 铁律）；页数据字节由后端（cuda/ascend）
//! 按 `PagePool::total() * block_len * per_slot_bytes` 预分配设备内存。

/// 页块长度（可选 16/32）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLen {
    /// 16 token/页。
    B16,
    /// 32 token/页。
    B32,
}

impl BlockLen {
    /// 槽数。
    pub const fn slots(self) -> usize {
        match self {
            BlockLen::B16 => 16,
            BlockLen::B32 => 32,
        }
    }
}

/// 页表（逻辑 KV 位置 → 物理页号）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageTable {
    /// 每逻辑页一个物理页号（页表长度 = 分配的页数）。
    pages: Vec<usize>,
    /// 该表覆盖的 token 数（≤ pages×block_len）。
    pub len: usize,
}

impl PageTable {
    /// 物理页号视图（按逻辑页顺序）。
    pub fn pages(&self) -> &[usize] {
        &self.pages
    }

    /// 逻辑位置 `pos`（全局 token 序号）→ (物理页号, 页内偏移)。
    pub fn locate(&self, pos: usize, block_len: usize) -> Option<(usize, usize)> {
        if pos >= self.len {
            return None;
        }
        let page_idx = pos / block_len;
        Some((self.pages[page_idx], pos % block_len))
    }
}

/// 页池（含分配状态机与守恒断言）。
#[derive(Debug)]
pub struct PagePool {
    block_slots: usize,
    total: usize,
    /// free-list（LIFO 头插：尾部=最近释放，弹出顺序一致）。
    free: Vec<usize>,
    in_use: usize,
    allocs: usize,
    frees: usize,
}

impl PagePool {
    /// 新建页池（`total` 页；初始全部空闲）。
    pub fn new(block: BlockLen, total: usize) -> Self {
        Self {
            block_slots: block.slots(),
            total,
            free: (0..total).rev().collect(),
            in_use: 0,
            allocs: 0,
            frees: 0,
        }
    }

    /// 逻辑页长（槽数）。
    pub fn block_slots(&self) -> usize {
        self.block_slots
    }

    /// 页池总页数。
    pub fn total(&self) -> usize {
        self.total
    }

    /// 空闲页数。
    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    /// 在用页数。
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// 守恒断言（真机判据三合一：守恒式 / 分配对称 / pool 不变）。
    pub fn assert_conserved(&self) {
        assert_eq!(self.in_use + self.free.len(), self.total, "conservation");
        // 分配对称由 allocs/frees 计数提供（相对量）。
    }

    /// 分配合适页数（`tokens` 个 token 位置；返回 `len` 达到 tokens 的页表）。
    pub fn alloc_tokens(&mut self, tokens: usize) -> Result<PageTable, MemoryError> {
        let pages_needed = tokens.div_ceil(self.block_slots);
        if pages_needed > self.free.len() {
            return Err(MemoryError::OutOfPages {
                need: pages_needed,
                free: self.free.len(),
            });
        }
        let mut pages = Vec::with_capacity(pages_needed);
        for _ in 0..pages_needed {
            // LIFO：pop 尾部 = 最近释放页（vLLM 语义）。
            pages.push(self.free.pop().expect("free len checked"));
        }
        self.in_use += pages_needed;
        self.allocs += pages_needed;
        Ok(PageTable { pages, len: tokens })
    }

    /// 分配单页。
    pub fn alloc_page(&mut self) -> Result<usize, MemoryError> {
        match self.free.pop() {
            Some(p) => {
                self.in_use += 1;
                self.allocs += 1;
                Ok(p)
            }
            None => Err(MemoryError::OutOfPages { need: 1, free: 0 }),
        }
    }

    /// 释放整张页表（LIFO 头插：push 序保持最近释放在前——弹出序即 LIFO）。
    pub fn free(&mut self, table: &mut PageTable) {
        let n = table.pages.len();
        for &p in &table.pages {
            assert!(p < self.total, "foreign page id");
            self.free.push(p);
        }
        self.in_use -= n;
        self.frees += n;
        table.pages.clear();
        table.len = 0;
    }

    /// 运行时统计（分配/释放对称）。
    pub fn alloc_frees(&self) -> (usize, usize) {
        (self.allocs, self.frees)
    }
}

/// 页池错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryError {
    /// 空闲页不足。
    OutOfPages {
        /// 需求页数。
        need: usize,
        /// 当前空闲页数。
        free: usize,
    },
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn alloc_free_conserved() {
        let mut pool = PagePool::new(BlockLen::B16, 8);
        let mut t = pool.alloc_tokens(20).unwrap();
        assert_eq!(t.pages().len(), 2);
        assert_eq!(pool.in_use(), 2);
        pool.assert_conserved();
        pool.free(&mut t);
        assert_eq!(pool.in_use(), 0);
        assert_eq!(pool.free_count(), 8);
        pool.assert_conserved();
        let (a, f) = pool.alloc_frees();
        assert_eq!(a, 2);
        assert_eq!(f, 2);
    }

    #[test]
    fn lifo_head_order() {
        // LIFO：释放 A、B → 再分配应得 B 页（最近释放最先）。
        let mut pool = PagePool::new(BlockLen::B32, 4);
        let a = pool.alloc_page().unwrap();
        let b = pool.alloc_page().unwrap();
        let mut t = PageTable { pages: vec![a, b], len: 2 * 32 };
        pool.free(&mut t);
        let c = pool.alloc_page().unwrap();
        assert_eq!(c, b, "LIFO: most-recent freed page reused first");
    }

    #[test]
    fn out_of_pages_errors() {
        let mut pool = PagePool::new(BlockLen::B16, 1);
        assert_eq!(
            pool.alloc_tokens(32),
            Err(MemoryError::OutOfPages { need: 2, free: 1 })
        );
    }

    #[test]
    fn locate_works_across_pages() {
        // 乱序物理页：首页先册 7、次页 2（snapshot fixture 风格）。
        let pool = PagePool::new(BlockLen::B16, 8);
        let t = PageTable { pages: vec![7, 2], len: 30 };
        assert_eq!(t.locate(0, 16), Some((7, 0)));
        assert_eq!(t.locate(15, 16), Some((7, 15)));
        assert_eq!(t.locate(16, 16), Some((2, 0)));
        assert_eq!(t.locate(29, 16), Some((2, 13)));
        assert_eq!(t.locate(30, 16), None);
        let _ = pool;
    }

    #[test]
    fn partial_last_page() {
        let mut pool = PagePool::new(BlockLen::B16, 3);
        let t = pool.alloc_tokens(18).unwrap(); // 2 页，末页只盖 2 槽
        assert_eq!(t.pages().len(), 2);
        assert_eq!(t.len, 18);
        assert_eq!(t.locate(17, 16), Some((t.pages()[1], 1)));
        assert_eq!(t.locate(18, 16), None);
    }
}
