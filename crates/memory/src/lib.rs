//! reinfer-memory：页式 KV 分配器、引用计数、页表；KV 预算公式
#![forbid(unsafe_code)]
pub mod budget;
pub mod pool;
pub mod segment;
