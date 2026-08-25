//! reinfer-jit：JitCache（仿 FlashInfer 三段式 + FileLock）
#![allow(unsafe_code)]  // 窄 FFI 宿主：unsafe 只允许出现在这里
