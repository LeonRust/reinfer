//! GGUF 文件格式的类型面：张量 dtype、元数据值、张量描述与错误分类。
//!
//! 锚定 `specs/001-gguf-loader`（T2/T3 数据类型约定）与 `specs/014-cuda-l3-single-request`（T1）。
//! 布局事实与 llama.cpp `gguf.h` / `gguf-py` 一致：头部 24 字节、元数据 KV 表、
//! 张量表（名称/维度/类型/偏移）、数据区（32 字节对齐）。

use std::fmt;

use reinfer_kernels::LaunchError;

/// GGUF 数据区对齐字节数（规范要求每个张量数据从 32 字节对齐的文件偏移开始）。
pub const GGUF_ALIGNMENT: u64 = 32;

/// 头部固定长度：magic(4) + version(4) + n_tensors(8) + n_kv(8)。
pub(crate) const HEADER_SIZE: usize = 24;

/// 元数据值类型码（GGUF spec / gguf.h `GGUFTypes`）。解析与测试 fixture 共用。
pub(crate) mod type_code {
    pub const U8: u32 = 0;
    pub const I8: u32 = 1;
    pub const U16: u32 = 2;
    pub const I16: u32 = 3;
    pub const U32: u32 = 4;
    pub const I32: u32 = 5;
    pub const F32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STR: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const U64: u32 = 10;
    pub const I64: u32 = 11;
    pub const F64: u32 = 12;
}

/// GGML 张量 dtype 码（`ggml.h` `ggml_type` 白名单子集）。
///
/// 未收录/未知码保留为 [`GgufDtype::Other`]，不拒绝解析（格式兼容优先）；
/// 是否可解码由 T2 codec 的 gate 决定。大小未知的类型（K 系/IQ 系）数据长度
/// 按张量间距推断（见 `GgufTensor::length`）。
// 变体名沿用 GGML 命名（q4_0/q8_k/iq2_xxs 等），故豁免命名风格 lint。
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufDtype {
    /// 32 位浮点。
    F32,
    /// 16 位半精度浮点。
    F16,
    /// 4 位块量化（block 32，18 B/block）。
    Q4_0,
    /// 4 位块量化带 min/max（block 32，20 B/block）。
    Q4_1,
    /// 5 位块量化（block 32，22 B/block）。
    Q5_0,
    /// 5 位块量化带 min/max（block 32，24 B/block）。
    Q5_1,
    /// 8 位块量化（block 32，34 B/block）。
    Q8_0,
    /// 8 位块量化双 scale（block 32，36 B/block）。
    Q8_1,
    /// K 系 2 位量化（block 256；大小表未收录，长度按间距推断）。
    Q2_K,
    /// K 系 3 位量化（block 256；同上）。
    Q3_K,
    /// K 系 4 位量化（block 256；同上）。
    Q4_K,
    /// K 系 5 位量化（block 256；同上）。
    Q5_K,
    /// K 系 6 位量化（block 256；同上）。
    Q6_K,
    /// K 系 8 位量化（block 256；同上）。
    Q8_K,
    /// 8 位有符号整数。
    I8,
    /// 16 位有符号整数。
    I16,
    /// 32 位有符号整数。
    I32,
    /// IQ 2 位超稀疏量化（xxs）。
    IQ2_XXS,
    /// IQ 2 位超稀疏量化（xs）。
    IQ2_XS,
    /// IQ 3 位超稀疏量化（xxs）。
    IQ3_XXS,
    /// IQ 1 位超稀疏量化（s）。
    IQ1_S,
    /// IQ 4 位超稀疏量化（nl）。
    IQ4_NL,
    /// IQ 3 位超稀疏量化（s）。
    IQ3_S,
    /// IQ 2 位超稀疏量化（s）。
    IQ2_S,
    /// IQ 4 位超稀疏量化（xs）。
    IQ4_XS,
    /// 64 位有符号整数。
    I64,
    /// 16 位 bfloat16。
    BF16,
    /// IQ 1 位量化（m）。
    IQ1_M,
    /// IQ 4 位量化（l）。
    IQ4_L,
    /// 64 位浮点。
    F64,
    /// IQ 2 位量化（m）。
    IQ2_M,
    /// IQ 3 位量化（m）。
    IQ3_M,
    /// IQ 3 位量化（xs）。
    IQ3_XS,
    /// IQ 1 位量化（sn）。
    IQ1_SN,
    /// 未收录/未知类型码（原样保留，便于诊断与兼容）。
    Other(u32),
}

impl GgufDtype {
    /// 由 GGML 类型码构造（未知码 → [`GgufDtype::Other`]）。
    pub fn from_code(code: u32) -> GgufDtype {
        match code {
            0 => GgufDtype::F32,
            1 => GgufDtype::F16,
            2 => GgufDtype::Q4_0,
            3 => GgufDtype::Q4_1,
            6 => GgufDtype::Q5_0,
            7 => GgufDtype::Q5_1,
            8 => GgufDtype::Q8_0,
            9 => GgufDtype::Q8_1,
            10 => GgufDtype::Q2_K,
            11 => GgufDtype::Q3_K,
            12 => GgufDtype::Q4_K,
            13 => GgufDtype::Q5_K,
            14 => GgufDtype::Q6_K,
            15 => GgufDtype::Q8_K,
            16 => GgufDtype::I8,
            17 => GgufDtype::I16,
            18 => GgufDtype::I32,
            19 => GgufDtype::IQ2_XXS,
            20 => GgufDtype::IQ2_XS,
            21 => GgufDtype::IQ3_XXS,
            22 => GgufDtype::IQ1_S,
            23 => GgufDtype::IQ4_NL,
            24 => GgufDtype::IQ3_S,
            25 => GgufDtype::IQ2_S,
            26 => GgufDtype::IQ4_XS,
            27 => GgufDtype::I64,
            28 => GgufDtype::BF16,
            29 => GgufDtype::IQ1_M,
            30 => GgufDtype::IQ4_L,
            31 => GgufDtype::F64,
            32 => GgufDtype::IQ2_M,
            33 => GgufDtype::IQ3_M,
            34 => GgufDtype::IQ3_XS,
            35 => GgufDtype::IQ1_SN,
            other => GgufDtype::Other(other),
        }
    }

    /// 原始 GGML 类型码（诊断/写入器用）。
    pub fn type_code(self) -> u32 {
        match self {
            GgufDtype::F32 => 0,
            GgufDtype::F16 => 1,
            GgufDtype::Q4_0 => 2,
            GgufDtype::Q4_1 => 3,
            GgufDtype::Q5_0 => 6,
            GgufDtype::Q5_1 => 7,
            GgufDtype::Q8_0 => 8,
            GgufDtype::Q8_1 => 9,
            GgufDtype::Q2_K => 10,
            GgufDtype::Q3_K => 11,
            GgufDtype::Q4_K => 12,
            GgufDtype::Q5_K => 13,
            GgufDtype::Q6_K => 14,
            GgufDtype::Q8_K => 15,
            GgufDtype::I8 => 16,
            GgufDtype::I16 => 17,
            GgufDtype::I32 => 18,
            GgufDtype::IQ2_XXS => 19,
            GgufDtype::IQ2_XS => 20,
            GgufDtype::IQ3_XXS => 21,
            GgufDtype::IQ1_S => 22,
            GgufDtype::IQ4_NL => 23,
            GgufDtype::IQ3_S => 24,
            GgufDtype::IQ2_S => 25,
            GgufDtype::IQ4_XS => 26,
            GgufDtype::I64 => 27,
            GgufDtype::BF16 => 28,
            GgufDtype::IQ1_M => 29,
            GgufDtype::IQ4_L => 30,
            GgufDtype::F64 => 31,
            GgufDtype::IQ2_M => 32,
            GgufDtype::IQ3_M => 33,
            GgufDtype::IQ3_XS => 34,
            GgufDtype::IQ1_SN => 35,
            GgufDtype::Other(c) => c,
        }
    }

    /// 稳定显示名（诊断/对拍日志用）。
    pub fn name(self) -> &'static str {
        match self {
            GgufDtype::F32 => "f32",
            GgufDtype::F16 => "f16",
            GgufDtype::Q4_0 => "q4_0",
            GgufDtype::Q4_1 => "q4_1",
            GgufDtype::Q5_0 => "q5_0",
            GgufDtype::Q5_1 => "q5_1",
            GgufDtype::Q8_0 => "q8_0",
            GgufDtype::Q8_1 => "q8_1",
            GgufDtype::Q2_K => "q2_k",
            GgufDtype::Q3_K => "q3_k",
            GgufDtype::Q4_K => "q4_k",
            GgufDtype::Q5_K => "q5_k",
            GgufDtype::Q6_K => "q6_k",
            GgufDtype::Q8_K => "q8_k",
            GgufDtype::I8 => "i8",
            GgufDtype::I16 => "i16",
            GgufDtype::I32 => "i32",
            GgufDtype::IQ2_XXS => "iq2_xxs",
            GgufDtype::IQ2_XS => "iq2_xs",
            GgufDtype::IQ3_XXS => "iq3_xxs",
            GgufDtype::IQ1_S => "iq1_s",
            GgufDtype::IQ4_NL => "iq4_nl",
            GgufDtype::IQ3_S => "iq3_s",
            GgufDtype::IQ2_S => "iq2_s",
            GgufDtype::IQ4_XS => "iq4_xs",
            GgufDtype::I64 => "i64",
            GgufDtype::BF16 => "bf16",
            GgufDtype::IQ1_M => "iq1_m",
            GgufDtype::IQ4_L => "iq4_l",
            GgufDtype::F64 => "f64",
            GgufDtype::IQ2_M => "iq2_m",
            GgufDtype::IQ3_M => "iq3_m",
            GgufDtype::IQ3_XS => "iq3_xs",
            GgufDtype::IQ1_SN => "iq1_sn",
            GgufDtype::Other(_) => "unknown",
        }
    }

    /// 是否拥有已知的大小公式（其余类型按张量间距推断长度）。
    pub fn size_known(self) -> bool {
        matches!(
            self,
            GgufDtype::F32
                | GgufDtype::F16
                | GgufDtype::Q4_0
                | GgufDtype::Q4_1
                | GgufDtype::Q5_0
                | GgufDtype::Q5_1
                | GgufDtype::Q8_0
                | GgufDtype::Q8_1
                | GgufDtype::I8
                | GgufDtype::I16
                | GgufDtype::I32
                | GgufDtype::I64
                | GgufDtype::BF16
                | GgufDtype::F64
        )
    }

    /// `n` 个元素的数据字节数（`None` = 类型大小未知，或元素数不满足块对齐）。
    pub fn size_bytes(self, n: u64) -> Option<u64> {
        // 块量化：block 32 元素 / N 字节（llama.cpp ggml_block_q*_struct 布局）
        let block = |block_elems: u64, block_bytes: u64| -> Option<u64> {
            if !n.is_multiple_of(block_elems) {
                return None;
            }
            (n / block_elems).checked_mul(block_bytes)
        };
        match self {
            GgufDtype::F32 => n.checked_mul(4),
            GgufDtype::F16 | GgufDtype::BF16 => n.checked_mul(2),
            GgufDtype::F64 => n.checked_mul(8),
            GgufDtype::I8 => n.checked_mul(1),
            GgufDtype::I16 => n.checked_mul(2),
            GgufDtype::I32 => n.checked_mul(4),
            GgufDtype::I64 => n.checked_mul(8),
            GgufDtype::Q4_0 => block(32, 18),
            GgufDtype::Q4_1 => block(32, 20),
            GgufDtype::Q5_0 => block(32, 22),
            GgufDtype::Q5_1 => block(32, 24),
            GgufDtype::Q8_0 => block(32, 34),
            GgufDtype::Q8_1 => block(32, 36),
            _ => None,
        }
    }
}

/// GGUF 元数据值（KV 值；类型码见 [`type_code`]）。
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    /// 8 位无符号整数。
    U8(u8),
    /// 8 位有符号整数。
    I8(i8),
    /// 16 位无符号整数。
    U16(u16),
    /// 16 位有符号整数。
    I16(i16),
    /// 32 位无符号整数（维度/计数等典型键的类型）。
    U32(u32),
    /// 32 位有符号整数。
    I32(i32),
    /// 32 位浮点。
    F32(f32),
    /// 布尔（文件内为 u8 0/1）。
    Bool(bool),
    /// 字符串。
    Str(String),
    /// 数组（元素类型统一，可嵌套）。
    Array(ArrayValue),
    /// 64 位无符号整数。
    U64(u64),
    /// 64 位有符号整数。
    I64(i64),
    /// 64 位浮点。
    F64(f64),
}

/// 元数据数组值（元素类型统一；`Nested` 用于数组的数组）。
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayValue {
    /// u8 数组。
    U8(Vec<u8>),
    /// i8 数组。
    I8(Vec<i8>),
    /// u16 数组。
    U16(Vec<u16>),
    /// i16 数组。
    I16(Vec<i16>),
    /// u32 数组（`tokenizer.ggml.token_type` 等）。
    U32(Vec<u32>),
    /// i32 数组。
    I32(Vec<i32>),
    /// f32 数组。
    F32(Vec<f32>),
    /// bool 数组。
    Bool(Vec<bool>),
    /// 字符串数组（`tokenizer.ggml.tokens` / `merges` 等）。
    Str(Vec<String>),
    /// u64 数组。
    U64(Vec<u64>),
    /// i64 数组。
    I64(Vec<i64>),
    /// f64 数组。
    F64(Vec<f64>),
    /// 数组的数组（每项各自带元素类型标签）。
    Nested(Vec<ArrayValue>),
}

/// 张量描述（文件内数据区间 + 形状/类型；数据本体按需经 `GgufReader::tensor_data` 读取）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensor {
    /// 张量名（文件内唯一，解析时校验）。
    pub name: String,
    /// 形状（维度序与 llama.cpp 一致：行主序，如 `[out, in]`）。
    pub shape: Vec<u64>,
    /// 量化/存储类型。
    pub dtype: GgufDtype,
    /// 数据区在文件中的绝对字节偏移（解析时校验 32 字节对齐）。
    pub offset: u64,
    /// 数据字节长度：已知类型=形状推导；未知类型=与下一张量（或文件尾）的间距。
    pub length: u64,
}

impl GgufTensor {
    /// 元素总数（形状乘积；溢出返回 `None`）。
    pub fn element_count(&self) -> Option<u64> {
        self.shape.iter().try_fold(1u64, |acc, d| acc.checked_mul(*d))
    }

    /// 数据区起点是否 32 字节对齐（解析保证为真；供诊断断言用）。
    pub fn offset_aligned(&self) -> bool {
        self.offset.is_multiple_of(GGUF_ALIGNMENT)
    }
}

/// GGUF 读取/解析错误（各变体携带字节偏移，便于定位损坏点）。
///
/// 不实现 `PartialEq`（`io::Error` 不支持）；测试用 `matches!` 断言变体。
#[derive(Debug)]
pub enum GgufError {
    /// 底层 I/O 失败（打开/读取）。
    Io(std::io::Error),
    /// 魔数不符（不是 GGUF 文件）。
    BadMagic {
        /// 实际读到的 4 字节。
        found: [u8; 4],
    },
    /// 版本不受支持（仅接受 2/3）。
    UnsupportedVersion(u32),
    /// 文件在 `at` 处截断，需要 `needed` 字节。
    Truncated {
        /// 实际可用字节数。
        at: u64,
        /// 需要的字节数。
        needed: u64,
    },
    /// 结构性损坏（坏 utf8/类型码/计数/深度/对齐）。
    Malformed {
        /// 损坏类别描述。
        what: &'static str,
        /// 出错处文件偏移。
        at: u64,
    },
    /// 张量数据区间越界/重叠。
    Oversized {
        /// 张量名。
        tensor: String,
        /// 数据起始偏移。
        offset: u64,
        /// 数据长度。
        length: u64,
        /// 允许的上界（下一张量起点或文件尾）。
        limit: u64,
    },
    /// 张量形状/属性非法。
    InvalidTensor {
        /// 张量名。
        name: String,
        /// 非法原因。
        why: &'static str,
    },
    /// 元数据键类型与访问器不符（`meta_*` 访问器误用）。
    InvalidMetadata {
        /// 键名。
        key: String,
        /// 期望的类型描述。
        why: &'static str,
    },
    /// 解量化失败：数据长度/缓冲不符（`at` 为上下文值，如 blob 长度或所需缓冲元素数）。
    BadData {
        /// 失败原因。
        what: &'static str,
        /// 上下文值（blob 长度或所需缓冲元素数）。
        at: u64,
    },
    /// 解量化不支持的 dtype（K 系/IQ 系等 codec 未实现）。
    UnsupportedDtype(GgufDtype),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::Io(e) => write!(f, "gguf io error: {e}"),
            GgufError::BadMagic { found } => {
                write!(f, "gguf: bad magic {found:02x?} (expected \"GGUF\")")
            }
            GgufError::UnsupportedVersion(v) => {
                write!(f, "gguf: unsupported version {v} (supported: 2, 3)")
            }
            GgufError::Truncated { at, needed } => {
                write!(f, "gguf: truncated at byte {at}, {needed} bytes needed")
            }
            GgufError::Malformed { what, at } => {
                write!(f, "gguf: malformed {what} at byte {at}")
            }
            GgufError::Oversized { tensor, offset, length, limit } => {
                write!(f, "gguf: tensor {tensor} data [{offset}, +{length}) exceeds limit {limit}")
            }
            GgufError::InvalidTensor { name, why } => {
                write!(f, "gguf: invalid tensor {name}: {why}")
            }
            GgufError::InvalidMetadata { key, why } => {
                write!(f, "gguf: metadata key {key}: {why}")
            }
            GgufError::BadData { what, at } => {
                write!(f, "gguf: bad tensor data: {what} (context {at})")
            }
            GgufError::UnsupportedDtype(dt) => {
                write!(f, "gguf: dtype {} not supported for dequantization", dt.name())
            }
        }
    }
}

impl std::error::Error for GgufError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GgufError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

impl From<GgufError> for LaunchError {
    /// 数据层失败一律归 `Fatal`：文件损坏/参数错误重试无意义；
    /// `Oom`/`Driver` 分类是执行层的语义，不适用于解析面。
    fn from(_: GgufError) -> Self {
        LaunchError::Fatal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_code_roundtrip_known() {
        for code in 0..=35u32 {
            let dt = GgufDtype::from_code(code);
            // 4/5 为历史废弃码（原 Q4_2/Q4_3），应落到 Other
            if matches!(dt, GgufDtype::Other(_)) {
                assert!(matches!(dt, GgufDtype::Other(c) if c == code));
            } else {
                assert_eq!(dt.type_code(), code, "code {code}");
            }
        }
        assert_eq!(GgufDtype::from_code(999).type_code(), 999);
        assert_eq!(GgufDtype::from_code(4).name(), "unknown");
    }

    #[test]
    fn dtype_names_are_stable() {
        assert_eq!(GgufDtype::F32.name(), "f32");
        assert_eq!(GgufDtype::Q8_0.name(), "q8_0");
        assert_eq!(GgufDtype::BF16.name(), "bf16");
        assert_eq!(GgufDtype::IQ1_SN.name(), "iq1_sn");
    }

    #[test]
    fn size_bytes_known_types() {
        assert_eq!(GgufDtype::F32.size_bytes(3), Some(12));
        assert_eq!(GgufDtype::F16.size_bytes(3), Some(6));
        assert_eq!(GgufDtype::BF16.size_bytes(3), Some(6));
        assert_eq!(GgufDtype::F64.size_bytes(3), Some(24));
        assert_eq!(GgufDtype::I32.size_bytes(3), Some(12));
        assert_eq!(GgufDtype::Q8_0.size_bytes(64), Some(68));
        assert_eq!(GgufDtype::Q8_0.size_bytes(63), None); // 块对齐不满足
        assert_eq!(GgufDtype::Q4_0.size_bytes(32), Some(18));
        assert_eq!(GgufDtype::Q8_1.size_bytes(32), Some(36));
        assert_eq!(GgufDtype::Q8_K.size_bytes(256), None); // 大小未知 → None
        assert_eq!(GgufDtype::Other(4).size_bytes(1), None);
        assert!(!GgufDtype::Q8_K.size_known());
        assert!(GgufDtype::F32.size_known());
    }

    #[test]
    fn element_count_overflow_detected() {
        let t = GgufTensor {
            name: "w".into(),
            shape: vec![u64::MAX, 2],
            dtype: GgufDtype::F32,
            offset: 1, // 非 32 对齐，配合校验 offset_aligned 的判假路径
            length: 0,
        };
        assert_eq!(t.element_count(), None);
        assert!(!t.offset_aligned());
    }

    #[test]
    fn error_display_is_readable() {
        let e = GgufError::Truncated { at: 24, needed: 100 };
        assert!(e.to_string().contains("truncated at byte 24"));
        let e = GgufError::BadMagic { found: [0x58, 0x58, 0x58, 0x58] };
        assert!(e.to_string().contains("bad magic"));
    }

    #[test]
    fn error_maps_to_launch_fatal() {
        let e = GgufError::UnsupportedVersion(1);
        assert_eq!(LaunchError::from(e), LaunchError::Fatal);
        assert!(!LaunchError::from(GgufError::BadMagic { found: [0; 4] }).retryable());
    }
}
