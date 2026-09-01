//! reinfer-safetensors：safetensors 模型文件读取（纯 Rust、无 unsafe）。
//!
//! 文件格式：`u64 LE header_len` + JSON header（每张量：dtype/shape/
//! data_offsets） + 数据段（offset 对齐 8）。本 crate 只提供**视图**
//! （字节切片），数值转换由消费方（tokenizer/engine）按 dtype 处理，与
//! GGUF 路径同构（`safe tensors 是统一模型对象`——模型文件格式无特判）。

#![forbid(unsafe_code)]

use serde_json::Value;
use std::path::Path;

/// 张量 dtype。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StDtype {
    /// IEEE 单精度（F32）。
    F32,
    /// IEEE 半精度（F16）。
    F16,
    /// bfloat16（BF16）。
    Bf16,
    /// 其它（预留）。
    Other(String),
}

impl StDtype {
    /// 元素字节数。
    pub fn size(self) -> usize {
        match self {
            StDtype::F32 => 4,
            StDtype::F16 | StDtype::Bf16 => 2,
            StDtype::Other(_) => 2,
        }
    }
}

/// 单个张量条目（data 视图 = 文件数据段 slice）。
#[derive(Debug)]
pub struct TensorView<'a> {
    /// 存储 dtype。
    pub dtype: StDtype,
    /// 形状（按 header 顺序）。
    pub shape: Vec<u64>,
    /// 数据字节（数据段内）。
    pub bytes: &'a [u8],
}

impl TensorView<'_> {
    /// 元素数（溢出 → None）。
    pub fn len(&self) -> Option<usize> {
        self.shape.iter().try_fold(1u64, |a, d| a.checked_mul(*d)).map(|n| n as usize)
    }

    /// 元素字节数（与 dtype 一致）。
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

/// 文件视图（header JSON + 数据段）。
#[derive(Debug)]
pub struct SafeFile<'a> {
    /// 解析后的 header JSON（`__metadata__` 键含文件级元数据）。
    pub header: Value,
    data: &'a [u8],
}

impl<'a> SafeFile<'a> {
    /// 内存缓冲解析（文件读入的 Vec 亦可）。
    pub fn from_bytes(buf: &'a [u8]) -> Result<Self, Error> {
        if buf.len() < 8 {
            return Err(Error::Truncated { at: 0, needed: 8 });
        }
        let len = u64::from_le_bytes(buf[..8].try_into().expect("8 bytes")) as usize;
        let end = 8usize.checked_add(len).ok_or(Error::BadLength)?;
        if end > buf.len() {
            return Err(Error::Truncated { at: end, needed: buf.len() });
        }
        let header: Value = serde_json::from_slice(&buf[8..end]).map_err(Error::Json)?;
        if !header.is_object() {
            return Err(Error::NotObject);
        }
        Ok(Self { header, data: &buf[end..] })
    }

    /// 文件解析（权重常驻：Box::leak 到 'static）。
    pub fn open(path: &Path) -> Result<SafeFile<'static>, Error> {
        let bytes = std::fs::read(path).map_err(Error::Io)?;
        let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        Ok(SafeFile::from_bytes(leaked)?)
    }

    /// 张量查看（header `data_offsets` 定位数据段）。
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, Error> {
        let obj = self.header.as_object().ok_or(Error::NotObject)?;
        let v = obj.get(name).ok_or(Error::MissingTensor(name.to_string()))?;
        let dtype = match v.get("dtype").and_then(|d| d.as_str()) {
            Some("F32") => StDtype::F32,
            Some("F16") => StDtype::F16,
            Some("BF16") => StDtype::Bf16,
            Some(other) => StDtype::Other(other.to_string()),
            None => return Err(Error::BadHeader("missing dtype")),
        };
        let offs = v
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or(Error::BadHeader("missing data_offsets"))?;
        let (s, e) = match (offs.first(), offs.get(1)) {
            (Some(Value::Number(s)), Some(Value::Number(e))) => (
                s.as_u64().ok_or(Error::BadHeader("offset not u64"))? as usize,
                e.as_u64().ok_or(Error::BadHeader("offset not u64"))? as usize,
            ),
            _ => return Err(Error::BadHeader("offset not u64 pair")),
        };
        if e > self.data.len() || s > e {
            return Err(Error::BadLength);
        }
        let shape = v
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<u64>>())
            .unwrap_or_default();
        Ok(TensorView { dtype, shape, bytes: &self.data[s..e] })
    }
}

/// safetensors 错误面。
#[derive(Debug)]
pub enum Error {
    /// 底层 IO。
    Io(std::io::Error),
    /// 头/数据截断。
    Truncated {
        /// 截断点字节偏移。
        at: usize,
        /// 需要的最小文件长度。
        needed: usize,
    },
    /// 长度非法。
    BadLength,
    /// header JSON 解析失败。
    Json(serde_json::Error),
    /// header 非对象。
    NotObject,
    /// 缺失张量。
    MissingTensor(String),
    /// header 非法。
    BadHeader(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io: {e}"),
            Error::Truncated { at, needed } => write!(f, "truncated at {at} need {needed}"),
            Error::BadLength => write!(f, "bad length"),
            Error::Json(e) => write!(f, "json: {e}"),
            Error::NotObject => write!(f, "header not object"),
            Error::MissingTensor(n) => write!(f, "missing tensor {n}"),
            Error::BadHeader(m) => write!(f, "bad header: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn parse_minimal() {
        let mut buf = Vec::new();
        let header = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&1.0f32.to_le_bytes());
        buf.extend_from_slice(&2.0f32.to_le_bytes());
        buf.extend_from_slice(&3.0f32.to_le_bytes());
        buf.extend_from_slice(&4.0f32.to_le_bytes());
        let f = SafeFile::from_bytes(&buf).unwrap();
        let t = f.tensor("a").unwrap();
        assert_eq!(t.dtype, StDtype::F32);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.bytes.len(), 16);
        assert_eq!(f32::from_le_bytes(t.bytes[4..8].try_into().unwrap()), 2.0);
        assert!(f.tensor("x").is_err());
    }

    #[test]
    fn truncated_rejected() {
        assert!(SafeFile::from_bytes(&[0u8; 4]).is_err());
    }
}
