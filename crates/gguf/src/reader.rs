//! `GgufReader`：GGUF 头部/元数据/张量表解析 + 惰性权重读取。
//!
//! 实现选择：**不采用 mmap**。memmap2 的 `map` 自 0.7 起为 `unsafe`（映射期间文件被
//! 并发截断即 UB），与本仓数据管道 `#![forbid(unsafe_code)]` 冲突（014 spec Constraints）。
//! 改用 `FileExt::read_at`：元数据/张量表区一次性读入内存（块读摊薄系统调用），
//! 权重按需 pread——OS 页缓存同样按需加载，RSS 只随实际读取增长，与 mmap 语义等价。

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io;
use std::path::Path;

use crate::schema::type_code;
use crate::schema::{
    ArrayValue, GGUF_ALIGNMENT, GgufDtype, GgufError, GgufTensor, HEADER_SIZE, MetaValue,
};

/// 元数据区读入的块大小（摊薄系统调用）。
const READ_CHUNK: usize = 256 * 1024;
/// 嵌套数组最大深度（规范实际用不超过 2；防恶意文件栈溢出）。
const MAX_ARRAY_DEPTH: u32 = 8;

/// GGUF 文件读取器。
///
/// `open` 后元数据与张量表已就绪；权重数据经 [`GgufReader::tensor_data`] 按需读取。
#[derive(Debug)]
pub struct GgufReader {
    source: Source,
    file_len: u64,
    meta: ModelMeta,
    tensors: Vec<GgufTensor>,
    by_name: HashMap<String, u32>,
}

/// 数据源：磁盘文件（惰性 pread）或内存缓冲（测试/内嵌场景）。
#[derive(Debug)]
enum Source {
    File(File),
    Mem(Vec<u8>),
}

impl Source {
    /// 从 `offset` 读入 `buf`（返回实际读取字节数；0 = EOF）。
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        match self {
            Source::Mem(bytes) => {
                let start = usize::try_from(offset).unwrap_or(bytes.len());
                if start >= bytes.len() {
                    return Ok(0);
                }
                let n = buf.len().min(bytes.len() - start);
                buf[..n].copy_from_slice(&bytes[start..start + n]);
                Ok(n)
            }
            Source::File(file) => read_at_file(file, buf, offset),
        }
    }
}

#[cfg(unix)]
fn read_at_file(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    file.read_at(buf, offset)
}

#[cfg(not(unix))]
fn read_at_file(file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    // 非 unix 退路：单线程假设下的 seek+read（Linux 目标走上面的 read_at）。
    use std::io::{Read, Seek, SeekFrom};
    let mut f = file;
    f.seek(SeekFrom::Start(offset))?;
    f.read(buf)
}

impl GgufReader {
    /// 打开 GGUF 文件（元数据/张量表立即解析；权重数据惰性读取）。
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        Self::parse(Source::File(file), file_len)
    }

    /// 从内存缓冲解析（测试与内嵌场景；与 `open` 走同一解析路径）。
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GgufError> {
        let file_len = bytes.len() as u64;
        Self::parse(Source::Mem(bytes), file_len)
    }

    fn parse(source: Source, file_len: u64) -> Result<Self, GgufError> {
        let mut region = RegionBuf { source: &source, file_len, buf: Vec::new() };
        let (n_tensors, n_kv) = parse_header(&mut region)?;
        let (meta, mut pos) = parse_metadata(&mut region, n_kv)?;
        let tensors = parse_tensors(&mut region, n_tensors, &mut pos)?;
        drop(region);
        let mut by_name = HashMap::with_capacity(tensors.len());
        for (i, t) in tensors.iter().enumerate() {
            if by_name.insert(t.name.clone(), i as u32).is_some() {
                return Err(GgufError::InvalidTensor {
                    name: t.name.clone(),
                    why: "duplicate tensor name",
                });
            }
        }
        Ok(Self { source, file_len, meta, tensors, by_name })
    }

    /// 元数据访问器（键不存在返回 `None`；类型不匹配返回错误）。
    pub fn metadata(&self) -> &ModelMeta {
        &self.meta
    }

    /// 按名称查找张量描述。
    pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
        self.by_name.get(name).map(|&i| &self.tensors[i as usize])
    }

    /// 全部张量描述（文件顺序）。
    pub fn tensors(&self) -> &[GgufTensor] {
        &self.tensors
    }

    /// 读取张量数据（惰性 pread；返回字节拷贝——解码侧本就需要独立缓冲）。
    pub fn tensor_data(&self, tensor: &GgufTensor) -> Result<Vec<u8>, GgufError> {
        let end = tensor.offset.checked_add(tensor.length).ok_or_else(|| GgufError::Oversized {
            tensor: tensor.name.clone(),
            offset: tensor.offset,
            length: tensor.length,
            limit: self.file_len,
        })?;
        if end > self.file_len {
            return Err(GgufError::Oversized {
                tensor: tensor.name.clone(),
                offset: tensor.offset,
                length: tensor.length,
                limit: self.file_len,
            });
        }
        let mut out = vec![0u8; tensor.length as usize];
        let mut done = 0usize;
        while done < out.len() {
            let got = self.source.read_at(&mut out[done..], tensor.offset + done as u64)?;
            if got == 0 {
                return Err(GgufError::Truncated { at: tensor.offset + done as u64, needed: end });
            }
            done += got;
        }
        Ok(out)
    }
}

/// 模型元数据表（GGUF KV；字典序存储保证确定性迭代）。
#[derive(Debug, Clone, Default)]
pub struct ModelMeta {
    kvs: BTreeMap<String, MetaValue>,
}

impl ModelMeta {
    /// 从 KV 列表构造元数据表（程序化嵌入/测试场景；与解析路径共用同一内部表示）。
    pub fn from_kvs(kvs: impl IntoIterator<Item = (String, MetaValue)>) -> Self {
        Self { kvs: kvs.into_iter().collect() }
    }

    /// 原始值访问（未知键返回 `None`）。
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.kvs.get(key)
    }

    /// 键值迭代（字典序，确定性）。
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MetaValue)> {
        self.kvs.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// 条目数。
    pub fn len(&self) -> usize {
        self.kvs.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.kvs.is_empty()
    }

    /// 字符串值访问：缺键 → `Ok(None)`；类型不匹配 → 错误。
    pub fn meta_str(&self, key: &str) -> Result<Option<&str>, GgufError> {
        typed(&self.kvs, key, "expected string", |v| match v {
            MetaValue::Str(s) => Some(s.as_str()),
            _ => None,
        })
    }

    /// u32 值访问（缺键 → `Ok(None)`；非 u32 → 错误）。
    pub fn meta_u32(&self, key: &str) -> Result<Option<u32>, GgufError> {
        typed(&self.kvs, key, "expected u32", |v| match v {
            MetaValue::U32(x) => Some(*x),
            _ => None,
        })
    }

    /// f32 值访问（缺键 → `Ok(None)`；非 f32 → 错误）。
    pub fn meta_f32(&self, key: &str) -> Result<Option<f32>, GgufError> {
        typed(&self.kvs, key, "expected f32", |v| match v {
            MetaValue::F32(x) => Some(*x),
            _ => None,
        })
    }

    /// 字符串数组访问（`tokenizer.ggml.tokens` / `merges` 等）。
    pub fn meta_array_str(&self, key: &str) -> Result<Option<&[String]>, GgufError> {
        typed(&self.kvs, key, "expected string array", |v| match v {
            MetaValue::Array(ArrayValue::Str(xs)) => Some(xs.as_slice()),
            _ => None,
        })
    }

    /// 字节数组访问（`tokenizer.ggml.model` 等二进制小值）。
    pub fn meta_bytes(&self, key: &str) -> Result<Option<&[u8]>, GgufError> {
        typed(&self.kvs, key, "expected byte array", |v| match v {
            MetaValue::Array(ArrayValue::U8(xs)) => Some(xs.as_slice()),
            _ => None,
        })
    }

    /// f32 二维数组访问（Qwen 系 rope 多维参数；扁平 f32 数组视为单行）。
    pub fn meta_nested_f32(&self, key: &str) -> Result<Option<Vec<Vec<f32>>>, GgufError> {
        match self.kvs.get(key) {
            None => Ok(None),
            Some(MetaValue::Array(ArrayValue::F32(v))) => Ok(Some(vec![v.clone()])),
            Some(MetaValue::Array(ArrayValue::Nested(items))) => {
                let mut rows = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        ArrayValue::F32(v) => rows.push(v.clone()),
                        _ => {
                            return Err(GgufError::InvalidMetadata {
                                key: key.to_string(),
                                why: "expected nested f32 arrays",
                            });
                        }
                    }
                }
                Ok(Some(rows))
            }
            Some(_) => Err(GgufError::InvalidMetadata {
                key: key.to_string(),
                why: "expected f32 array or nested f32 arrays",
            }),
        }
    }
}

/// 通用「缺键 → None / 类型匹配 → Some / 不匹配 → 错误」访问器骨架。
fn typed<'a, T>(
    kvs: &'a BTreeMap<String, MetaValue>,
    key: &str,
    why: &'static str,
    pick: impl FnOnce(&'a MetaValue) -> Option<T>,
) -> Result<Option<T>, GgufError> {
    match kvs.get(key) {
        None => Ok(None),
        Some(v) => pick(v)
            .map(Some)
            .ok_or_else(|| GgufError::InvalidMetadata { key: key.to_string(), why }),
    }
}

/// 可增长的文件前缀缓冲：解析器按需请求字节，块读摊薄系统调用。
struct RegionBuf<'a> {
    source: &'a Source,
    file_len: u64,
    buf: Vec<u8>,
}

impl RegionBuf<'_> {
    /// 保证 `buf` 覆盖 `[0, n)`；文件不足 → `Truncated`。
    fn need(&mut self, n: usize) -> Result<(), GgufError> {
        if n <= self.buf.len() {
            return Ok(());
        }
        let file_len = usize::try_from(self.file_len)
            .map_err(|_| GgufError::Truncated { at: self.file_len, needed: n as u64 })?;
        if n > file_len {
            return Err(GgufError::Truncated { at: self.file_len, needed: n as u64 });
        }
        let want = n.max(self.buf.len() + READ_CHUNK).min(file_len);
        self.read_exact(want)
    }

    /// 精确读到 `want` 字节（中途 EOF → 回退缓冲并报 `Truncated`）。
    fn read_exact(&mut self, want: usize) -> Result<(), GgufError> {
        let have = self.buf.len();
        if want <= have {
            return Ok(());
        }
        self.buf.resize(want, 0);
        let mut done = 0usize;
        while done < want - have {
            let got = self.source.read_at(&mut self.buf[have + done..], (have + done) as u64)?;
            if got == 0 {
                break;
            }
            done += got;
        }
        if done < want - have {
            self.buf.truncate(have + done);
            return Err(GgufError::Truncated { at: (have + done) as u64, needed: want as u64 });
        }
        Ok(())
    }
}

/// 解析头部：魔数/版本/计数（含计数守卫，防恶意文件巨量分配）。
fn parse_header(region: &mut RegionBuf<'_>) -> Result<(u64, u64), GgufError> {
    region.need(HEADER_SIZE)?;
    let buf = &region.buf[..HEADER_SIZE];
    let magic = [buf[0], buf[1], buf[2], buf[3]];
    if &magic != b"GGUF" {
        return Err(GgufError::BadMagic { found: magic });
    }
    let version = le_u32_buf(buf, 4);
    if !matches!(version, 2 | 3) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let n_tensors = le_u64_buf(buf, 8);
    let n_kv = le_u64_buf(buf, 16);
    // 计数守卫：每条 KV ≥ 13 字节、每个张量记录 ≥ 32 字节（fail-closed，防 OOM）
    if n_kv > region.file_len / 13 {
        return Err(GgufError::Malformed { what: "metadata kv count implausible", at: 16 });
    }
    if n_tensors > region.file_len / 32 {
        return Err(GgufError::Malformed { what: "tensor count implausible", at: 8 });
    }
    Ok((n_tensors, n_kv))
}

/// 解析元数据 KV 表，返回表与结束游标（张量表紧接着元数据区，无填充）。
fn parse_metadata(region: &mut RegionBuf<'_>, n_kv: u64) -> Result<(ModelMeta, usize), GgufError> {
    let mut meta = ModelMeta::default();
    let mut pos = HEADER_SIZE;
    for _ in 0..n_kv {
        let key = parse_string(region, &mut pos, "metadata key")?;
        let code = le_u32(region, &mut pos)?;
        let value = parse_value(region, &mut pos, code)?;
        meta.kvs.insert(key, value);
    }
    Ok((meta, pos))
}

/// 解析单个元数据值（按类型码分发；数组递归）。
fn parse_value(
    region: &mut RegionBuf<'_>,
    pos: &mut usize,
    code: u32,
) -> Result<MetaValue, GgufError> {
    Ok(match code {
        type_code::U8 => MetaValue::U8(le_u8(region, pos)?),
        type_code::I8 => MetaValue::I8(le_u8(region, pos)? as i8),
        type_code::U16 => MetaValue::U16(le_u16(region, pos)?),
        type_code::I16 => MetaValue::I16(le_u16(region, pos)? as i16),
        type_code::U32 => MetaValue::U32(le_u32(region, pos)?),
        type_code::I32 => MetaValue::I32(le_u32(region, pos)? as i32),
        type_code::F32 => MetaValue::F32(le_f32(region, pos)?),
        type_code::BOOL => MetaValue::Bool(le_u8(region, pos)? != 0),
        type_code::STR => MetaValue::Str(parse_string(region, pos, "metadata string")?),
        type_code::ARRAY => MetaValue::Array(parse_array(region, pos, 0)?),
        type_code::U64 => MetaValue::U64(le_u64(region, pos)?),
        type_code::I64 => MetaValue::I64(le_u64(region, pos)? as i64),
        type_code::F64 => MetaValue::F64(le_f64(region, pos)?),
        _ => {
            return Err(GgufError::Malformed {
                what: "unknown metadata value type",
                at: *pos as u64 - 4,
            });
        }
    })
}

/// 解析数组值（元素类型统一，可嵌套；`depth` 防栈溢出）。
///
/// 数组元素在文件中不带各自类型标签（统一由 `elem_code` 决定），
/// 故各分支直接按类型逐个读取，类型一致性由结构保证。
fn parse_array(
    region: &mut RegionBuf<'_>,
    pos: &mut usize,
    depth: u32,
) -> Result<ArrayValue, GgufError> {
    if depth >= MAX_ARRAY_DEPTH {
        return Err(GgufError::Malformed { what: "array nesting too deep", at: *pos as u64 });
    }
    let elem_code = le_u32(region, pos)?;
    let count = le_u64(region, pos)?;
    // 计数守卫：每个元素至少 1 字节
    if count > region.file_len {
        return Err(GgufError::Malformed { what: "array count implausible", at: *pos as u64 - 8 });
    }
    let count = count as usize;
    Ok(match elem_code {
        type_code::U8 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u8(region, pos)?);
            }
            ArrayValue::U8(v)
        }
        type_code::I8 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u8(region, pos)? as i8);
            }
            ArrayValue::I8(v)
        }
        type_code::U16 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u16(region, pos)?);
            }
            ArrayValue::U16(v)
        }
        type_code::I16 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u16(region, pos)? as i16);
            }
            ArrayValue::I16(v)
        }
        type_code::U32 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u32(region, pos)?);
            }
            ArrayValue::U32(v)
        }
        type_code::I32 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u32(region, pos)? as i32);
            }
            ArrayValue::I32(v)
        }
        type_code::F32 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_f32(region, pos)?);
            }
            ArrayValue::F32(v)
        }
        type_code::BOOL => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u8(region, pos)? != 0);
            }
            ArrayValue::Bool(v)
        }
        type_code::STR => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(parse_string(region, pos, "array string item")?);
            }
            ArrayValue::Str(v)
        }
        type_code::U64 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u64(region, pos)?);
            }
            ArrayValue::U64(v)
        }
        type_code::I64 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_u64(region, pos)? as i64);
            }
            ArrayValue::I64(v)
        }
        type_code::F64 => {
            let mut v = Vec::with_capacity(count);
            for _ in 0..count {
                v.push(le_f64(region, pos)?);
            }
            ArrayValue::F64(v)
        }
        type_code::ARRAY => {
            let mut nested = Vec::with_capacity(count);
            for _ in 0..count {
                nested.push(parse_array(region, pos, depth + 1)?);
            }
            ArrayValue::Nested(nested)
        }
        _ => {
            return Err(GgufError::Malformed {
                what: "unknown array element type",
                at: *pos as u64 - 4,
            });
        }
    })
}

/// 解析张量表（名称/维度/类型/偏移），并校验数据区间。
fn parse_tensors(
    region: &mut RegionBuf<'_>,
    n_tensors: u64,
    pos: &mut usize,
) -> Result<Vec<GgufTensor>, GgufError> {
    let mut tensors: Vec<GgufTensor> = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = parse_string(region, pos, "tensor name")?;
        if name.is_empty() {
            return Err(GgufError::Malformed { what: "empty tensor name", at: *pos as u64 });
        }
        let n_dims = le_u32(region, pos)?;
        if n_dims == 0 {
            return Err(GgufError::InvalidTensor { name, why: "zero dimensions" });
        }
        if n_dims > 8 {
            return Err(GgufError::InvalidTensor { name, why: "too many dimensions" });
        }
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(le_u64(region, pos)?);
        }
        let dtype = GgufDtype::from_code(le_u32(region, pos)?);
        let offset = le_u64(region, pos)?;
        tensors.push(GgufTensor { name, shape, dtype, offset, length: 0 });
    }
    finalize_tensor_lengths(&mut tensors, region.file_len)?;
    Ok(tensors)
}

/// 填张量长度并校验数据区间：已知类型按形状推导（越界/重叠 → 错误）；
/// 未知类型取与下一张量（或文件尾）的间距。
fn finalize_tensor_lengths(tensors: &mut [GgufTensor], file_len: u64) -> Result<(), GgufError> {
    for i in 0..tensors.len() {
        let (name, dtype, offset) = {
            let t = &tensors[i];
            (t.name.clone(), t.dtype, t.offset)
        };
        if offset % GGUF_ALIGNMENT != 0 {
            return Err(GgufError::Malformed {
                what: "tensor data not 32-byte aligned",
                at: offset,
            });
        }
        if offset > file_len {
            return Err(GgufError::Oversized { tensor: name, offset, length: 0, limit: file_len });
        }
        let elems = tensors[i].element_count().ok_or_else(|| GgufError::InvalidTensor {
            name: name.clone(),
            why: "shape element count overflow",
        })?;
        if dtype.size_known() {
            let len = dtype.size_bytes(elems).ok_or_else(|| GgufError::InvalidTensor {
                name: name.clone(),
                why: "element count not compatible with dtype block size",
            })?;
            let end = offset.checked_add(len).ok_or_else(|| GgufError::Oversized {
                tensor: name.clone(),
                offset,
                length: len,
                limit: file_len,
            })?;
            let limit = tensors.get(i + 1).map_or(file_len, |n| n.offset);
            if end > limit {
                return Err(GgufError::Oversized { tensor: name, offset, length: len, limit });
            }
            tensors[i].length = len;
        } else {
            // 未知类型：长度 = 与下一张量（或文件尾）的间距
            let next = tensors.get(i + 1).map_or(file_len, |n| n.offset);
            let len = next.checked_sub(offset).ok_or_else(|| GgufError::Oversized {
                tensor: name.clone(),
                offset,
                length: 0,
                limit: next,
            })?;
            tensors[i].length = len;
        }
    }
    Ok(())
}

// ===================== 低层小端读取（调用方已保证边界） =====================

#[inline]
fn le_u8(region: &mut RegionBuf<'_>, pos: &mut usize) -> Result<u8, GgufError> {
    region.need(*pos + 1)?;
    let v = region.buf[*pos];
    *pos += 1;
    Ok(v)
}

#[inline]
fn le_u16(region: &mut RegionBuf<'_>, pos: &mut usize) -> Result<u16, GgufError> {
    region.need(*pos + 2)?;
    let v = u16::from_le_bytes([region.buf[*pos], region.buf[*pos + 1]]);
    *pos += 2;
    Ok(v)
}

#[inline]
fn le_u32(region: &mut RegionBuf<'_>, pos: &mut usize) -> Result<u32, GgufError> {
    region.need(*pos + 4)?;
    let bytes: [u8; 4] = region.buf[*pos..*pos + 4].try_into().expect("边界已由 need 保证");
    *pos += 4;
    Ok(u32::from_le_bytes(bytes))
}

#[inline]
fn le_f32(region: &mut RegionBuf<'_>, pos: &mut usize) -> Result<f32, GgufError> {
    Ok(f32::from_bits(le_u32(region, pos)?))
}

#[inline]
fn le_u64(region: &mut RegionBuf<'_>, pos: &mut usize) -> Result<u64, GgufError> {
    region.need(*pos + 8)?;
    let bytes: [u8; 8] = region.buf[*pos..*pos + 8].try_into().expect("边界已由 need 保证");
    *pos += 8;
    Ok(u64::from_le_bytes(bytes))
}

#[inline]
fn le_f64(region: &mut RegionBuf<'_>, pos: &mut usize) -> Result<f64, GgufError> {
    Ok(f64::from_bits(le_u64(region, pos)?))
}

#[inline]
fn le_u32_buf(buf: &[u8], at: usize) -> u32 {
    let bytes: [u8; 4] = buf[at..at + 4].try_into().expect("调用方已保证边界");
    u32::from_le_bytes(bytes)
}

#[inline]
fn le_u64_buf(buf: &[u8], at: usize) -> u64 {
    let bytes: [u8; 8] = buf[at..at + 8].try_into().expect("调用方已保证边界");
    u64::from_le_bytes(bytes)
}

/// 解析字符串（u64 长度 + UTF-8）。
fn parse_string(
    region: &mut RegionBuf<'_>,
    pos: &mut usize,
    what: &'static str,
) -> Result<String, GgufError> {
    let len = le_u64(region, pos)?;
    let len =
        usize::try_from(len).map_err(|_| GgufError::Malformed { what, at: *pos as u64 - 8 })?;
    region.need(*pos + len)?;
    let bytes = &region.buf[*pos..*pos + len];
    let s =
        std::str::from_utf8(bytes).map_err(|_| GgufError::Malformed { what, at: *pos as u64 })?;
    *pos += len;
    Ok(s.to_string())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

    use super::*;
    use crate::fixture::{FixtureTensor, build_gguf};
    use crate::schema::{ArrayValue, GgufDtype, MetaValue};

    /// 构造一个 Qwen2 形状的元数据 fixture（键为 llama.cpp 格式常量；
    /// `general.name` 用虚构值——013 模型标识零硬编码铁律）。
    fn qwen2_shaped_kvs() -> Vec<(&'static str, MetaValue)> {
        vec![
            ("general.architecture", MetaValue::Str("qwen2".into())),
            ("general.name", MetaValue::Str("fixture-qwen2-shaped".into())),
            ("qwen2.block_count", MetaValue::U32(28)),
            ("qwen2.embedding_length", MetaValue::U32(896)),
            ("qwen2.attention.head_count", MetaValue::U32(14)),
            ("qwen2.attention.head_count_kv", MetaValue::U32(2)),
            ("qwen2.attention.layer_norm_rms_epsilon", MetaValue::F32(1e-6)),
            ("qwen2.attention.rope_freq_base", MetaValue::F32(1_000_000.0)),
            ("qwen2.rope.dimension_count", MetaValue::U32(64)),
            ("tokenizer.ggml.model", MetaValue::Str("gpt2".into())),
            (
                "tokenizer.ggml.tokens",
                MetaValue::Array(ArrayValue::Str(vec![
                    "<unk>".into(),
                    "<s>".into(),
                    "hello".into(),
                    "世界".into(),
                    "🙂".into(),
                ])),
            ),
            ("tokenizer.ggml.token_type", MetaValue::Array(ArrayValue::U32(vec![2, 3, 1, 1, 1]))),
            ("tokenizer.ggml.byte_probe", MetaValue::Array(ArrayValue::U8(vec![1, 2, 3]))),
            ("tokenizer.ggml.add_bos_token", MetaValue::Bool(true)),
            (
                "qwen2.attention.rope_scaling",
                MetaValue::Array(ArrayValue::Nested(vec![
                    ArrayValue::F32(vec![1.0, 2.0]),
                    ArrayValue::F32(vec![3.0]),
                ])),
            ),
        ]
    }

    #[test]
    fn golden_bytes_match_spec_layout() {
        // 手工拼装的规范字节布局（独立于写入器实现），逐字节对拍 —— golden 基准。
        // 布局推演：header 24；KV("a",U32(7)) = 8+1 + 4 + 4 = 17 → 41；
        // 张量表 ("w",[2],F32) = 8+1 + 4 + 8 + 4 + 8 = 33 → 74；
        // 数据区 = align32(74) = 96；数据 2×f32 = 8 → 总 104 字节。
        let mut expect: Vec<u8> = Vec::new();
        expect.extend_from_slice(b"GGUF");
        expect.extend_from_slice(&3u32.to_le_bytes());
        expect.extend_from_slice(&1u64.to_le_bytes()); // n_tensors
        expect.extend_from_slice(&1u64.to_le_bytes()); // n_kv
        expect.extend_from_slice(&1u64.to_le_bytes()); // 键长
        expect.push(b'a');
        expect.extend_from_slice(&4u32.to_le_bytes()); // U32
        expect.extend_from_slice(&7u32.to_le_bytes());
        expect.extend_from_slice(&1u64.to_le_bytes()); // 张量名长
        expect.push(b'w');
        expect.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        expect.extend_from_slice(&2u64.to_le_bytes()); // dims[0]
        expect.extend_from_slice(&0u32.to_le_bytes()); // F32
        expect.extend_from_slice(&96u64.to_le_bytes()); // offset（数据区起点）
        expect.resize(96, 0); // 74 → 96 的 22 字节填充
        expect.extend_from_slice(&1.0f32.to_le_bytes());
        expect.extend_from_slice(&2.0f32.to_le_bytes());
        expect.resize(128, 0); // 数据区尾部同样按 32 对齐（写入器契约）
        assert_eq!(expect.len(), 128);

        let kvs = [("a", MetaValue::U32(7))];
        let tensor = FixtureTensor {
            name: "w".into(),
            shape: vec![2],
            dtype: GgufDtype::F32,
            data: [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
        };
        let got = build_gguf(3, &kvs, &[tensor]);
        assert_eq!(got, expect, "写入器字节必须与规范布局逐字节一致");

        // golden 文件必须能读回
        let reader = GgufReader::from_bytes(got).expect("golden 文件必须可解析");
        assert_eq!(reader.metadata().meta_u32("a").unwrap(), Some(7));
        let t = reader.tensor("w").expect("tensor 存在");
        assert_eq!(t.offset, 96);
        assert_eq!(t.length, 8);
        assert!(t.offset_aligned());
    }

    #[test]
    fn qwen2_shape_probe() {
        // 014 T1「Qwen2 元数据探针」：qwen2 形状的元数据全访问器验证
        let kvs = qwen2_shaped_kvs();
        let reader = GgufReader::from_bytes(build_gguf(3, &kvs, &[])).expect("可解析");
        let meta = reader.metadata();
        assert_eq!(meta.meta_str("general.architecture").unwrap(), Some("qwen2"));
        assert_eq!(meta.meta_u32("qwen2.block_count").unwrap(), Some(28));
        assert_eq!(meta.meta_u32("qwen2.attention.head_count").unwrap(), Some(14));
        assert_eq!(meta.meta_u32("qwen2.attention.head_count_kv").unwrap(), Some(2));
        assert_eq!(meta.meta_f32("qwen2.attention.layer_norm_rms_epsilon").unwrap(), Some(1e-6));
        assert_eq!(meta.meta_f32("qwen2.attention.rope_freq_base").unwrap(), Some(1_000_000.0));
        assert_eq!(meta.meta_u32("qwen2.rope.dimension_count").unwrap(), Some(64));
        let tokens = meta.meta_array_str("tokenizer.ggml.tokens").unwrap().expect("tokens");
        assert_eq!(tokens, &["<unk>", "<s>", "hello", "世界", "🙂"]);
        assert_eq!(meta.meta_bytes("tokenizer.ggml.byte_probe").unwrap(), Some(&[1u8, 2, 3][..]));
        assert_eq!(
            meta.meta_nested_f32("qwen2.attention.rope_scaling").unwrap(),
            Some(vec![vec![1.0, 2.0], vec![3.0]])
        );
        // 缺键 → None
        assert_eq!(meta.meta_u32("qwen2.attention.key_length").unwrap(), None);
    }

    #[test]
    fn accessor_type_mismatch_errors() {
        let kvs = [("k", MetaValue::Str("v".into()))];
        let reader = GgufReader::from_bytes(build_gguf(3, &kvs, &[])).expect("可解析");
        let err = reader.metadata().meta_u32("k").unwrap_err();
        assert!(matches!(err, GgufError::InvalidMetadata { .. }));
        assert!(err.to_string().contains("key k"));
    }

    #[test]
    fn all_meta_value_types_roundtrip() {
        let kvs = [
            ("u8", MetaValue::U8(1)),
            ("i8", MetaValue::I8(-2)),
            ("u16", MetaValue::U16(3)),
            ("i16", MetaValue::I16(-4)),
            ("u32", MetaValue::U32(5)),
            ("i32", MetaValue::I32(-6)),
            ("f32", MetaValue::F32(0.5)),
            ("bool-true", MetaValue::Bool(true)),
            ("bool-false", MetaValue::Bool(false)),
            ("str", MetaValue::Str("你好, world!".into())),
            ("u64", MetaValue::U64(u64::MAX)),
            ("i64", MetaValue::I64(i64::MIN)),
            ("f64", MetaValue::F64(1e300)),
            ("arr-u8", MetaValue::Array(ArrayValue::U8(vec![1, 2, 3]))),
            ("arr-i8", MetaValue::Array(ArrayValue::I8(vec![-1, 2]))),
            ("arr-u16", MetaValue::Array(ArrayValue::U16(vec![1, 2]))),
            ("arr-i16", MetaValue::Array(ArrayValue::I16(vec![-1, 2]))),
            ("arr-u32", MetaValue::Array(ArrayValue::U32(vec![1, 2, 3]))),
            ("arr-i32", MetaValue::Array(ArrayValue::I32(vec![-1, 2]))),
            ("arr-f32", MetaValue::Array(ArrayValue::F32(vec![0.25, 0.5]))),
            ("arr-bool", MetaValue::Array(ArrayValue::Bool(vec![true, false]))),
            ("arr-str", MetaValue::Array(ArrayValue::Str(vec!["a".into(), "b".into()]))),
            ("arr-u64", MetaValue::Array(ArrayValue::U64(vec![1, 2]))),
            ("arr-i64", MetaValue::Array(ArrayValue::I64(vec![-1, 2]))),
            ("arr-f64", MetaValue::Array(ArrayValue::F64(vec![1.5, 2.5]))),
            (
                "arr-nested",
                MetaValue::Array(ArrayValue::Nested(vec![
                    ArrayValue::U32(vec![1, 2]),
                    ArrayValue::Str(vec!["x".into()]),
                ])),
            ),
        ];
        let reader = GgufReader::from_bytes(build_gguf(3, &kvs, &[])).expect("可解析");
        let meta = reader.metadata();
        assert_eq!(meta.len(), kvs.len());
        for (k, v) in &kvs {
            assert_eq!(meta.get(k), Some(v), "key {k}");
        }
    }

    #[test]
    fn tensor_table_and_data() {
        // 数据长度必须与 shape×dtype 自洽（读回侧按形状推导长度并校验数据区）
        let f32_data: Vec<u8> = (0..12).map(|i| (i * 7) as u8).collect(); // 3 元素 × 4B
        let q8_data: Vec<u8> = (0..34).map(|i| (i * 3 + 1) as u8).collect(); // 1 块（32 元素）× 34B
        let tensors = vec![
            FixtureTensor {
                name: "token_embd.weight".into(),
                shape: vec![3, 1],
                dtype: GgufDtype::F32,
                data: f32_data.clone(),
            },
            FixtureTensor {
                name: "output_norm.weight".into(),
                shape: vec![32],
                dtype: GgufDtype::Q8_0,
                data: q8_data.clone(),
            },
        ];
        let reader = GgufReader::from_bytes(build_gguf(3, &[], &tensors)).expect("可解析");
        assert_eq!(reader.tensors().len(), 2);
        let a = reader.tensor("token_embd.weight").expect("按名查找");
        assert_eq!(a.shape, vec![3, 1]);
        assert_eq!(a.dtype, GgufDtype::F32);
        assert_eq!(a.length, 12); // 3×1×4
        assert!(a.offset_aligned());
        assert_eq!(reader.tensor_data(a).unwrap(), f32_data);
        let b = reader.tensor("output_norm.weight").expect("按名查找");
        assert_eq!(b.dtype, GgufDtype::Q8_0);
        assert_eq!(b.length, 34); // 1 块 × 34B
        assert_eq!(reader.tensor_data(b).unwrap(), q8_data);
        assert_eq!(reader.tensor("no_such_tensor"), None);
    }

    #[test]
    fn unknown_dtype_length_from_gap() {
        // Other(4)（历史废弃码）大小未知：长度 = 与下一张量（或文件尾）的间距
        let tensors = vec![
            FixtureTensor {
                name: "odd.weight".into(),
                shape: vec![96],
                dtype: GgufDtype::Other(4),
                data: vec![0xAA; 96],
            },
            FixtureTensor {
                name: "tail.weight".into(),
                shape: vec![4],
                dtype: GgufDtype::F32,
                data: vec![0xBB; 16],
            },
        ];
        let reader = GgufReader::from_bytes(build_gguf(3, &[], &tensors)).expect("可解析");
        let odd = reader.tensor("odd.weight").expect("存在");
        assert_eq!(odd.dtype, GgufDtype::Other(4));
        // 96 字节已对齐（96 % 32 == 0）→ 间距恰为 96
        assert_eq!(odd.length, 96);
        assert_eq!(reader.tensor_data(odd).unwrap(), vec![0xAA; 96]);
        // 已知类型张量长度仍按形状推导
        let tail = reader.tensor("tail.weight").expect("存在");
        assert_eq!(tail.length, 16);
    }

    #[test]
    fn error_bad_magic() {
        let mut bytes = build_gguf(3, &[], &[]);
        bytes[0] = b'X';
        assert!(matches!(GgufReader::from_bytes(bytes), Err(GgufError::BadMagic { .. })));
    }

    #[test]
    fn error_unsupported_version() {
        for v in [1u32, 4, 100] {
            let bytes = build_gguf(v, &[], &[]);
            assert!(
                matches!(GgufReader::from_bytes(bytes), Err(GgufError::UnsupportedVersion(x)) if x == v),
                "version {v}"
            );
        }
    }

    #[test]
    fn error_truncated_at_every_prefix() {
        // 头部/元数据/张量表/数据区内任意截断都必须 Err（绝不 panic）。
        // 数据区尾部对齐填充允许截掉（读取器不读填充，属合法文件）。
        let tensor = FixtureTensor {
            name: "w".into(),
            shape: vec![2],
            dtype: GgufDtype::F32,
            data: [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
        };
        let full = build_gguf(3, &qwen2_shaped_kvs(), &[tensor]);
        let data_start = full.len() - 32; // 8 字节数据 + 尾部 24 字节对齐 = 数据区 32 字节
        for cut in [0, 1, 4, 23, 24, 25, 100, data_start - 1, data_start, data_start + 7] {
            let mut bytes = full.clone();
            bytes.truncate(cut);
            assert!(GgufReader::from_bytes(bytes).is_err(), "截断到 {cut} 必须报错");
        }
        // 恰好截到最后一个张量末尾：合法文件（数据完整）
        let mut bytes = full.clone();
        bytes.truncate(data_start + 8);
        assert!(GgufReader::from_bytes(bytes).is_ok(), "截到张量末尾应合法");
    }

    #[test]
    fn error_malformed_structures() {
        // 布局（带张量 "w"）：header 24 + kv("a" u32) 17 + 张量名 9 → 名长 41..49、'w' 49、
        // n_dims 50..54、dims 54..62、dtype 62..66、offset 66..74、pad → 96、data 96..104、pad → 128
        let tensor = FixtureTensor {
            name: "w".into(),
            shape: vec![2],
            dtype: GgufDtype::F32,
            data: [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat(),
        };
        let with_tensor =
            || build_gguf(3, &[("a", MetaValue::U32(7))], std::slice::from_ref(&tensor));

        // 坏 utf8 键（'a' 在偏移 32）
        let mut bytes = with_tensor();
        bytes[32] = 0xFF;
        assert!(matches!(GgufReader::from_bytes(bytes), Err(GgufError::Malformed { .. })));

        // n_dims = 0（偏移 50）
        let mut bytes = with_tensor();
        bytes[50..54].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(GgufReader::from_bytes(bytes), Err(GgufError::InvalidTensor { .. })));

        // 空张量名（偏移 41，u64 长度 = 0）
        let mut bytes = with_tensor();
        bytes[41..49].copy_from_slice(&0u64.to_le_bytes());
        assert!(matches!(GgufReader::from_bytes(bytes), Err(GgufError::Malformed { .. })));

        // 未对齐偏移：offset 字段在 66..74，改为 95（95 % 32 != 0）
        let mut bytes = with_tensor();
        bytes[66..74].copy_from_slice(&95u64.to_le_bytes());
        assert!(matches!(GgufReader::from_bytes(bytes), Err(GgufError::Malformed { .. })));

        // 越界偏移：128（对齐，但 128+8 超出文件尾 128）
        let mut bytes = with_tensor();
        bytes[66..74].copy_from_slice(&128u64.to_le_bytes());
        assert!(matches!(GgufReader::from_bytes(bytes), Err(GgufError::Oversized { .. })));
    }

    #[test]
    fn error_implausible_counts() {
        // n_kv 被改到 2^40 → 计数守卫
        let mut bytes = build_gguf(3, &[], &[]);
        bytes[16..24].copy_from_slice(&(1u64 << 40).to_le_bytes());
        assert!(matches!(GgufReader::from_bytes(bytes), Err(GgufError::Malformed { .. })));
    }

    #[test]
    fn error_duplicate_tensor_names() {
        let tensors = vec![
            FixtureTensor {
                name: "w".into(),
                shape: vec![4],
                dtype: GgufDtype::F32,
                data: vec![0; 16],
            },
            FixtureTensor {
                name: "w".into(),
                shape: vec![4],
                dtype: GgufDtype::F32,
                data: vec![0; 16],
            },
        ];
        assert!(matches!(
            GgufReader::from_bytes(build_gguf(3, &[], &tensors)),
            Err(GgufError::InvalidTensor { .. })
        ));
    }

    #[test]
    fn error_quant_block_mismatch() {
        // Q8_0 元素数不是 32 的倍数 → 明确错误
        let tensors = vec![FixtureTensor {
            name: "bad.weight".into(),
            shape: vec![10],
            dtype: GgufDtype::Q8_0,
            data: vec![0; 34],
        }];
        assert!(matches!(
            GgufReader::from_bytes(build_gguf(3, &[], &tensors)),
            Err(GgufError::InvalidTensor { .. })
        ));
    }

    #[test]
    fn version_2_accepted() {
        let reader = GgufReader::from_bytes(build_gguf(2, &[("a", MetaValue::U32(7))], &[]));
        assert_eq!(reader.expect("v2 可解析").metadata().meta_u32("a").unwrap(), Some(7));
    }

    #[test]
    fn from_bytes_empty_is_error() {
        assert!(GgufReader::from_bytes(Vec::new()).is_err());
    }

    #[test]
    fn meta_iter_is_deterministic() {
        let kvs = [("z", MetaValue::U32(1)), ("a", MetaValue::U32(2)), ("m", MetaValue::U32(3))];
        let reader = GgufReader::from_bytes(build_gguf(3, &kvs, &[])).expect("可解析");
        let keys: Vec<&str> = reader.metadata().iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["a", "m", "z"]);
    }
}

#[cfg(test)]
mod proptests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

    use super::*;
    use crate::fixture::{FixtureTensor, build_gguf};
    use proptest::prelude::*;

    fn any_string() -> impl Strategy<Value = String> {
        prop::collection::vec(any::<char>(), 0..24).prop_map(|cs| cs.into_iter().collect())
    }

    fn any_scalar() -> impl Strategy<Value = MetaValue> {
        prop_oneof![
            any::<u8>().prop_map(MetaValue::U8),
            any::<i8>().prop_map(MetaValue::I8),
            any::<u16>().prop_map(MetaValue::U16),
            any::<i16>().prop_map(MetaValue::I16),
            any::<u32>().prop_map(MetaValue::U32),
            any::<i32>().prop_map(MetaValue::I32),
            any::<f32>().prop_map(MetaValue::F32),
            any::<bool>().prop_map(MetaValue::Bool),
            any_string().prop_map(MetaValue::Str),
            any::<u64>().prop_map(MetaValue::U64),
            any::<i64>().prop_map(MetaValue::I64),
            any::<f64>().prop_map(MetaValue::F64),
        ]
    }

    fn any_flat_array() -> impl Strategy<Value = ArrayValue> {
        prop_oneof![
            prop::collection::vec(any::<u8>(), 0..8).prop_map(ArrayValue::U8),
            prop::collection::vec(any::<i8>(), 0..8).prop_map(ArrayValue::I8),
            prop::collection::vec(any::<u16>(), 0..8).prop_map(ArrayValue::U16),
            prop::collection::vec(any::<i16>(), 0..8).prop_map(ArrayValue::I16),
            prop::collection::vec(any::<u32>(), 0..8).prop_map(ArrayValue::U32),
            prop::collection::vec(any::<i32>(), 0..8).prop_map(ArrayValue::I32),
            prop::collection::vec(any::<f32>(), 0..8).prop_map(ArrayValue::F32),
            prop::collection::vec(any::<bool>(), 0..8).prop_map(ArrayValue::Bool),
            prop::collection::vec(any_string(), 0..8).prop_map(ArrayValue::Str),
            prop::collection::vec(any::<u64>(), 0..8).prop_map(ArrayValue::U64),
            prop::collection::vec(any::<i64>(), 0..8).prop_map(ArrayValue::I64),
            prop::collection::vec(any::<f64>(), 0..8).prop_map(ArrayValue::F64),
        ]
    }

    fn any_value(depth: usize) -> impl Strategy<Value = MetaValue> {
        let flat =
            prop_oneof![any_scalar().boxed(), any_flat_array().prop_map(MetaValue::Array).boxed()];
        if depth >= 2 {
            flat.boxed()
        } else {
            prop_oneof![
                flat,
                prop::collection::vec(any_flat_array(), 0..4)
                    .prop_map(|items| MetaValue::Array(ArrayValue::Nested(items)))
                    .boxed(),
            ]
            .boxed()
        }
    }

    /// f32/f64 按位比较（NaN 也是合法值，位相等即值相等）。
    fn value_eq(a: &MetaValue, b: &MetaValue) -> bool {
        match (a, b) {
            (MetaValue::F32(x), MetaValue::F32(y)) => x.to_bits() == y.to_bits(),
            (MetaValue::F64(x), MetaValue::F64(y)) => x.to_bits() == y.to_bits(),
            (MetaValue::Array(x), MetaValue::Array(y)) => array_eq(x, y),
            _ => a == b,
        }
    }

    fn array_eq(a: &ArrayValue, b: &ArrayValue) -> bool {
        match (a, b) {
            (ArrayValue::F32(x), ArrayValue::F32(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (ArrayValue::F64(x), ArrayValue::F64(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (ArrayValue::Nested(x), ArrayValue::Nested(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(x, y)| array_eq(x, y))
            }
            _ => a == b,
        }
    }

    proptest! {
        /// 随机元数据 → 写入 → 读回 → 逐键值相等。
        #[test]
        fn metadata_roundtrip(kvs in prop::collection::vec((any_string(), any_value(0)), 0..8)) {
            // 重复键（随机字符串会撞键）：写入器照写全部条目，读取器 BTreeMap 后写覆盖。
            // 期望模型与读取器同构：逐条 insert，后写覆盖。
            let mut expect = std::collections::BTreeMap::new();
            for (k, v) in &kvs {
                expect.insert(k.as_str(), v.clone());
            }
            let kvs: Vec<(&str, MetaValue)> =
                expect.iter().map(|(k, v)| (*k, v.clone())).collect();
            let reader = GgufReader::from_bytes(build_gguf(3, &kvs, &[])).expect("fixture 必须可解析");
            for (k, v) in &expect {
                let got = reader.metadata().get(k).unwrap_or_else(|| panic!("键 {k:?} 缺失"));
                prop_assert!(value_eq(got, v), "kv {k:?} 不一致");
            }
            prop_assert_eq!(reader.metadata().len(), expect.len(), "重复键只保留一条");
        }

        /// 随机张量表 → 写入 → 读回：形状/类型/偏移对齐/数据逐字节一致。
        #[test]
        fn tensor_roundtrip(specs in prop::collection::vec(any_tensor_spec(), 1..4)) {
            let tensors: Vec<FixtureTensor> = specs
                .iter()
                .enumerate()
                .map(|(i, (shape, dtype, data))| FixtureTensor {
                    name: format!("t{i}"),
                    shape: shape.clone(),
                    dtype: *dtype,
                    data: data.clone(),
                })
                .collect();
            let reader = GgufReader::from_bytes(build_gguf(3, &[], &tensors)).expect("fixture 必须可解析");
            assert_eq!(reader.tensors().len(), tensors.len());
            for (i, t) in reader.tensors().iter().enumerate() {
                let expect = &tensors[i];
                prop_assert_eq!(&t.name, &expect.name);
                prop_assert_eq!(&t.shape, &expect.shape);
                prop_assert_eq!(t.dtype, expect.dtype);
                prop_assert!(t.offset % GGUF_ALIGNMENT == 0, "offset 未对齐: {}", t.offset);
                let data = reader.tensor_data(t).unwrap();
                prop_assert_eq!(data.as_slice(), expect.data.as_slice());
                prop_assert_eq!(reader.tensor(&expect.name), Some(t));
            }
        }

        /// 任意字节（注入合法魔数/版本后）只允许 Ok/Err，绝不 panic。
        #[test]
        fn arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
            let mut b = bytes;
            if b.len() >= 4 {
                b[0..4].copy_from_slice(b"GGUF");
            }
            if b.len() >= 8 {
                b[4..8].copy_from_slice(&3u32.to_le_bytes());
            }
            let _ = GgufReader::from_bytes(b);
        }
    }

    fn any_tensor_spec() -> impl Strategy<Value = (Vec<u64>, GgufDtype, Vec<u8>)> {
        prop_oneof![
            // F32：字节数 = 4 × 元素数
            prop::collection::vec(0u64..9, 1..4).prop_flat_map(|shape| {
                let elems = shape.iter().product::<u64>() as usize;
                // flat_map 闭包是 Fn（多次调用），内层 move 闭包须捕获局部所有权值，体内只克隆不消费
                let owned_shape = shape.clone();
                prop::collection::vec(any::<u8>(), 4 * elems)
                    .prop_map(move |data| (owned_shape.clone(), GgufDtype::F32, data))
            }),
            // F16：字节数 = 2 × 元素数
            prop::collection::vec(0u64..9, 1..4).prop_flat_map(|shape| {
                let elems = shape.iter().product::<u64>() as usize;
                let owned_shape = shape.clone();
                prop::collection::vec(any::<u8>(), 2 * elems)
                    .prop_map(move |data| (owned_shape.clone(), GgufDtype::F16, data))
            }),
            // Q8_0：元素数为 32 的倍数，字节数 = 34 × 块数
            prop::collection::vec(prop::sample::select(&[0u64, 32, 64, 96]), 1..3).prop_flat_map(
                |shape| {
                    let blocks = shape.iter().product::<u64>() as usize / 32;
                    let owned_shape = shape.clone();
                    prop::collection::vec(any::<u8>(), 34 * blocks)
                        .prop_map(move |data| (owned_shape.clone(), GgufDtype::Q8_0, data))
                }
            ),
        ]
    }
}
