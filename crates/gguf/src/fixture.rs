//! 测试用 GGUF fixture 生成器（仅测试编译）。
//!
//! 职责：按规范字节布局自生成最小合法 GGUF（golden 字节基准 + proptest 输入）。
//! 真实模型文件不进测试（013 模型标识零硬编码铁律）；与 llama.cpp 真机对拍属 014 T10。
//! 布局事实与 `reader` 模块一致：header 24 → 元数据 KV → 张量表 → 数据区（32 对齐）。

use crate::schema::type_code;
use crate::schema::{ArrayValue, GGUF_ALIGNMENT, GgufDtype, HEADER_SIZE, MetaValue};

/// 待写入的张量（数据长度必须与 shape×dtype 一致；写入器不校验，读回侧兜底）。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FixtureTensor {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgufDtype,
    pub data: Vec<u8>,
}

/// 生成完整 GGUF 字节（确定性：同输入同输出）。
pub(crate) fn build_gguf(
    version: u32,
    kvs: &[(&str, MetaValue)],
    tensors: &[FixtureTensor],
) -> Vec<u8> {
    // 数据区起点可预先算出（元数据区/张量表长度与内容无关地确定）：
    // header 24 + KV 区 + 张量表区 → 对齐 32
    let mut meta_len = 0usize;
    for (k, v) in kvs {
        meta_len += str_len(k) + 4 + value_len(v);
    }
    let mut info_len = 0usize;
    for t in tensors {
        info_len += str_len(&t.name) + 4 + 8 * t.shape.len() + 4 + 8;
    }
    let data_start = align32(HEADER_SIZE + meta_len + info_len);

    let mut w = Vec::with_capacity(data_start);
    w.extend_from_slice(b"GGUF");
    push_u32(&mut w, version);
    push_u64(&mut w, tensors.len() as u64);
    push_u64(&mut w, kvs.len() as u64);
    for (k, v) in kvs {
        push_str(&mut w, k);
        push_value(&mut w, v);
    }
    // 张量表（offset 按数据区布局预计算：每个张量 32 对齐）
    let mut offset = data_start;
    for t in tensors {
        push_str(&mut w, &t.name);
        push_u32(&mut w, t.shape.len() as u32);
        for d in &t.shape {
            push_u64(&mut w, *d);
        }
        push_u32(&mut w, t.dtype.type_code());
        push_u64(&mut w, offset as u64);
        offset = align32(offset + t.data.len());
    }
    // 数据区
    pad_to(&mut w, data_start);
    for t in tensors {
        debug_assert_eq!(w.len() % GGUF_ALIGNMENT as usize, 0, "数据区必须保持对齐");
        w.extend_from_slice(&t.data);
        let target = align32(w.len());
        pad_to(&mut w, target);
    }
    w
}

/// 32 字节对齐（向上取整）。
fn align32(n: usize) -> usize {
    (n + GGUF_ALIGNMENT as usize - 1) & !(GGUF_ALIGNMENT as usize - 1)
}

fn pad_to(w: &mut Vec<u8>, n: usize) {
    w.resize(n.max(w.len()), 0);
}

fn str_len(s: &str) -> usize {
    8 + s.len()
}

fn push_u16(w: &mut Vec<u8>, v: u16) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn push_u32(w: &mut Vec<u8>, v: u32) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(w: &mut Vec<u8>, v: u64) {
    w.extend_from_slice(&v.to_le_bytes());
}

fn push_str(w: &mut Vec<u8>, s: &str) {
    push_u64(w, s.len() as u64);
    w.extend_from_slice(s.as_bytes());
}

fn value_len(v: &MetaValue) -> usize {
    match v {
        MetaValue::U8(_) | MetaValue::I8(_) | MetaValue::Bool(_) => 4 + 1,
        MetaValue::U16(_) | MetaValue::I16(_) => 4 + 2,
        MetaValue::U32(_) | MetaValue::I32(_) | MetaValue::F32(_) => 4 + 4,
        MetaValue::U64(_) | MetaValue::I64(_) | MetaValue::F64(_) => 4 + 8,
        MetaValue::Str(s) => 4 + str_len(s),
        MetaValue::Array(a) => 4 + array_len(a),
    }
}

fn array_len(a: &ArrayValue) -> usize {
    match a {
        ArrayValue::U8(v) => 4 + 8 + v.len(),
        ArrayValue::I8(v) => 4 + 8 + v.len(),
        ArrayValue::Bool(v) => 4 + 8 + v.len(),
        ArrayValue::U16(v) => 4 + 8 + 2 * v.len(),
        ArrayValue::I16(v) => 4 + 8 + 2 * v.len(),
        ArrayValue::U32(v) => 4 + 8 + 4 * v.len(),
        ArrayValue::I32(v) => 4 + 8 + 4 * v.len(),
        ArrayValue::F32(v) => 4 + 8 + 4 * v.len(),
        ArrayValue::U64(v) => 4 + 8 + 8 * v.len(),
        ArrayValue::I64(v) => 4 + 8 + 8 * v.len(),
        ArrayValue::F64(v) => 4 + 8 + 8 * v.len(),
        ArrayValue::Str(v) => 4 + 8 + v.iter().map(|s| str_len(s)).sum::<usize>(),
        ArrayValue::Nested(v) => 4 + 8 + v.iter().map(array_len).sum::<usize>(),
    }
}

fn push_value(w: &mut Vec<u8>, v: &MetaValue) {
    match v {
        MetaValue::U8(x) => {
            push_u32(w, type_code::U8);
            w.push(*x);
        }
        MetaValue::I8(x) => {
            push_u32(w, type_code::I8);
            w.push(*x as u8);
        }
        MetaValue::U16(x) => {
            push_u32(w, type_code::U16);
            push_u16(w, *x);
        }
        MetaValue::I16(x) => {
            push_u32(w, type_code::I16);
            push_u16(w, *x as u16);
        }
        MetaValue::U32(x) => {
            push_u32(w, type_code::U32);
            push_u32(w, *x);
        }
        MetaValue::I32(x) => {
            push_u32(w, type_code::I32);
            push_u32(w, *x as u32);
        }
        MetaValue::F32(x) => {
            push_u32(w, type_code::F32);
            push_u32(w, x.to_bits());
        }
        MetaValue::Bool(x) => {
            push_u32(w, type_code::BOOL);
            w.push(u8::from(*x));
        }
        MetaValue::Str(s) => {
            push_u32(w, type_code::STR);
            push_str(w, s);
        }
        MetaValue::Array(a) => {
            push_u32(w, type_code::ARRAY);
            push_array(w, a);
        }
        MetaValue::U64(x) => {
            push_u32(w, type_code::U64);
            push_u64(w, *x);
        }
        MetaValue::I64(x) => {
            push_u32(w, type_code::I64);
            push_u64(w, *x as u64);
        }
        MetaValue::F64(x) => {
            push_u32(w, type_code::F64);
            push_u64(w, x.to_bits());
        }
    }
}

fn push_array(w: &mut Vec<u8>, a: &ArrayValue) {
    match a {
        ArrayValue::U8(v) => {
            push_u32(w, type_code::U8);
            push_u64(w, v.len() as u64);
            w.extend_from_slice(v);
        }
        ArrayValue::I8(v) => {
            push_u32(w, type_code::I8);
            push_u64(w, v.len() as u64);
            w.extend(v.iter().map(|x| *x as u8));
        }
        ArrayValue::U16(v) => {
            push_u32(w, type_code::U16);
            push_u64(w, v.len() as u64);
            for x in v {
                push_u16(w, *x);
            }
        }
        ArrayValue::I16(v) => {
            push_u32(w, type_code::I16);
            push_u64(w, v.len() as u64);
            for x in v {
                push_u16(w, *x as u16);
            }
        }
        ArrayValue::U32(v) => {
            push_u32(w, type_code::U32);
            push_u64(w, v.len() as u64);
            for x in v {
                push_u32(w, *x);
            }
        }
        ArrayValue::I32(v) => {
            push_u32(w, type_code::I32);
            push_u64(w, v.len() as u64);
            for x in v {
                push_u32(w, *x as u32);
            }
        }
        ArrayValue::F32(v) => {
            push_u32(w, type_code::F32);
            push_u64(w, v.len() as u64);
            for x in v {
                push_u32(w, x.to_bits());
            }
        }
        ArrayValue::Bool(v) => {
            push_u32(w, type_code::BOOL);
            push_u64(w, v.len() as u64);
            w.extend(v.iter().map(|b| u8::from(*b)));
        }
        ArrayValue::Str(v) => {
            push_u32(w, type_code::STR);
            push_u64(w, v.len() as u64);
            for s in v {
                push_str(w, s);
            }
        }
        ArrayValue::U64(v) => {
            push_u32(w, type_code::U64);
            push_u64(w, v.len() as u64);
            for x in v {
                push_u64(w, *x);
            }
        }
        ArrayValue::I64(v) => {
            push_u32(w, type_code::I64);
            push_u64(w, v.len() as u64);
            for x in v {
                push_u64(w, *x as u64);
            }
        }
        ArrayValue::F64(v) => {
            push_u32(w, type_code::F64);
            push_u64(w, v.len() as u64);
            for x in v {
                push_u64(w, x.to_bits());
            }
        }
        ArrayValue::Nested(v) => {
            push_u32(w, type_code::ARRAY);
            push_u64(w, v.len() as u64);
            for item in v {
                push_array(w, item);
            }
        }
    }
}
