//! GGUF headers (versions 2 and 3): the version, the typed metadata and the
//! tensor index of a whole file in memory, read without copying. The checks,
//! their order and the descriptor budget are those of the C++ `GgufReader`
//! in 0.35.0, so a file opens or fails as it did there.

use std::collections::{HashSet, TryReserveError};

const MAGIC: &[u8; 4] = b"GGUF";
/// Magic, version, tensor count and metadata count.
const FIXED_HEADER_BYTES: usize = 24;
/// The alignment of tensor data without a `general.alignment` key.
const DEFAULT_ALIGNMENT: u64 = 32;
/// The counts are bounded by the bytes after the fixed header, at these
/// sizes per metadata entry and per tensor entry.
const METADATA_ENTRY_FILE_BYTES: u64 = 13;
const TENSOR_ENTRY_FILE_BYTES: u64 = 32;
const MAX_NAME_BYTES: usize = 65535;
const MAX_DIMENSIONS: usize = 8;
/// The budget charge per number in an array, which widens to 64 bits.
const NUMBER_ITEM_BYTES: usize = 8;

// Metadata value types.
const UINT8: u32 = 0;
const INT8: u32 = 1;
const UINT16: u32 = 2;
const INT16: u32 = 3;
const UINT32: u32 = 4;
const INT32: u32 = 5;
const FLOAT32: u32 = 6;
const BOOL: u32 = 7;
const STRING: u32 = 8;
const ARRAY: u32 = 9;
const UINT64: u32 = 10;
const INT64: u32 = 11;
const FLOAT64: u32 = 12;
/// The fewest file bytes of one array item, by value type.
const ITEM_FILE_BYTES: [u64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, 8, 0, 8, 8, 8];

/// The caller's descriptor budget: the host memory its metadata and tensor
/// descriptors may take, and what it charges for each.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub budget: usize,
    pub metadata_entry_bytes: usize,
    pub tensor_entry_bytes: usize,
    /// Per string of a string array; its bytes are charged one each.
    pub string_item_bytes: usize,
    /// Accept tensors the sink cannot size instead of rejecting the file.
    pub allow_unsized: bool,
}

/// A metadata value. The items of an array reach the sink before it
/// (`Sink::items`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value<'a> {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(&'a [u8]),
    /// An array of this many items of this kind.
    Array(ArrayKind, usize),
}

/// What the items of an array widen to: integers and booleans to 64 bits,
/// floats to `f64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayKind {
    Strings,
    Unsigneds,
    Signeds,
    Floats,
}

/// A block of array items in the parser's buffer, with the strings of a
/// string array as the sink holds them (`Sink::Str`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Items<'s, S> {
    Strings(&'s [S]),
    Unsigneds(&'s [u64]),
    Signeds(&'s [i64]),
    Floats(&'s [f64]),
}

/// Array items reach the sink in blocks of at most this many, so that the
/// parser's buffers stay small (512 KiB for strings) however long an array
/// is. Smaller blocks made reading a 262,144-token vocabulary measurably
/// slower. gguf_header_test covers arrays around this length.
const BLOCK_ITEMS: usize = 32768;

/// One tensor of the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tensor<'a> {
    pub name: &'a [u8],
    dimensions: [u64; MAX_DIMENSIONS],
    rank: usize,
    pub ggml_type: u32,
    /// The offset in the data section, as the file records it.
    pub offset: u64,
    /// Where the data starts in the parsed bytes.
    pub start: usize,
    pub size: usize,
    /// False when the sink could not size the storage type or shape; `size`
    /// then spans the bytes the file reserves for the tensor: up to the next
    /// tensor's offset, or to the end of the data.
    pub sized: bool,
}

impl Tensor<'_> {
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions[..self.rank]
    }
}

/// A tensor as the parser holds it until every tensor passed its checks:
/// 48 bytes, as its dimensions stay in the file.
struct Indexed<'a> {
    name: &'a [u8],
    /// Where its dimensions start in the parsed bytes.
    dimensions_at: usize,
    offset: u64,
    size: usize,
    ggml_type: u16,
    rank: u8,
    sized: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub alignment: u64,
}

/// A sink ended the parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stopped;

/// Receives what the parser reads. Entries already taken are void if the
/// parse fails later.
pub trait Sink<'a> {
    /// A string of a string array, as the sink takes it.
    type Str: Copy;
    fn string(bytes: &'a [u8]) -> Self::Str;
    /// The encoded bytes of a tensor of this storage type and shape, or 0
    /// when the type or shape cannot be sized.
    fn tensor_bytes(&mut self, ggml_type: u32, elements: u64, first_dimension: u64) -> usize;
    /// Takes the entry counts once they passed the limits, before any entry.
    fn counts(&mut self, metadata: usize, tensors: usize) -> Result<(), Stopped>;
    /// Takes the next block of the array the next metadata entry holds: its
    /// items from index `first` on, of `len`. Each array starts with a block
    /// at 0, which is empty for an empty array.
    fn items(
        &mut self,
        items: Items<'_, Self::Str>,
        first: usize,
        len: usize,
    ) -> Result<(), Stopped>;
    /// Takes one metadata entry, in file order, once its checks passed.
    fn metadata(&mut self, key: &'a [u8], value_type: u32, value: Value<'a>)
    -> Result<(), Stopped>;
    /// Takes one tensor, in file order, once every tensor passed its checks.
    fn tensor(&mut self, tensor: &Tensor<'a>) -> Result<(), Stopped>;
}

/// A rejected file, by the check that rejected it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejection<'a> {
    TooSmall,
    Magic,
    Version(u32),
    /// The counts exceed what the rest of the file can hold.
    Counts,
    /// The descriptors exceed the caller's budget of this many bytes, a
    /// limit of the reader and not of the format.
    Budget(usize),
    KeyUnreadable,
    KeyLimits,
    /// The type of this metadata key is missing.
    TypeUnreadable(&'a [u8]),
    Alignment,
    DuplicateKey,
    NameUnreadable,
    Dimensions,
    /// This tensor's storage type or shape cannot be sized.
    Unsized(&'a [u8]),
    DuplicateTensor,
    DataTruncated,
    /// This tensor's data is unaligned or outside the file.
    Extent(&'a [u8]),
    /// Malformed or truncated in a way that has no message of its own.
    Malformed,
}

impl Rejection<'_> {
    /// The C++ reader's message, or `None` for `Malformed`, which its caller
    /// names.
    pub fn message(&self) -> Result<Option<Vec<u8>>, TryReserveError> {
        let (text, detail): (&str, &[u8]) = match *self {
            Self::TooSmall => ("File too small for GGUF header", b""),
            Self::Magic => ("Invalid GGUF magic header", b""),
            Self::Version(version) => {
                return join("Unsupported GGUF version: ", version.to_string().as_bytes())
                    .map(Some);
            }
            Self::Counts => ("GGUF descriptor counts exceed file or memory limits", b""),
            Self::Budget(budget) => return budget_message(budget).map(Some),
            Self::KeyUnreadable => ("Failed to read metadata key", b""),
            Self::KeyLimits => ("GGUF metadata key exceeds limits", b""),
            Self::TypeUnreadable(key) => ("Failed to read metadata type for key ", key),
            Self::Alignment => ("Invalid GGUF alignment", b""),
            Self::DuplicateKey => ("Duplicate GGUF metadata key", b""),
            Self::NameUnreadable => ("Failed to read tensor name", b""),
            Self::Dimensions => ("Invalid tensor dimensions count", b""),
            Self::Unsized(name) => (
                "Invalid GGUF tensor shape or unsupported storage type: ",
                name,
            ),
            Self::DuplicateTensor => ("Duplicate GGUF tensor name", b""),
            Self::DataTruncated => ("GGUF data section is truncated before its alignment", b""),
            Self::Extent(name) => ("Tensor extent or alignment is invalid: ", name),
            Self::Malformed => return Ok(None),
        };
        join(text, detail).map(Some)
    }
}

/// Names the limit, so that a valid file it rejects is not taken for a broken
/// one.
fn budget_message(budget: usize) -> Result<Vec<u8>, TryReserveError> {
    const MIB: usize = 1 << 20;
    let limit = if budget % MIB == 0 {
        format!("{} MiB", budget / MIB)
    } else {
        format!("{budget} bytes")
    };
    join(
        "GGUF metadata and tensor index exceed the header limit of ",
        limit.as_bytes(),
    )
}

fn join(text: &str, detail: &[u8]) -> Result<Vec<u8>, TryReserveError> {
    let mut message = Vec::new();
    message.try_reserve_exact(text.len() + detail.len())?;
    message.extend_from_slice(text.as_bytes());
    message.extend_from_slice(detail);
    Ok(message)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure<'a> {
    Rejected(Rejection<'a>),
    NoMemory,
    Stopped,
}

impl<'a> From<Rejection<'a>> for Failure<'a> {
    fn from(rejection: Rejection<'a>) -> Self {
        Self::Rejected(rejection)
    }
}

impl From<Stopped> for Failure<'_> {
    fn from(_: Stopped) -> Self {
        Self::Stopped
    }
}

impl From<TryReserveError> for Failure<'_> {
    fn from(_: TryReserveError) -> Self {
        Self::NoMemory
    }
}

const MALFORMED: Failure<'static> = Failure::Rejected(Rejection::Malformed);

struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
    /// What remains of `limit`, the caller's descriptor budget.
    budget: usize,
    limit: usize,
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> u64 {
        (self.bytes.len() - self.at) as u64
    }

    fn take(&mut self, len: u64) -> Option<&'a [u8]> {
        let len = usize::try_from(len)
            .ok()
            .filter(|&len| len <= self.bytes.len() - self.at)?;
        let taken = &self.bytes[self.at..self.at + len];
        self.at += len;
        Some(taken)
    }

    fn array<const N: usize>(&mut self) -> Option<[u8; N]> {
        self.take(N as u64)?.try_into().ok()
    }

    fn u8(&mut self) -> Option<u8> {
        self.array::<1>().map(|[byte]| byte)
    }

    fn u16(&mut self) -> Option<u16> {
        self.array().map(u16::from_le_bytes)
    }

    fn u32(&mut self) -> Option<u32> {
        self.array().map(u32::from_le_bytes)
    }

    fn u64(&mut self) -> Option<u64> {
        self.array().map(u64::from_le_bytes)
    }

    fn string(&mut self) -> Option<&'a [u8]> {
        let len = self.u64()?;
        self.take(len)
    }

    /// Charges `count` descriptors of `item` bytes, or nothing when they
    /// exceed the budget.
    fn charge(&mut self, count: u64, item: usize) -> Result<(), Failure<'static>> {
        if item != 0 && count > (self.budget / item) as u64 {
            return Err(Rejection::Budget(self.limit).into());
        }
        self.budget -= count as usize * item;
        Ok(())
    }

    /// `rank` dimensions; the others are 0.
    fn dimensions(&mut self, rank: usize) -> Option<[u64; MAX_DIMENSIONS]> {
        let mut dimensions = [0; MAX_DIMENSIONS];
        for dimension in &mut dimensions[..rank] {
            *dimension = self.u64()?;
        }
        Some(dimensions)
    }

    /// A string whose bytes are charged one each.
    fn charged_string(&mut self) -> Result<&'a [u8], Failure<'static>> {
        let string = self.string().ok_or(MALFORMED)?;
        self.charge(string.len() as u64, 1)?;
        Ok(string)
    }

    /// A value of any type but `STRING` and `ARRAY`.
    fn scalar(&mut self, value_type: u32) -> Option<Value<'a>> {
        Some(match value_type {
            UINT8 => Value::Unsigned(self.u8()?.into()),
            INT8 => Value::Signed(i8::from_le_bytes(self.array()?).into()),
            UINT16 => Value::Unsigned(self.u16()?.into()),
            INT16 => Value::Signed(i16::from_le_bytes(self.array()?).into()),
            UINT32 => Value::Unsigned(self.u32()?.into()),
            INT32 => Value::Signed(i32::from_le_bytes(self.array()?).into()),
            FLOAT32 => Value::Float(f32::from_bits(self.u32()?).into()),
            BOOL => Value::Bool(match self.u8()? {
                0 => false,
                1 => true,
                _ => return None,
            }),
            UINT64 => Value::Unsigned(self.u64()?),
            INT64 => Value::Signed(i64::from_le_bytes(self.array()?)),
            FLOAT64 => Value::Float(f64::from_bits(self.u64()?)),
            _ => return None,
        })
    }
}

/// Buffers for one block of array items, reused across arrays.
struct Arrays<S> {
    strings: Vec<S>,
    unsigneds: Vec<u64>,
    signeds: Vec<i64>,
    floats: Vec<f64>,
}

impl<S> Default for Arrays<S> {
    fn default() -> Self {
        Self {
            strings: Vec::new(),
            unsigneds: Vec::new(),
            signeds: Vec::new(),
            floats: Vec::new(),
        }
    }
}

impl<S: Copy> Arrays<S> {
    /// Reads an array and hands its items to the sink in blocks. Writers
    /// pick the narrowest integer type, so every width widens; booleans read
    /// as 0 and 1.
    fn read<'a, K: Sink<'a, Str = S>>(
        &mut self,
        cursor: &mut Cursor<'a>,
        limits: &Limits,
        sink: &mut K,
    ) -> Result<Value<'a>, Failure<'a>> {
        let item_type = cursor.u32().ok_or(MALFORMED)?;
        let len = cursor.u64().ok_or(MALFORMED)?;
        if item_type > FLOAT64 || item_type == ARRAY {
            return Err(MALFORMED);
        }
        let charge = if item_type == STRING {
            limits.string_item_bytes
        } else {
            NUMBER_ITEM_BYTES
        };
        if len > cursor.remaining() / ITEM_FILE_BYTES[item_type as usize] {
            return Err(MALFORMED);
        }
        cursor.charge(len, charge)?;
        // At most one item per remaining byte.
        let len = len as usize;
        let strings = &mut self.strings;
        let unsigneds = &mut self.unsigneds;
        let signeds = &mut self.signeds;
        let floats = &mut self.floats;
        let kind = match item_type {
            STRING => {
                blocks(
                    strings,
                    len,
                    || cursor.charged_string().map(K::string),
                    |items, first| sink.items(Items::Strings(items), first, len),
                )?;
                ArrayKind::Strings
            }
            UINT8 => {
                blocks(
                    unsigneds,
                    len,
                    || cursor.u8().map(u64::from).ok_or(MALFORMED),
                    |items, first| sink.items(Items::Unsigneds(items), first, len),
                )?;
                ArrayKind::Unsigneds
            }
            UINT16 => {
                blocks(
                    unsigneds,
                    len,
                    || cursor.u16().map(u64::from).ok_or(MALFORMED),
                    |items, first| sink.items(Items::Unsigneds(items), first, len),
                )?;
                ArrayKind::Unsigneds
            }
            UINT32 => {
                blocks(
                    unsigneds,
                    len,
                    || cursor.u32().map(u64::from).ok_or(MALFORMED),
                    |items, first| sink.items(Items::Unsigneds(items), first, len),
                )?;
                ArrayKind::Unsigneds
            }
            UINT64 => {
                blocks(
                    unsigneds,
                    len,
                    || cursor.u64().ok_or(MALFORMED),
                    |items, first| sink.items(Items::Unsigneds(items), first, len),
                )?;
                ArrayKind::Unsigneds
            }
            BOOL => {
                blocks(
                    unsigneds,
                    len,
                    || {
                        cursor
                            .u8()
                            .filter(|&value| value <= 1)
                            .map(u64::from)
                            .ok_or(MALFORMED)
                    },
                    |items, first| sink.items(Items::Unsigneds(items), first, len),
                )?;
                ArrayKind::Unsigneds
            }
            INT8 => {
                blocks(
                    signeds,
                    len,
                    || {
                        cursor
                            .array()
                            .map(|bytes| i8::from_le_bytes(bytes).into())
                            .ok_or(MALFORMED)
                    },
                    |items, first| sink.items(Items::Signeds(items), first, len),
                )?;
                ArrayKind::Signeds
            }
            INT16 => {
                blocks(
                    signeds,
                    len,
                    || {
                        cursor
                            .array()
                            .map(|bytes| i16::from_le_bytes(bytes).into())
                            .ok_or(MALFORMED)
                    },
                    |items, first| sink.items(Items::Signeds(items), first, len),
                )?;
                ArrayKind::Signeds
            }
            INT32 => {
                blocks(
                    signeds,
                    len,
                    || {
                        cursor
                            .array()
                            .map(|bytes| i32::from_le_bytes(bytes).into())
                            .ok_or(MALFORMED)
                    },
                    |items, first| sink.items(Items::Signeds(items), first, len),
                )?;
                ArrayKind::Signeds
            }
            INT64 => {
                blocks(
                    signeds,
                    len,
                    || cursor.array().map(i64::from_le_bytes).ok_or(MALFORMED),
                    |items, first| sink.items(Items::Signeds(items), first, len),
                )?;
                ArrayKind::Signeds
            }
            FLOAT32 => {
                blocks(
                    floats,
                    len,
                    || {
                        cursor
                            .u32()
                            .map(|bits| f32::from_bits(bits).into())
                            .ok_or(MALFORMED)
                    },
                    |items, first| sink.items(Items::Floats(items), first, len),
                )?;
                ArrayKind::Floats
            }
            _ => {
                blocks(
                    floats,
                    len,
                    || cursor.u64().map(f64::from_bits).ok_or(MALFORMED),
                    |items, first| sink.items(Items::Floats(items), first, len),
                )?;
                ArrayKind::Floats
            }
        };
        Ok(Value::Array(kind, len))
    }
}

/// Reads `len` items from `next` into `buffer` and hands them to `deliver`
/// block by block, with the index of each block's first item, until `next`
/// fails. The first block starts at 0, even when there are no items.
fn blocks<T>(
    buffer: &mut Vec<T>,
    len: usize,
    mut next: impl FnMut() -> Result<T, Failure<'static>>,
    mut deliver: impl FnMut(&[T], usize) -> Result<(), Stopped>,
) -> Result<(), Failure<'static>> {
    buffer.clear();
    buffer.try_reserve(len.min(BLOCK_ITEMS))?;
    let mut first = 0;
    loop {
        let end = len.min(first + BLOCK_ITEMS);
        buffer.clear();
        for _ in first..end {
            buffer.push(next()?);
        }
        deliver(buffer, first)?;
        if end == len {
            return Ok(());
        }
        first = end;
    }
}

/// The product of the dimensions, or 0 when one is 0 or the product
/// overflows, as `GgufTensorInfo::ElementCount` counts them.
fn element_count(dimensions: &[u64]) -> u64 {
    dimensions
        .iter()
        .try_fold(1u64, |count, &dimension| {
            if dimension == 0 {
                None
            } else {
                count.checked_mul(dimension)
            }
        })
        .unwrap_or(0)
}

/// Reads the header of the GGUF file in `bytes`, handing its metadata to the
/// sink as it goes and its tensors once all are checked.
pub fn parse<'a, K: Sink<'a>>(
    bytes: &'a [u8],
    limits: &Limits,
    sink: &mut K,
) -> Result<Header, Failure<'a>> {
    if bytes.len() < FIXED_HEADER_BYTES {
        return Err(Rejection::TooSmall.into());
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err(Rejection::Magic.into());
    }
    let mut cursor = Cursor {
        bytes,
        at: MAGIC.len(),
        budget: limits.budget,
        limit: limits.budget,
    };
    // The fixed header is there, as checked above.
    let version = cursor.u32().ok_or(MALFORMED)?;
    if version != 2 && version != 3 {
        return Err(Rejection::Version(version).into());
    }
    let tensor_count = cursor.u64().ok_or(MALFORMED)?;
    let metadata_count = cursor.u64().ok_or(MALFORMED)?;
    let remaining = cursor.remaining();
    if metadata_count > remaining / METADATA_ENTRY_FILE_BYTES
        || tensor_count > remaining / TENSOR_ENTRY_FILE_BYTES
    {
        return Err(Rejection::Counts.into());
    }
    cursor.charge(metadata_count, limits.metadata_entry_bytes)?;
    cursor.charge(tensor_count, limits.tensor_entry_bytes)?;
    // At most one entry per remaining byte.
    sink.counts(metadata_count as usize, tensor_count as usize)?;

    let mut keys = HashSet::new();
    keys.try_reserve(metadata_count as usize)?;
    let mut alignment = DEFAULT_ALIGNMENT;
    let mut arrays = Arrays::<K::Str>::default();
    for _ in 0..metadata_count {
        let key = cursor.string().ok_or(Rejection::KeyUnreadable)?;
        if key.is_empty() || key.len() > MAX_NAME_BYTES {
            return Err(Rejection::KeyLimits.into());
        }
        cursor.charge(key.len() as u64, 1)?;
        let value_type = cursor.u32().ok_or(Rejection::TypeUnreadable(key))?;
        let value = match value_type {
            ARRAY => arrays.read(&mut cursor, limits, sink)?,
            STRING => Value::String(cursor.charged_string()?),
            _ => cursor.scalar(value_type).ok_or(MALFORMED)?,
        };
        if key == b"general.alignment" {
            alignment = match value {
                Value::Unsigned(value)
                    if value.is_power_of_two() && value <= u64::from(u32::MAX) =>
                {
                    value
                }
                _ => return Err(Rejection::Alignment.into()),
            };
        }
        if !keys.insert(key) {
            return Err(Rejection::DuplicateKey.into());
        }
        sink.metadata(key, value_type, value)?;
    }
    // Duplicate checks and buffers are freed once done, so that only the
    // tensor index below remains while the sink takes the tensors.
    drop((keys, arrays));

    let mut tensors = Vec::new();
    tensors.try_reserve_exact(tensor_count as usize)?;
    let mut names = HashSet::new();
    names.try_reserve(tensor_count as usize)?;
    for _ in 0..tensor_count {
        let name = cursor.string().ok_or(Rejection::NameUnreadable)?;
        if name.is_empty() || name.len() > MAX_NAME_BYTES {
            return Err(MALFORMED);
        }
        cursor.charge(name.len() as u64, 1)?;
        let rank = match cursor.u32() {
            Some(rank) if (1..=MAX_DIMENSIONS as u32).contains(&rank) => rank as usize,
            _ => return Err(Rejection::Dimensions.into()),
        };
        let dimensions_at = cursor.at;
        let dimensions = cursor.dimensions(rank).ok_or(MALFORMED)?;
        let ggml_type = cursor
            .u32()
            .and_then(|ggml_type| u16::try_from(ggml_type).ok())
            .ok_or(MALFORMED)?;
        let elements = element_count(&dimensions[..rank]);
        let size = sink.tensor_bytes(ggml_type.into(), elements, dimensions[0]);
        // A layout the sink cannot size may belong to an extension; its
        // extent follows from the next tensor's offset once all are known.
        let sized = size != 0;
        if !sized && (!limits.allow_unsized || elements == 0) {
            return Err(Rejection::Unsized(name).into());
        }
        let offset = cursor.u64().ok_or(MALFORMED)?;
        if !names.insert(name) {
            return Err(Rejection::DuplicateTensor.into());
        }
        tensors.push(Indexed {
            name,
            dimensions_at,
            offset,
            size,
            ggml_type,
            rank: rank as u8,
            sized,
        });
    }
    drop(names);

    let header = Header { version, alignment };
    // Tensor data starts at the next alignment boundary. A tensorless file,
    // such as a metadata-only first split shard, may end unpadded.
    if tensors.is_empty() {
        return Ok(header);
    }
    let padding = (alignment - cursor.at as u64 % alignment) % alignment;
    if padding > cursor.remaining() {
        return Err(Rejection::DataTruncated.into());
    }
    let data_start = cursor.at + padding as usize;
    let data_size = (bytes.len() - data_start) as u64;
    let mut offsets = Vec::new();
    if tensors.iter().any(|tensor| !tensor.sized) {
        offsets.try_reserve_exact(tensors.len())?;
        offsets.extend(tensors.iter().map(|tensor| tensor.offset));
        offsets.sort_unstable();
    }
    for tensor in &mut tensors {
        if !tensor.sized && tensor.offset < data_size {
            let next = offsets.partition_point(|&offset| offset <= tensor.offset);
            let end = offsets.get(next).copied().unwrap_or(data_size);
            tensor.size = (end - tensor.offset) as usize;
        }
        if tensor.offset % alignment != 0
            || tensor.offset > data_size
            || (!tensor.sized && tensor.size == 0)
            || tensor.size as u64 > data_size - tensor.offset
        {
            return Err(Rejection::Extent(tensor.name).into());
        }
    }
    drop(offsets);
    for tensor in &tensors {
        let rank = tensor.rank.into();
        // Read once already, so they are there.
        let mut at = Cursor {
            bytes,
            at: tensor.dimensions_at,
            budget: 0,
            limit: 0,
        };
        let dimensions = at.dimensions(rank).ok_or(MALFORMED)?;
        sink.tensor(&Tensor {
            name: tensor.name,
            dimensions,
            rank,
            ggml_type: tensor.ggml_type.into(),
            offset: tensor.offset,
            // Within the file, as checked above.
            start: data_start + tensor.offset as usize,
            size: tensor.size,
            sized: tensor.sized,
        })?;
    }
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: Limits = Limits {
        budget: 128 << 20,
        metadata_entry_bytes: 40 + 64,
        tensor_entry_bytes: 80 + 128,
        string_item_bytes: 16,
        allow_unsized: false,
    };

    /// Writes GGUF bytes field by field.
    #[derive(Default)]
    struct Bytes(Vec<u8>);

    impl Bytes {
        fn header(tensors: u64, metadata: u64) -> Self {
            Self::default().raw(MAGIC).u32(3).u64(tensors).u64(metadata)
        }
        fn raw(mut self, bytes: &[u8]) -> Self {
            self.0.extend_from_slice(bytes);
            self
        }
        fn u32(self, value: u32) -> Self {
            self.raw(&value.to_le_bytes())
        }
        fn u64(self, value: u64) -> Self {
            self.raw(&value.to_le_bytes())
        }
        fn string(self, value: &[u8]) -> Self {
            self.u64(value.len() as u64).raw(value)
        }
        fn key(self, key: &[u8], value_type: u32) -> Self {
            self.string(key).u32(value_type)
        }
        fn tensor(self, name: &[u8], dimensions: &[u64], ggml_type: u32, offset: u64) -> Self {
            let mut out = self.string(name).u32(dimensions.len() as u32);
            for &dimension in dimensions {
                out = out.u64(dimension);
            }
            out.u32(ggml_type).u64(offset)
        }
        fn pad(mut self, alignment: usize, data: usize) -> Self {
            while self.0.len() % alignment != 0 {
                self.0.push(0);
            }
            self.0.resize(self.0.len() + data, 0xAB);
            self
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    enum Owned {
        Unsigned(u64),
        Signed(i64),
        Float(u64),
        Bool(bool),
        String(Vec<u8>),
        Strings(Vec<Vec<u8>>),
        Unsigneds(Vec<u64>),
        Signeds(Vec<i64>),
        Floats(Vec<u64>),
    }

    /// Name, dimensions, type, offset, start, size and whether it is sized.
    type Row = (Vec<u8>, Vec<u64>, u32, u64, usize, usize, bool);

    /// Sizes F32 (type 0) and 32-element Q8_0 blocks (type 8), as core does.
    #[derive(Default)]
    struct Collect {
        metadata: Vec<(Vec<u8>, u32, Owned)>,
        tensors: Vec<Row>,
        counts: Option<(usize, usize)>,
        /// The array the blocks so far hold, and how many blocks came.
        array: Option<Owned>,
        blocks: usize,
        stop_after: Option<usize>,
    }

    impl<'a> Sink<'a> for Collect {
        type Str = &'a [u8];
        fn string(bytes: &'a [u8]) -> Self::Str {
            bytes
        }
        fn tensor_bytes(&mut self, ggml_type: u32, elements: u64, first: u64) -> usize {
            match ggml_type {
                0 => elements as usize * 4,
                8 if first % 32 == 0 => elements as usize / 32 * 34,
                _ => 0,
            }
        }
        fn counts(&mut self, metadata: usize, tensors: usize) -> Result<(), Stopped> {
            assert!(self.counts.replace((metadata, tensors)).is_none());
            Ok(())
        }
        fn items(
            &mut self,
            items: Items<'_, &'a [u8]>,
            first: usize,
            len: usize,
        ) -> Result<(), Stopped> {
            if first == 0 {
                self.array = Some(match items {
                    Items::Strings(_) => Owned::Strings(Vec::new()),
                    Items::Unsigneds(_) => Owned::Unsigneds(Vec::new()),
                    Items::Signeds(_) => Owned::Signeds(Vec::new()),
                    Items::Floats(_) => Owned::Floats(Vec::new()),
                });
            }
            self.blocks += 1;
            let (held, added) = match (self.array.as_mut(), items) {
                (Some(Owned::Strings(all)), Items::Strings(new)) => {
                    let held = all.len();
                    all.extend(new.iter().map(|s| s.to_vec()));
                    (held, new.len())
                }
                (Some(Owned::Unsigneds(all)), Items::Unsigneds(new)) => {
                    let held = all.len();
                    all.extend_from_slice(new);
                    (held, new.len())
                }
                (Some(Owned::Signeds(all)), Items::Signeds(new)) => {
                    let held = all.len();
                    all.extend_from_slice(new);
                    (held, new.len())
                }
                (Some(Owned::Floats(all)), Items::Floats(new)) => {
                    let held = all.len();
                    all.extend(new.iter().map(|f| f.to_bits()));
                    (held, new.len())
                }
                _ => panic!("a block of another array"),
            };
            assert_eq!(held, first);
            assert!(first + added <= len && (added > 0 || len == 0));
            Ok(())
        }
        fn metadata(
            &mut self,
            key: &'a [u8],
            value_type: u32,
            value: Value<'a>,
        ) -> Result<(), Stopped> {
            if self.stop_after == Some(self.metadata.len()) {
                return Err(Stopped);
            }
            let value = match value {
                Value::Unsigned(v) => Owned::Unsigned(v),
                Value::Signed(v) => Owned::Signed(v),
                Value::Float(v) => Owned::Float(v.to_bits()),
                Value::Bool(v) => Owned::Bool(v),
                Value::String(v) => Owned::String(v.to_vec()),
                Value::Array(kind, len) => {
                    let array = self.array.take().expect("items before an array");
                    let held = match &array {
                        Owned::Strings(v) => (ArrayKind::Strings, v.len()),
                        Owned::Unsigneds(v) => (ArrayKind::Unsigneds, v.len()),
                        Owned::Signeds(v) => (ArrayKind::Signeds, v.len()),
                        Owned::Floats(v) => (ArrayKind::Floats, v.len()),
                        _ => unreachable!(),
                    };
                    assert_eq!(held, (kind, len));
                    array
                }
            };
            self.metadata.push((key.to_vec(), value_type, value));
            Ok(())
        }
        fn tensor(&mut self, tensor: &Tensor<'a>) -> Result<(), Stopped> {
            self.tensors.push((
                tensor.name.to_vec(),
                tensor.dimensions().to_vec(),
                tensor.ggml_type,
                tensor.offset,
                tensor.start,
                tensor.size,
                tensor.sized,
            ));
            Ok(())
        }
    }

    fn run<'a>(bytes: &'a [u8], limits: &Limits) -> (Result<Header, Failure<'a>>, Collect) {
        let mut sink = Collect::default();
        (parse(bytes, limits, &mut sink), sink)
    }

    fn rejection(bytes: &[u8]) -> Rejection<'_> {
        match parse(bytes, &LIMITS, &mut Collect::default()) {
            Err(Failure::Rejected(rejection)) => rejection,
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    #[test]
    fn every_value_type() {
        let bytes = Bytes::header(0, 14)
            .key(b"u8", UINT8)
            .raw(&[200])
            .key(b"i8", INT8)
            .raw(&[0x80])
            .key(b"u16", UINT16)
            .raw(&65535u16.to_le_bytes())
            .key(b"i16", INT16)
            .raw(&(-2i16).to_le_bytes())
            .key(b"u32", UINT32)
            .u32(7)
            .key(b"i32", INT32)
            .raw(&(-3i32).to_le_bytes())
            .key(b"f32", FLOAT32)
            .raw(&1.5f32.to_le_bytes())
            .key(b"bool", BOOL)
            .raw(&[1])
            .key(b"str", STRING)
            .string(b"\xff\x00x")
            .key(b"u64", UINT64)
            .u64(u64::MAX)
            .key(b"i64", INT64)
            .u64(u64::MAX)
            .key(b"f64", FLOAT64)
            .raw(&(-0.25f64).to_le_bytes())
            .key(b"strings", ARRAY)
            .u32(STRING)
            .u64(2)
            .string(b"a")
            .string(b"")
            .key(b"flags", ARRAY)
            .u32(BOOL)
            .u64(2)
            .raw(&[0, 1]);
        let (header, sink) = run(&bytes.0, &LIMITS);
        assert_eq!(
            header,
            Ok(Header {
                version: 3,
                alignment: 32
            })
        );
        let values: Vec<_> = sink.metadata.into_iter().map(|(_, t, v)| (t, v)).collect();
        assert_eq!(
            values,
            [
                (UINT8, Owned::Unsigned(200)),
                (INT8, Owned::Signed(-128)),
                (UINT16, Owned::Unsigned(65535)),
                (INT16, Owned::Signed(-2)),
                (UINT32, Owned::Unsigned(7)),
                (INT32, Owned::Signed(-3)),
                (FLOAT32, Owned::Float(1.5f64.to_bits())),
                (BOOL, Owned::Bool(true)),
                (STRING, Owned::String(b"\xff\x00x".to_vec())),
                (UINT64, Owned::Unsigned(u64::MAX)),
                (INT64, Owned::Signed(-1)),
                (FLOAT64, Owned::Float((-0.25f64).to_bits())),
                (ARRAY, Owned::Strings(vec![b"a".to_vec(), Vec::new()])),
                (ARRAY, Owned::Unsigneds(vec![0, 1])),
            ]
        );
    }

    #[test]
    fn narrow_arrays_widen() {
        let bytes = Bytes::header(0, 3)
            .key(b"i16", ARRAY)
            .u32(INT16)
            .u64(2)
            .raw(&(-32768i16).to_le_bytes())
            .raw(&7i16.to_le_bytes())
            .key(b"u8", ARRAY)
            .u32(UINT8)
            .u64(1)
            .raw(&[255])
            .key(b"f32", ARRAY)
            .u32(FLOAT32)
            .u64(1)
            .raw(&0.5f32.to_le_bytes());
        let (header, sink) = run(&bytes.0, &LIMITS);
        assert!(header.is_ok());
        assert_eq!(sink.metadata[0].2, Owned::Signeds(vec![-32768, 7]));
        assert_eq!(sink.metadata[1].2, Owned::Unsigneds(vec![255]));
        assert_eq!(sink.metadata[2].2, Owned::Floats(vec![0.5f64.to_bits()]));
    }

    #[test]
    fn long_arrays_arrive_in_blocks() {
        let numbers = 2 * BLOCK_ITEMS + 1;
        let mut bytes = Bytes::header(0, 3)
            .key(b"numbers", ARRAY)
            .u32(UINT16)
            .u64(numbers as u64);
        for i in 0..numbers {
            bytes = bytes.raw(&(i as u16).to_le_bytes());
        }
        bytes = bytes
            .key(b"strings", ARRAY)
            .u32(STRING)
            .u64(BLOCK_ITEMS as u64);
        for i in 0..BLOCK_ITEMS {
            bytes = bytes.string(i.to_string().as_bytes());
        }
        let bytes = bytes.key(b"empty", ARRAY).u32(FLOAT32).u64(0);
        let (header, sink) = run(&bytes.0, &LIMITS);
        assert!(header.is_ok());
        // Three blocks of numbers, one of strings, one empty.
        assert_eq!(sink.blocks, 5);
        assert_eq!(
            sink.metadata[0].2,
            Owned::Unsigneds((0..numbers as u64).map(|i| i as u16 as u64).collect())
        );
        assert_eq!(
            sink.metadata[1].2,
            Owned::Strings(
                (0..BLOCK_ITEMS)
                    .map(|i| i.to_string().into_bytes())
                    .collect()
            )
        );
        assert_eq!(sink.metadata[2].2, Owned::Floats(Vec::new()));
    }

    #[test]
    fn tensors_and_alignment() {
        let bytes = Bytes::header(2, 1)
            .key(b"general.alignment", UINT32)
            .u32(64)
            .tensor(b"a", &[32, 2], 8, 0)
            .tensor(b"b", &[3], 0, 128)
            .pad(64, 140);
        let (header, sink) = run(&bytes.0, &LIMITS);
        assert_eq!(
            header,
            Ok(Header {
                version: 3,
                alignment: 64
            })
        );
        assert_eq!(sink.counts, Some((1, 2)));
        let data = bytes.0.len() - 140;
        assert_eq!(
            sink.tensors,
            [
                (b"a".to_vec(), vec![32, 2], 8, 0, data, 68, true),
                (b"b".to_vec(), vec![3], 0, 128, data + 128, 12, true),
            ]
        );
    }

    #[test]
    fn unsized_tensors_reach_the_next_offset() {
        let bytes = Bytes::header(3, 0)
            .tensor(b"odd", &[100, 2], 8, 0)
            .tensor(b"plain", &[8], 0, 256)
            .tensor(b"unknown", &[64], 200, 288)
            .pad(32, 288 + 96);
        assert_eq!(rejection(&bytes.0), Rejection::Unsized(b"odd"));
        let limits = Limits {
            allow_unsized: true,
            ..LIMITS
        };
        let (header, sink) = run(&bytes.0, &limits);
        assert!(header.is_ok());
        let sizes: Vec<_> = sink.tensors.iter().map(|t| (t.5, t.6)).collect();
        assert_eq!(sizes, [(256, false), (32, true), (96, false)]);
        // An unsized tensor with no elements is never accepted.
        let empty = Bytes::header(1, 0).tensor(b"e", &[0], 200, 0).pad(32, 32);
        assert_eq!(
            parse(&empty.0, &limits, &mut Collect::default()),
            Err(Rejection::Unsized(b"e").into())
        );
    }

    #[test]
    fn rejections_in_check_order() {
        assert_eq!(rejection(&[0; 23]), Rejection::TooSmall);
        assert_eq!(rejection(&[0; 24]), Rejection::Magic);
        let mut wrong = Bytes::header(0, 0).0;
        wrong[4] = 4;
        assert_eq!(rejection(&wrong), Rejection::Version(4));
        assert_eq!(rejection(&Bytes::header(0, 1).0), Rejection::Counts);
        let short_key = Bytes::header(0, 1).u64(100).raw(&[0; 5]);
        assert_eq!(rejection(&short_key.0), Rejection::KeyUnreadable);
        let empty_key = Bytes::header(0, 1).string(b"").raw(&[0; 5]);
        assert_eq!(rejection(&empty_key.0), Rejection::KeyLimits);
        let no_type = Bytes::header(0, 1).string(b"key").raw(&[0; 2]);
        assert_eq!(rejection(&no_type.0), Rejection::TypeUnreadable(b"key"));
        let bad_type = Bytes::header(0, 1).key(b"k", 13).raw(&[0; 8]);
        assert_eq!(rejection(&bad_type.0), Rejection::Malformed);
        let bad_bool = Bytes::header(0, 1).key(b"k", BOOL).raw(&[2]).raw(&[0; 4]);
        assert_eq!(rejection(&bad_bool.0), Rejection::Malformed);
        for value_type in [BOOL, INT32] {
            let alignment = Bytes::header(0, 1)
                .key(b"general.alignment", value_type)
                .u32(1);
            assert_eq!(rejection(&alignment.0), Rejection::Alignment);
        }
        for alignment in [0, 3, 1 << 32] {
            let bytes = Bytes::header(0, 1)
                .key(b"general.alignment", UINT64)
                .u64(alignment);
            assert_eq!(rejection(&bytes.0), Rejection::Alignment);
        }
        let duplicate = Bytes::header(0, 2)
            .key(b"k", UINT8)
            .raw(&[0])
            .key(b"k", UINT8)
            .raw(&[0]);
        assert_eq!(rejection(&duplicate.0), Rejection::DuplicateKey);
        let nested = Bytes::header(0, 1).key(b"k", ARRAY).u32(ARRAY).u64(0);
        assert_eq!(rejection(&nested.0), Rejection::Malformed);
        let long = Bytes::header(0, 1)
            .key(b"k", ARRAY)
            .u32(UINT32)
            .u64(2)
            .u32(0);
        assert_eq!(rejection(&long.0), Rejection::Malformed);

        let no_name = Bytes::header(1, 0).u64(1 << 40).raw(&[0; 24]);
        assert_eq!(rejection(&no_name.0), Rejection::NameUnreadable);
        let empty_name = Bytes::header(1, 0).tensor(b"", &[32], 0, 0).pad(32, 128);
        assert_eq!(rejection(&empty_name.0), Rejection::Malformed);
        for rank in [0, 9] {
            let bytes = Bytes::header(1, 0).string(b"t").u32(rank).raw(&[0; 32]);
            assert_eq!(rejection(&bytes.0), Rejection::Dimensions);
        }
        let wide_type = Bytes::header(1, 0)
            .tensor(b"t", &[8], 0x10000, 0)
            .pad(32, 32);
        assert_eq!(rejection(&wide_type.0), Rejection::Malformed);
        let overflow = Bytes::header(1, 0)
            .tensor(b"t", &[u64::MAX, 2], 0, 0)
            .pad(32, 32);
        assert_eq!(rejection(&overflow.0), Rejection::Unsized(b"t"));
        let duplicate = Bytes::header(2, 0)
            .tensor(b"t", &[8], 0, 0)
            .tensor(b"t", &[8], 0, 32)
            .pad(32, 64);
        assert_eq!(rejection(&duplicate.0), Rejection::DuplicateTensor);
        let unpadded = Bytes::header(1, 0).tensor(b"t", &[8], 0, 0);
        assert_eq!(rejection(&unpadded.0), Rejection::DataTruncated);
        for offset in [1, 32, u64::MAX] {
            let bytes = Bytes::header(1, 0)
                .tensor(b"t", &[8], 0, offset)
                .pad(32, 32);
            assert_eq!(rejection(&bytes.0), Rejection::Extent(b"t"));
        }
    }

    #[test]
    fn tensorless_files_need_no_padding() {
        let bytes = Bytes::header(0, 1).key(b"k", UINT8).raw(&[1]);
        assert!(run(&bytes.0, &LIMITS).0.is_ok());
    }

    #[test]
    fn budget() {
        // One string of three bytes after two descriptors fits exactly.
        let bytes = Bytes::header(0, 1).key(b"k", STRING).string(b"abc");
        let exact = Limits {
            budget: 104 + 1 + 3,
            ..LIMITS
        };
        assert!(run(&bytes.0, &exact).0.is_ok());
        let short = Limits {
            budget: 104 + 1 + 2,
            ..LIMITS
        };
        assert_eq!(run(&bytes.0, &short).0, Err(Rejection::Budget(107).into()));
        let no_key = Limits {
            budget: 104,
            ..LIMITS
        };
        assert_eq!(run(&bytes.0, &no_key).0, Err(Rejection::Budget(104).into()));
        let no_entry = Limits {
            budget: 103,
            ..LIMITS
        };
        assert_eq!(
            run(&bytes.0, &no_entry).0,
            Err(Rejection::Budget(103).into())
        );
        // Numbers in an array are charged 8 bytes each.
        let numbers = Bytes::header(0, 1)
            .key(b"k", ARRAY)
            .raw(&UINT8.to_le_bytes());
        let numbers = numbers.raw(&2u64.to_le_bytes()).raw(&[1, 2]);
        let one_number = Limits {
            budget: 104 + 1 + 8,
            ..LIMITS
        };
        assert_eq!(
            run(&numbers.0, &one_number).0,
            Err(Rejection::Budget(113).into())
        );
    }

    #[test]
    fn truncation_never_reaches_tensors() {
        let bytes = Bytes::header(2, 2)
            .key(b"names", ARRAY)
            .u32(STRING)
            .u64(2)
            .string(b"x")
            .string(b"yz")
            .key(b"general.alignment", UINT8)
            .raw(&[16])
            .tensor(b"a", &[32], 8, 0)
            .tensor(b"b", &[4], 0, 48)
            .pad(16, 64);
        assert!(run(&bytes.0, &LIMITS).0.is_ok());
        for len in 0..bytes.0.len() {
            let (header, sink) = run(&bytes.0[..len], &LIMITS);
            assert!(header.is_err(), "{len} bytes");
            assert!(sink.tensors.is_empty());
        }
    }

    #[test]
    fn a_sink_can_stop() {
        let bytes = Bytes::header(0, 2)
            .key(b"a", UINT8)
            .raw(&[0])
            .key(b"b", UINT8)
            .raw(&[0]);
        let mut sink = Collect {
            stop_after: Some(1),
            ..Collect::default()
        };
        assert_eq!(parse(&bytes.0, &LIMITS, &mut sink), Err(Failure::Stopped));
        assert_eq!(sink.metadata.len(), 1);
    }

    #[test]
    fn messages() {
        let message = |rejection: Rejection| rejection.message().unwrap();
        assert_eq!(message(Rejection::Malformed), None);
        assert_eq!(
            message(Rejection::Version(7)).unwrap(),
            b"Unsupported GGUF version: 7"
        );
        assert_eq!(
            message(Rejection::Extent(b"w\xff")).unwrap(),
            b"Tensor extent or alignment is invalid: w\xff"
        );
        assert_eq!(
            message(Rejection::Budget(128 << 20)).unwrap(),
            b"GGUF metadata and tensor index exceed the header limit of 128 MiB"
        );
        assert_eq!(
            message(Rejection::Budget(107)).unwrap(),
            b"GGUF metadata and tensor index exceed the header limit of 107 bytes"
        );
    }
}
