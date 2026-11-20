//! The `GUFOMNF2` disk manifest of format versions 2 and 3: explicit
//! little-endian, length-delimited fields, then the CRC-64 of every preceding
//! byte. Only manifests with window components use version 3, which also
//! records the window size, so each manifest has exactly one encoding.

use crate::crc;

/// include/gufo/cache_manifest.h repeats the limits and versions for C++.
pub const MAX_MANIFEST_BYTES: usize = 16 * 1024 * 1024;
pub const MAGIC: [u8; 8] = *b"GUFOMNF2";
pub const FORMAT_VERSION: u32 = 2;
pub const WINDOW_FORMAT_VERSION: u32 = 3;
pub const MAX_TOKENS: usize = 2 * 1024 * 1024;
pub const MAX_COMPONENTS: usize = 256;
pub const MAX_REFERENCES: usize = 65536;
pub const MAX_INPUT_BYTES: usize = 65536;

/// `ComponentKind` in src/cache/types.hpp.
pub const APPEND_ROWS: u8 = 0;
pub const PRIVATE_STATE: u8 = 1;
pub const WINDOW_ROWS: u8 = 2;
/// The last `CheckpointPurpose` (`kLearned`).
pub const MAX_PURPOSE: u8 = 3;

/// The trailer, the smallest prefix the plan reads past, and the fixed fields
/// before the input length.
const TRAILER_BYTES: usize = 8;
const MIN_MANIFEST_BYTES: usize = 20;
const FIXED_BYTES: usize = 65;
/// A component's ID, layout version, kind, row and chunk sizes, state bytes
/// and valid rows; with the chunk count and both presence bytes, the
/// smallest component.
const DESCRIPTOR_BYTES: usize = 41;
const MIN_COMPONENT_BYTES: usize = DESCRIPTOR_BYTES + 6;
const PAYLOAD_BYTES: usize = 32;
/// Decoded items handed over per callback.
const TOKEN_BLOCK: usize = 4096;
const PAYLOAD_BLOCK: usize = 512;

/// One payload reference, laid out as `GufoCachePayload`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Payload {
    pub file: [u8; 16],
    pub bytes: u64,
    pub checksum: u64,
}

/// The manifest's fixed fields.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Header {
    pub checkpoint: u64,
    pub lineage: u64,
    pub compatibility: [u8; 32],
    pub purpose: u8,
    pub rank: u32,
}

/// A component's descriptor, position and references; `chunks` holds or
/// encodes its full chunks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Component<C> {
    pub id: u32,
    pub layout_version: u32,
    pub kind: u8,
    pub row_bytes: u64,
    pub rows_per_chunk: u64,
    pub state_bytes: u64,
    pub window_rows: u64,
    pub position_id: u32,
    pub valid_rows: u64,
    pub chunks: C,
    pub tail: Option<Payload>,
    pub private_state: Option<Payload>,
}

impl<C> Component<C> {
    /// The same component with other chunks.
    pub fn with_chunks<D>(&self, chunks: D) -> Component<D> {
        Component {
            id: self.id,
            layout_version: self.layout_version,
            kind: self.kind,
            row_bytes: self.row_bytes,
            rows_per_chunk: self.rows_per_chunk,
            state_bytes: self.state_bytes,
            window_rows: self.window_rows,
            position_id: self.position_id,
            valid_rows: self.valid_rows,
            chunks,
            tail: self.tail,
            private_state: self.private_state,
        }
    }
}

/// A component's full chunks, decoded or encoded.
pub trait Chunks {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> Payload;
}

impl Chunks for &[Payload] {
    fn len(&self) -> usize {
        <[Payload]>::len(self)
    }

    fn get(&self, index: usize) -> Payload {
        self[index]
    }
}

/// Chunks still in their 32-byte encoding.
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodedChunks<'a>(&'a [u8]);

impl Chunks for EncodedChunks<'_> {
    fn len(&self) -> usize {
        self.0.len() / PAYLOAD_BYTES
    }

    fn get(&self, index: usize) -> Payload {
        payload(&self.0[index * PAYLOAD_BYTES..][..PAYLOAD_BYTES])
    }
}

fn payload(bytes: &[u8]) -> Payload {
    let word = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("eight bytes"));
    Payload {
        file: bytes[..16].try_into().expect("16-byte file ID"),
        bytes: word(16),
        checksum: word(24),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stopped;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Failure {
    /// The manifest breaks the format or its limits.
    Rejected,
    NoMemory,
    Stopped,
}

impl From<Stopped> for Failure {
    fn from(_: Stopped) -> Self {
        Self::Stopped
    }
}

impl From<std::collections::TryReserveError> for Failure {
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::NoMemory
    }
}

type Result<T> = std::result::Result<T, Failure>;

fn multiply(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).ok_or(Failure::Rejected)
}

fn has_rows(kind: u8) -> bool {
    kind == APPEND_ROWS || kind == WINDOW_ROWS
}

/// `FirstChunkRow` in src/cache/types.hpp: the first row of the first chunk a
/// checkpoint at `rows` retains.
fn first_chunk_row<C>(c: &Component<C>, rows: u64) -> u64 {
    let first = if c.kind == WINDOW_ROWS && rows > c.window_rows {
        rows - c.window_rows
    } else {
        0
    };
    if c.rows_per_chunk == 0 {
        0
    } else {
        first - first % c.rows_per_chunk
    }
}

/// Whether `files` holds a file twice; sorts it.
fn duplicates(files: &mut [[u8; 16]]) -> bool {
    files.sort_unstable();
    files.windows(2).any(|pair| pair[0] == pair[1])
}

/// Rejects incoherent geometry and excessive metadata, as both directions
/// must. `files` is scratch space for the duplicate checks.
pub fn validate<C: Chunks>(
    header: &Header,
    input_bytes: usize,
    tokens: usize,
    components: &[Component<C>],
    files: &mut Vec<[u8; 16]>,
) -> Result<()> {
    if header.checkpoint == 0
        || header.lineage == 0
        || tokens > MAX_TOKENS
        || input_bytes > MAX_INPUT_BYTES
        || components.is_empty()
        || components.len() > MAX_COMPONENTS
        || header.purpose > MAX_PURPOSE
    {
        return Err(Failure::Rejected);
    }
    let mut private_files: Vec<[u8; 16]> = Vec::new();
    private_files.try_reserve_exact(2 * components.len())?;
    let mut references = 0usize;
    let mut previous: Option<u32> = None;
    for c in components {
        if c.position_id != c.id
            || c.layout_version == 0
            || previous.is_some_and(|previous| c.id <= previous)
        {
            return Err(Failure::Rejected);
        }
        previous = Some(c.id);
        references = references
            .saturating_add(c.chunks.len())
            .saturating_add(usize::from(c.tail.is_some()))
            .saturating_add(usize::from(c.private_state.is_some()));
        if references > MAX_REFERENCES {
            return Err(Failure::Rejected);
        }
        if has_rows(c.kind) {
            if c.row_bytes == 0
                || c.rows_per_chunk == 0
                || c.state_bytes != 0
                || c.private_state.is_some()
                || (c.kind == WINDOW_ROWS) != (c.window_rows != 0)
                || c.chunks.len() as u64
                    != (c.valid_rows - first_chunk_row(c, c.valid_rows)) / c.rows_per_chunk
            {
                return Err(Failure::Rejected);
            }
            multiply(c.row_bytes, c.valid_rows)?;
            let full_bytes = multiply(c.row_bytes, c.rows_per_chunk)?;
            files.clear();
            files.try_reserve(c.chunks.len())?;
            for i in 0..c.chunks.len() {
                let p = c.chunks.get(i);
                if p.bytes != full_bytes {
                    return Err(Failure::Rejected);
                }
                files.push(p.file);
            }
            if duplicates(files) {
                return Err(Failure::Rejected);
            }
            let remaining = c.valid_rows % c.rows_per_chunk;
            match c.tail {
                None if remaining == 0 => {}
                Some(tail) if remaining != 0 && tail.bytes == multiply(c.row_bytes, remaining)? => {
                }
                _ => return Err(Failure::Rejected),
            }
        } else if c.kind == PRIVATE_STATE {
            match c.private_state {
                Some(state)
                    if c.row_bytes == 0
                        && c.rows_per_chunk == 0
                        && c.state_bytes != 0
                        && c.window_rows == 0
                        && c.chunks.is_empty()
                        && c.tail.is_none()
                        && state.bytes == c.state_bytes
                        && c.valid_rows == tokens as u64 => {}
                _ => return Err(Failure::Rejected),
            }
        } else {
            return Err(Failure::Rejected);
        }
        private_files.extend(c.tail.iter().chain(&c.private_state).map(|p| p.file));
    }
    if duplicates(&mut private_files) {
        return Err(Failure::Rejected);
    }
    Ok(())
}

/// Little-endian fields, each checked against the remaining bytes.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn blob(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.bytes.len() {
            return Err(Failure::Rejected);
        }
        let (blob, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(blob)
    }

    fn number(&mut self, width: usize) -> Result<u64> {
        let mut word = [0u8; 8];
        word[..width].copy_from_slice(self.blob(width)?);
        Ok(u64::from_le_bytes(word))
    }

    /// A count of at most `max` elements of `element_bytes`, which must fit
    /// in the remaining bytes.
    fn count(&mut self, max: usize, element_bytes: usize) -> Result<usize> {
        let n = self.number(4)?;
        if n > max as u64 || n > (self.bytes.len() / element_bytes) as u64 {
            return Err(Failure::Rejected);
        }
        Ok(n as usize)
    }

    fn payload(&mut self) -> Result<Payload> {
        Ok(payload(self.blob(PAYLOAD_BYTES)?))
    }

    /// An optional payload, counted against `references`.
    fn optional_payload(&mut self, references: &mut usize) -> Result<Option<Payload>> {
        match self.number(1)? {
            0 => Ok(None),
            1 => {
                *references += 1;
                if *references > MAX_REFERENCES {
                    return Err(Failure::Rejected);
                }
                Ok(Some(self.payload()?))
            }
            _ => Err(Failure::Rejected),
        }
    }
}

/// What a decoded manifest holds, so the caller can reserve its storage
/// before decoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Plan {
    pub input_bytes: usize,
    pub tokens: usize,
    pub components: usize,
    pub chunks: usize,
    pub references: usize,
}

/// Reads the counts of a manifest whose structure is sound, without checking
/// its magic, version, trailer or geometry, which decoding does.
pub fn plan(bytes: &[u8]) -> Result<Plan> {
    if bytes.len() < MIN_MANIFEST_BYTES || bytes.len() > MAX_MANIFEST_BYTES {
        return Err(Failure::Rejected);
    }
    let mut r = Reader {
        bytes: &bytes[..bytes.len() - TRAILER_BYTES],
    };
    r.blob(FIXED_BYTES)?;
    let mut plan = Plan {
        input_bytes: r.count(MAX_INPUT_BYTES, 1)?,
        ..Plan::default()
    };
    r.blob(plan.input_bytes)?;
    plan.tokens = r.count(MAX_TOKENS, 4)?;
    r.blob(plan.tokens * 4)?;
    plan.components = r.count(MAX_COMPONENTS, MIN_COMPONENT_BYTES)?;
    for _ in 0..plan.components {
        // The kind byte follows the ID and layout version; a window
        // component also records its window size.
        if r.blob(DESCRIPTOR_BYTES)?[8] == WINDOW_ROWS {
            r.blob(8)?;
        }
        let chunks = r.count(MAX_REFERENCES - plan.references, PAYLOAD_BYTES)?;
        plan.chunks += chunks;
        plan.references += chunks;
        r.blob(chunks * PAYLOAD_BYTES)?;
        for _ in 0..2 {
            r.optional_payload(&mut plan.references)?;
        }
    }
    if !r.bytes.is_empty() {
        return Err(Failure::Rejected);
    }
    Ok(plan)
}

/// Receives a decoded manifest: `begin` once, `tokens` in blocks, then for
/// each component `component` followed by its chunks in blocks. Empty blocks
/// are not delivered.
pub trait Sink {
    fn begin(
        &mut self,
        header: &Header,
        input: &[u8],
        tokens: usize,
        components: usize,
    ) -> std::result::Result<(), Stopped>;
    fn tokens(&mut self, tokens: &[u32]) -> std::result::Result<(), Stopped>;
    fn component(&mut self, component: &Component<usize>) -> std::result::Result<(), Stopped>;
    fn chunks(&mut self, chunks: &[Payload]) -> std::result::Result<(), Stopped>;
}

/// Decodes and validates `bytes`, then reports the manifest to `sink`, which
/// takes nothing from a manifest that fails.
pub fn decode(bytes: &[u8], sink: &mut impl Sink) -> Result<()> {
    if bytes.len() < MIN_MANIFEST_BYTES || bytes.len() > MAX_MANIFEST_BYTES {
        return Err(Failure::Rejected);
    }
    let (body, trailer) = bytes.split_at(bytes.len() - TRAILER_BYTES);
    if crc::checksum(body, 0) != u64::from_le_bytes(trailer.try_into().expect("eight bytes")) {
        return Err(Failure::Rejected);
    }
    let mut r = Reader { bytes: body };
    if r.blob(MAGIC.len())? != MAGIC {
        return Err(Failure::Rejected);
    }
    let version = r.number(4)?;
    if version != u64::from(FORMAT_VERSION) && version != u64::from(WINDOW_FORMAT_VERSION) {
        return Err(Failure::Rejected);
    }
    let mut header = Header {
        checkpoint: r.number(8)?,
        lineage: r.number(8)?,
        ..Header::default()
    };
    header.compatibility.copy_from_slice(r.blob(32)?);
    header.purpose = r.number(1)? as u8;
    header.rank = r.number(4)? as u32;
    let input_bytes = r.count(MAX_INPUT_BYTES, 1)?;
    let input = r.blob(input_bytes)?;
    let tokens = r.count(MAX_TOKENS, 4)?;
    let token_bytes = r.blob(tokens * 4)?;
    let count = r.count(MAX_COMPONENTS, MIN_COMPONENT_BYTES)?;
    let mut components: Vec<Component<EncodedChunks<'_>>> = Vec::new();
    components.try_reserve_exact(count)?;
    let mut references = 0usize;
    for _ in 0..count {
        let id = r.number(4)? as u32;
        let mut c = Component {
            id,
            layout_version: r.number(4)? as u32,
            kind: r.number(1)? as u8,
            row_bytes: r.number(8)?,
            rows_per_chunk: r.number(8)?,
            state_bytes: r.number(8)?,
            position_id: id,
            valid_rows: r.number(8)?,
            ..Component::default()
        };
        if c.kind == WINDOW_ROWS {
            if version != u64::from(WINDOW_FORMAT_VERSION) {
                return Err(Failure::Rejected);
            }
            c.window_rows = r.number(8)?;
        }
        let chunks = r.count(MAX_REFERENCES - references, PAYLOAD_BYTES)?;
        references += chunks;
        c.chunks = EncodedChunks(r.blob(chunks * PAYLOAD_BYTES)?);
        c.tail = r.optional_payload(&mut references)?;
        c.private_state = r.optional_payload(&mut references)?;
        components.push(c);
    }
    if !r.bytes.is_empty()
        || (version == u64::from(WINDOW_FORMAT_VERSION)
            && !components.iter().any(|c| c.kind == WINDOW_ROWS))
    {
        return Err(Failure::Rejected);
    }
    validate(&header, input.len(), tokens, &components, &mut Vec::new())?;

    sink.begin(&header, input, tokens, components.len())?;
    let mut block = [0u32; TOKEN_BLOCK];
    for words in token_bytes.chunks(4 * TOKEN_BLOCK) {
        let block = &mut block[..words.len() / 4];
        for (token, word) in block.iter_mut().zip(words.chunks_exact(4)) {
            *token = u32::from_le_bytes(word.try_into().expect("four bytes"));
        }
        sink.tokens(block)?;
    }
    let mut block = [Payload::default(); PAYLOAD_BLOCK];
    for c in &components {
        let chunks = c.chunks;
        sink.component(&c.with_chunks(chunks.len()))?;
        for encoded in chunks.0.chunks(PAYLOAD_BYTES * PAYLOAD_BLOCK) {
            let block = &mut block[..encoded.len() / PAYLOAD_BYTES];
            for (p, bytes) in block.iter_mut().zip(encoded.chunks_exact(PAYLOAD_BYTES)) {
                *p = payload(bytes);
            }
            sink.chunks(block)?;
        }
    }
    Ok(())
}

/// Fills a caller buffer field by field.
struct Writer<'a> {
    bytes: &'a mut [u8],
    at: usize,
}

impl Writer<'_> {
    fn blob(&mut self, bytes: &[u8]) {
        self.bytes[self.at..self.at + bytes.len()].copy_from_slice(bytes);
        self.at += bytes.len();
    }

    fn number(&mut self, n: u64, width: usize) {
        self.blob(&n.to_le_bytes()[..width]);
    }

    fn payload(&mut self, p: &Payload) {
        self.blob(&p.file);
        self.number(p.bytes, 8);
        self.number(p.checksum, 8);
    }

    fn optional_payload(&mut self, p: &Option<Payload>) {
        self.number(p.is_some().into(), 1);
        if let Some(p) = p {
            self.payload(p);
        }
    }
}

/// Validates and encodes a manifest into the buffer of exactly the encoded
/// length that `buffer` returns.
pub fn encode<'o>(
    header: &Header,
    input: &[u8],
    tokens: &[u32],
    components: &[Component<&[Payload]>],
    buffer: impl FnOnce(usize) -> std::result::Result<&'o mut [u8], Stopped>,
) -> Result<()> {
    validate(
        header,
        input.len(),
        tokens.len(),
        components,
        &mut Vec::new(),
    )?;
    let windowed = components.iter().any(|c| c.kind == WINDOW_ROWS);
    // Validation bounds every term; the whole stays far below the limit.
    let mut len = FIXED_BYTES + 4 + input.len() + 4 + 4 * tokens.len() + 4;
    for c in components {
        len += MIN_COMPONENT_BYTES + PAYLOAD_BYTES * c.chunks.len();
        len += if c.kind == WINDOW_ROWS { 8 } else { 0 };
        len += PAYLOAD_BYTES
            * (usize::from(c.tail.is_some()) + usize::from(c.private_state.is_some()));
    }
    if len > MAX_MANIFEST_BYTES - TRAILER_BYTES {
        return Err(Failure::Rejected);
    }
    let bytes = buffer(len + TRAILER_BYTES)?;
    if bytes.len() != len + TRAILER_BYTES {
        return Err(Failure::Stopped);
    }
    let mut w = Writer { bytes, at: 0 };
    w.blob(&MAGIC);
    w.number(
        u64::from(if windowed {
            WINDOW_FORMAT_VERSION
        } else {
            FORMAT_VERSION
        }),
        4,
    );
    w.number(header.checkpoint, 8);
    w.number(header.lineage, 8);
    w.blob(&header.compatibility);
    w.number(header.purpose.into(), 1);
    w.number(header.rank.into(), 4);
    w.number(input.len() as u64, 4);
    w.blob(input);
    w.number(tokens.len() as u64, 4);
    let token_bytes = &mut w.bytes[w.at..w.at + 4 * tokens.len()];
    for (word, token) in token_bytes.chunks_exact_mut(4).zip(tokens) {
        word.copy_from_slice(&token.to_le_bytes());
    }
    w.at += 4 * tokens.len();
    w.number(components.len() as u64, 4);
    for c in components {
        w.number(c.id.into(), 4);
        w.number(c.layout_version.into(), 4);
        w.number(c.kind.into(), 1);
        w.number(c.row_bytes, 8);
        w.number(c.rows_per_chunk, 8);
        w.number(c.state_bytes, 8);
        w.number(c.valid_rows, 8);
        if c.kind == WINDOW_ROWS {
            w.number(c.window_rows, 8);
        }
        w.number(c.chunks.len() as u64, 4);
        for p in c.chunks {
            w.payload(p);
        }
        w.optional_payload(&c.tail);
        w.optional_payload(&c.private_state);
    }
    debug_assert_eq!(w.at, len);
    // The checksum covers every preceding byte, not itself.
    let checksum = crc::checksum(&w.bytes[..len], 0);
    w.number(checksum, 8);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(n: u64) -> [u8; 16] {
        let mut file = [0u8; 16];
        file[..8].copy_from_slice(&n.to_le_bytes());
        file
    }

    #[derive(Default, Debug, PartialEq)]
    struct Decoded {
        header: Header,
        input: Vec<u8>,
        tokens: Vec<u32>,
        components: Vec<(Component<usize>, Vec<Payload>)>,
        stop_at: Option<usize>,
        calls: usize,
    }

    impl Decoded {
        fn call(&mut self) -> std::result::Result<(), Stopped> {
            self.calls += 1;
            if self.stop_at == Some(self.calls) {
                return Err(Stopped);
            }
            Ok(())
        }
    }

    impl Sink for Decoded {
        fn begin(
            &mut self,
            header: &Header,
            input: &[u8],
            tokens: usize,
            components: usize,
        ) -> std::result::Result<(), Stopped> {
            self.call()?;
            self.header = *header;
            self.input = input.to_vec();
            self.tokens.reserve(tokens);
            self.components.reserve(components);
            Ok(())
        }

        fn tokens(&mut self, tokens: &[u32]) -> std::result::Result<(), Stopped> {
            self.call()?;
            assert!(!tokens.is_empty());
            self.tokens.extend_from_slice(tokens);
            Ok(())
        }

        fn component(&mut self, c: &Component<usize>) -> std::result::Result<(), Stopped> {
            self.call()?;
            self.components.push((*c, Vec::new()));
            Ok(())
        }

        fn chunks(&mut self, chunks: &[Payload]) -> std::result::Result<(), Stopped> {
            self.call()?;
            assert!(!chunks.is_empty());
            self.components
                .last_mut()
                .unwrap()
                .1
                .extend_from_slice(chunks);
            Ok(())
        }
    }

    struct Fixture {
        header: Header,
        input: Vec<u8>,
        tokens: Vec<u32>,
        chunks: Vec<Vec<Payload>>,
        components: Vec<Component<()>>,
    }

    impl Fixture {
        /// Append rows over `tokens` rows in chunks of `chunk` rows, a window
        /// component when `window`, and private state.
        fn with_chunks(tokens: usize, window: bool, chunk: u64) -> Self {
            let mut f = Fixture {
                header: Header {
                    checkpoint: 9,
                    lineage: 4,
                    compatibility: [0x42; 32],
                    purpose: 2,
                    rank: 77,
                },
                input: b"input identity".to_vec(),
                tokens: (0..tokens as u32).collect(),
                chunks: Vec::new(),
                components: Vec::new(),
            };
            let rows = tokens as u64;
            let tail = |n: u64, bytes: u64| {
                (rows % chunk != 0).then_some(Payload {
                    file: id(n),
                    bytes,
                    checksum: 5,
                })
            };
            f.chunks.push(
                (0..rows / chunk)
                    .map(|i| Payload {
                        file: id(i + 1),
                        bytes: 4 * chunk,
                        checksum: i,
                    })
                    .collect(),
            );
            f.components.push(Component {
                id: 1,
                layout_version: 1,
                kind: APPEND_ROWS,
                row_bytes: 4,
                rows_per_chunk: chunk,
                position_id: 1,
                valid_rows: rows,
                tail: tail(100_000, 4 * (rows % chunk)),
                ..Component::default()
            });
            if window {
                let window_rows = chunk + chunk / 2;
                let first = rows.saturating_sub(window_rows) / chunk * chunk;
                f.chunks.push(
                    (first / chunk..rows / chunk)
                        .map(|i| Payload {
                            file: id(50_000 + i),
                            bytes: chunk * 8,
                            checksum: 1,
                        })
                        .collect(),
                );
                f.components.push(Component {
                    id: 3,
                    layout_version: 2,
                    kind: WINDOW_ROWS,
                    row_bytes: 8,
                    rows_per_chunk: chunk,
                    window_rows,
                    position_id: 3,
                    valid_rows: rows,
                    tail: tail(100_001, 8 * (rows % chunk)),
                    ..Component::default()
                });
            }
            f.chunks.push(Vec::new());
            f.components.push(Component {
                id: 7,
                layout_version: 1,
                kind: PRIVATE_STATE,
                state_bytes: 16,
                position_id: 7,
                valid_rows: rows,
                private_state: Some(Payload {
                    file: id(100_002),
                    bytes: 16,
                    checksum: 3,
                }),
                ..Component::default()
            });
            f
        }

        fn new(tokens: usize, window: bool) -> Self {
            Self::with_chunks(tokens, window, 2048)
        }

        fn views(&self) -> Vec<Component<&[Payload]>> {
            self.components
                .iter()
                .zip(&self.chunks)
                .map(|(c, chunks)| c.with_chunks(&chunks[..]))
                .collect()
        }

        fn encode(&self) -> Result<Vec<u8>> {
            let mut out = Vec::new();
            encode(
                &self.header,
                &self.input,
                &self.tokens,
                &self.views(),
                |len| {
                    out.resize(len, 0);
                    Ok(&mut out[..])
                },
            )?;
            Ok(out)
        }
    }

    fn decoded(bytes: &[u8]) -> Result<Decoded> {
        let mut d = Decoded::default();
        decode(bytes, &mut d)?;
        Ok(d)
    }

    #[test]
    fn round_trip_and_versions() {
        for window in [false, true] {
            for tokens in [0, 1, 2048, 5000, 3 * TOKEN_BLOCK + 1] {
                if window && tokens == 0 {
                    continue;
                }
                let f = Fixture::new(tokens, window);
                let bytes = f.encode().unwrap();
                let version = if window { 3 } else { 2 };
                assert_eq!(&bytes[..12], [&MAGIC[..], &[version, 0, 0, 0]].concat());
                let d = decoded(&bytes).unwrap();
                assert_eq!(d.header, f.header);
                assert_eq!(d.input, f.input);
                assert_eq!(d.tokens, f.tokens);
                assert_eq!(d.components.len(), f.components.len());
                for ((c, chunks), (expected, expected_chunks)) in
                    d.components.iter().zip(f.components.iter().zip(&f.chunks))
                {
                    assert_eq!(*c, expected.with_chunks(expected_chunks.len()));
                    assert_eq!(chunks, expected_chunks);
                }
                let p = plan(&bytes).unwrap();
                assert_eq!(p.tokens, tokens);
                assert_eq!(p.components, f.components.len());
                assert_eq!(p.chunks, f.chunks.iter().map(Vec::len).sum::<usize>());
            }
        }
    }

    #[test]
    fn chunk_blocks_split_large_components() {
        let f = Fixture::new(2048 * (PAYLOAD_BLOCK + 3), false);
        let d = decoded(&f.encode().unwrap()).unwrap();
        assert_eq!(d.components[0].1, f.chunks[0]);
        // begin, the token blocks, the first component and its two blocks,
        // then the private state.
        let token_blocks = f.tokens.len().div_ceil(TOKEN_BLOCK);
        assert_eq!(d.calls, 1 + token_blocks + 1 + 2 + 1);
    }

    #[test]
    fn every_truncation_and_bit_flip_is_rejected() {
        let bytes = Fixture::with_chunks(50, true, 8).encode().unwrap();
        for len in 0..bytes.len() {
            assert_eq!(
                decoded(&bytes[..len]).unwrap_err(),
                Failure::Rejected,
                "{len}"
            );
        }
        for at in 0..bytes.len() {
            let mut flipped = bytes.clone();
            flipped[at] ^= 1 << (at % 8);
            assert_eq!(decoded(&flipped).unwrap_err(), Failure::Rejected, "{at}");
        }
    }

    /// Re-signs a mutated body so its structure is checked.
    fn signed(mut body: Vec<u8>) -> Vec<u8> {
        let checksum = crc::checksum(&body, 0);
        body.extend_from_slice(&checksum.to_le_bytes());
        body
    }

    #[test]
    fn signed_truncations_are_rejected() {
        let bytes = Fixture::with_chunks(50, true, 8).encode().unwrap();
        let body = &bytes[..bytes.len() - 8];
        for len in 12..body.len() {
            assert_eq!(
                decoded(&signed(body[..len].to_vec())).unwrap_err(),
                Failure::Rejected
            );
        }
        // A trailing byte, a window component in version 2 and a version 3
        // manifest without one.
        let mut long = body.to_vec();
        long.push(0);
        assert_eq!(decoded(&signed(long)).unwrap_err(), Failure::Rejected);
        let mut v2 = body.to_vec();
        v2[8] = 2;
        assert_eq!(decoded(&signed(v2)).unwrap_err(), Failure::Rejected);
        let plain = Fixture::with_chunks(50, false, 8).encode().unwrap();
        let mut v3 = plain[..plain.len() - 8].to_vec();
        v3[8] = 3;
        assert_eq!(decoded(&signed(v3)).unwrap_err(), Failure::Rejected);
    }

    #[test]
    fn validation_rejects_incoherent_manifests() {
        let rejected = |change: &dyn Fn(&mut Fixture)| {
            let mut f = Fixture::new(5000, true);
            change(&mut f);
            assert_eq!(f.encode().unwrap_err(), Failure::Rejected);
        };
        rejected(&|f| f.header.checkpoint = 0);
        rejected(&|f| f.header.lineage = 0);
        rejected(&|f| f.header.purpose = MAX_PURPOSE + 1);
        rejected(&|f| f.input = vec![0; MAX_INPUT_BYTES + 1]);
        rejected(&|f| f.components[1].id = 1);
        rejected(&|f| f.components[0].position_id = 2);
        rejected(&|f| f.components[0].layout_version = 0);
        rejected(&|f| f.components[0].kind = 3);
        rejected(&|f| f.components[0].window_rows = 1);
        rejected(&|f| f.components[1].window_rows = 0);
        rejected(&|f| f.components[0].state_bytes = 1);
        rejected(&|f| f.components[0].row_bytes = u64::MAX);
        rejected(&|f| f.components[0].tail = None);
        rejected(&|f| f.chunks[0][1].file = f.chunks[0][0].file);
        rejected(&|f| f.chunks[0][1].bytes += 1);
        rejected(&|f| {
            f.chunks[0].pop();
        });
        rejected(&|f| f.components[2].valid_rows += 1);
        rejected(&|f| f.components[2].private_state = None);
        rejected(&|f| f.components[2].tail = f.components[0].tail);
        rejected(&|f| {
            f.components[2].private_state = f.components[0].tail.map(|t| Payload { bytes: 16, ..t })
        });
        rejected(&|f| {
            f.components.clear();
            f.chunks.clear();
        });
        let mut f = Fixture::new(5000, true);
        f.tokens = vec![0; MAX_TOKENS + 1];
        f.components[2].valid_rows = f.tokens.len() as u64;
        assert_eq!(f.encode().unwrap_err(), Failure::Rejected);
    }

    #[test]
    fn stopping_sink_and_buffer() {
        let f = Fixture::new(5000, true);
        let bytes = f.encode().unwrap();
        let all = decoded(&bytes).unwrap().calls;
        for stop_at in 1..=all {
            let mut d = Decoded {
                stop_at: Some(stop_at),
                ..Decoded::default()
            };
            assert_eq!(decode(&bytes, &mut d).unwrap_err(), Failure::Stopped);
        }
        let result = encode(&f.header, &f.input, &f.tokens, &f.views(), |_| Err(Stopped));
        assert_eq!(result.unwrap_err(), Failure::Stopped);
    }
}
