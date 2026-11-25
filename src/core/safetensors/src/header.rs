//! The header: a JSON object of tensor entries and an optional
//! `__metadata__` object of strings (or null), checked against the file it
//! heads.
//!
//! The header is read from the JSON parser's events without building a
//! value tree: the walk keeps only what it checks. It notes the first fault
//! and reads on, so a JSON error anywhere wins, as when the whole header
//! is parsed before any member is checked.

use crate::dtype::Dtype;
use crate::entry::{self, Entry, Member, Numbers, Raw};
use crate::layout::{self, LENGTH_BYTES, Untiled};
use crate::{Problem, Reason};
use gufo_json::{Event, MAX_DEPTH, Sink, Stopped};
use std::collections::TryReserveError;
use std::ops::Range;

/// The member that holds free-form metadata instead of a tensor.
pub const METADATA: &str = "__metadata__";

/// A rejected file: why, the member that fails (empty for the file or the
/// header as a whole) and what is wrong, in words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rejection {
    pub reason: Reason,
    pub name: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    Rejected(Rejection),
    NoMemory,
    /// The JSON parser broke its contract: a bug, not bad input.
    Internal,
}

impl From<TryReserveError> for Failure {
    fn from(_: TryReserveError) -> Self {
        Self::NoMemory
    }
}

/// Joins `parts`, which may hold input such as a name, into one string.
fn join(parts: &[&str]) -> Result<String, Failure> {
    let mut out = String::new();
    out.try_reserve_exact(parts.iter().map(|part| part.len()).sum())?;
    parts.iter().for_each(|part| out.push_str(part));
    Ok(out)
}

/// Rejects the file, saying what is wrong in `message`, joined from parts.
fn reject<T>(reason: Reason, name: &str, message: &[&str]) -> Result<T, Failure> {
    Err(Failure::Rejected(Rejection {
        reason,
        name: join(&[name])?,
        message: join(message)?,
    }))
}

/// A checked header. Its names and strings live in one buffer; its tensors
/// keep the header's order.
#[derive(Debug)]
pub struct Header {
    text: String,
    tensors: Vec<(Range<usize>, Entry)>,
    metadata: Vec<(Range<usize>, Range<usize>)>,
}

impl Header {
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &Entry)> {
        self.tensors
            .iter()
            .map(|(name, entry)| (&self.text[name.clone()], entry))
    }

    /// The `__metadata__` entries in header order, each a key and a string.
    pub fn metadata(&self) -> impl Iterator<Item = (&str, &str)> {
        self.metadata
            .iter()
            .map(|(key, value)| (&self.text[key.clone()], &self.text[value.clone()]))
    }
}

/// Where the walk is: the container each open event entered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    /// The header object, whose members are tensors and `__metadata__`.
    Root,
    Metadata,
    /// A tensor entry and the arrays it checks.
    Entry,
    Shape,
    Offsets,
    /// A value nothing reads.
    Skip,
}

/// The tensor entry member whose value comes next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Field {
    Dtype,
    Shape,
    Offsets,
    Other,
}

/// The first fault: why, what is wrong, and where in `text` its name and,
/// for an unknown type, the type are.
#[derive(Debug, Clone)]
struct Fault {
    reason: Reason,
    message: &'static str,
    name: Range<usize>,
    dtype: Option<Range<usize>>,
}

struct Walk {
    payload: u64,
    text: String,
    places: Vec<Place>,
    /// The header member being read, and the metadata key.
    member: Range<usize>,
    key: Range<usize>,
    field: Field,
    entry: Raw,
    tensors: Vec<(Range<usize>, Entry)>,
    metadata: Vec<(Range<usize>, Range<usize>)>,
    fault: Option<Fault>,
    /// Why the walk stopped the parser.
    failure: Option<Failure>,
}

impl Walk {
    fn new(payload: u64) -> Result<Self, Failure> {
        let mut places = Vec::new();
        // The parser nests at most MAX_DEPTH containers below the header.
        places.try_reserve_exact(MAX_DEPTH + 1)?;
        Ok(Self {
            payload,
            text: String::new(),
            places,
            member: 0..0,
            key: 0..0,
            field: Field::Other,
            entry: Raw::default(),
            tensors: Vec::new(),
            metadata: Vec::new(),
            fault: None,
            failure: None,
        })
    }

    /// Keeps `bytes`, which the parser decoded as UTF-8, in `text`.
    fn keep(&mut self, bytes: &[u8]) -> Result<Range<usize>, Failure> {
        let text = std::str::from_utf8(bytes).map_err(|_| Failure::Internal)?;
        self.text.try_reserve(text.len())?;
        let start = self.text.len();
        self.text.push_str(text);
        Ok(start..self.text.len())
    }

    fn note(&mut self, (reason, message): Problem, name: Range<usize>) {
        self.note_dtype((reason, message), name, None);
    }

    /// Notes a fault, with the unknown type the header names if that is it.
    fn note_dtype(
        &mut self,
        (reason, message): Problem,
        name: Range<usize>,
        dtype: Option<Range<usize>>,
    ) {
        if self.fault.is_none() {
            self.fault = Some(Fault {
                reason,
                message,
                name,
                dtype,
            });
        }
    }

    /// Enters a container. The parser bounds the depth, so this cannot grow
    /// past what `new` reserved.
    fn open(&mut self, place: Place) -> Result<(), Failure> {
        if self.places.len() == self.places.capacity() {
            return Err(Failure::Internal);
        }
        self.places.push(place);
        Ok(())
    }

    /// Passes over a value nothing reads, entering it if it is a container.
    fn skip(&mut self, event: &Event<'_>) -> Result<(), Failure> {
        match event {
            Event::BeginArray | Event::BeginObject => self.open(Place::Skip),
            _ => Ok(()),
        }
    }

    fn take(&mut self, event: Event<'_>) -> Result<(), Failure> {
        if matches!(event, Event::End) {
            if self.places.pop() == Some(Place::Entry) {
                self.end_entry()?;
            }
            return Ok(());
        }
        let Some(&place) = self.places.last() else {
            if matches!(event, Event::BeginObject) {
                return self.open(Place::Root);
            }
            self.note((Reason::NotObject, "not a JSON object"), 0..0);
            return self.skip(&event);
        };
        match (place, event) {
            (Place::Root, Event::Key(name)) => {
                self.member = self.keep(name)?;
                Ok(())
            }
            (Place::Root, event) => self.member_value(event),
            (Place::Metadata, Event::Key(key)) => {
                self.key = self.keep(key)?;
                Ok(())
            }
            (Place::Metadata, Event::String(text)) => {
                let value = self.keep(text)?;
                self.metadata.try_reserve(1)?;
                self.metadata.push((self.key.clone(), value));
                Ok(())
            }
            (Place::Metadata, event) => {
                let problem = (Reason::Metadata, "metadata value is not a string");
                self.note(problem, self.key.clone());
                self.skip(&event)
            }
            (Place::Entry, Event::Key(field)) => {
                self.field = match field {
                    b"dtype" => Field::Dtype,
                    b"shape" => Field::Shape,
                    b"data_offsets" => Field::Offsets,
                    _ => Field::Other,
                };
                Ok(())
            }
            (Place::Entry, event) => self.field_value(event),
            (Place::Shape | Place::Offsets, event) => {
                let value = match &event {
                    Event::Number(_, token) => entry::integer(token),
                    _ => None,
                };
                let numbers = match place {
                    Place::Shape => &mut self.entry.shape,
                    _ => &mut self.entry.offsets,
                };
                if let Member::Read(numbers) = numbers {
                    numbers.push(value);
                }
                self.skip(&event)
            }
            (Place::Skip, event) => self.skip(&event),
        }
    }

    /// The value of a header member: `__metadata__` or a tensor entry.
    fn member_value(&mut self, event: Event<'_>) -> Result<(), Failure> {
        let object = matches!(event, Event::BeginObject);
        if &self.text[self.member.clone()] == METADATA {
            if object {
                return self.open(Place::Metadata);
            }
            // As in the safetensors reference reader, null means none.
            if matches!(event, Event::Null) {
                return Ok(());
            }
            let problem = (Reason::Metadata, "neither null nor an object");
            self.note(problem, self.member.clone());
            return self.skip(&event);
        }
        if object {
            self.entry = Raw::default();
            return self.open(Place::Entry);
        }
        let problem = (Reason::Entry, "tensor entry is not an object");
        self.note(problem, self.member.clone());
        self.skip(&event)
    }

    /// The value of a tensor entry member.
    fn field_value(&mut self, event: Event<'_>) -> Result<(), Failure> {
        let array = matches!(event, Event::BeginArray);
        let numbers = || {
            if array {
                Member::Read(Numbers::default())
            } else {
                Member::Wrong
            }
        };
        match self.field {
            Field::Dtype => {
                self.entry.dtype = match &event {
                    Event::String(name) => {
                        let known = std::str::from_utf8(name).ok().and_then(Dtype::parse);
                        Member::Read(match known {
                            Some(dtype) => Ok(dtype),
                            None => Err(self.keep(name)?),
                        })
                    }
                    _ => Member::Wrong,
                };
            }
            Field::Shape => {
                self.entry.shape = numbers();
                if array {
                    return self.open(Place::Shape);
                }
            }
            Field::Offsets => {
                self.entry.offsets = numbers();
                if array {
                    return self.open(Place::Offsets);
                }
            }
            Field::Other => {}
        }
        self.skip(&event)
    }

    fn end_entry(&mut self) -> Result<(), Failure> {
        match entry::check(&self.entry, self.payload) {
            Ok(entry) => {
                self.tensors.try_reserve(1)?;
                self.tensors.push((self.member.clone(), entry));
            }
            Err(problem) => {
                let dtype = match &self.entry.dtype {
                    Member::Read(Err(text)) if problem.0 == Reason::Dtype => Some(text.clone()),
                    _ => None,
                };
                self.note_dtype(problem, self.member.clone(), dtype);
            }
        }
        Ok(())
    }

    /// The header the walk read, once the parser accepted all of it.
    fn finish(self) -> Result<Header, Failure> {
        if let Some(fault) = &self.fault {
            let name = &self.text[fault.name.clone()];
            return match &fault.dtype {
                Some(dtype) => {
                    let dtype = &self.text[dtype.clone()];
                    reject(fault.reason, name, &[fault.message, " \"", dtype, "\""])
                }
                None => reject(fault.reason, name, &[fault.message]),
            };
        }
        if self.tensors.is_empty() {
            return reject(Reason::Empty, "", &["no tensors"]);
        }
        // Tensors without elements hold no bytes, so only the others tile
        // the payload.
        let mut extents = Vec::new();
        extents.try_reserve_exact(self.tensors.len())?;
        extents.extend(
            self.tensors
                .iter()
                .enumerate()
                .filter(|(_, (_, entry))| entry.begin != entry.end)
                .map(|(index, (_, entry))| (entry.begin, entry.end, index)),
        );
        let name = |index: usize| &self.text[self.tensors[index].0.clone()];
        match layout::tile(&mut extents, self.payload) {
            Ok(()) => {}
            Err(Untiled::Overlap { tensor, other }) => {
                let message = ["data overlaps that of ", name(other)];
                return reject(Reason::Overlap, name(tensor), &message);
            }
            Err(Untiled::Gap(tensor)) => {
                let message = ["data begins after a gap in the payload"];
                return reject(Reason::Gap, name(tensor), &message);
            }
            Err(Untiled::Trailing) => {
                let message = ["payload continues after the last tensor"];
                return reject(Reason::Trailing, "", &message);
            }
        }
        // As in safetensors 0.8, a tensor without elements lies between the
        // others or at either end of the payload, not inside one's data.
        let empty = self
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, (_, entry))| entry.begin == entry.end);
        for (index, (_, entry)) in empty {
            if let Some(other) = layout::holder(&extents, entry.begin) {
                let message = ["offset lies inside the data of ", name(other)];
                return reject(Reason::Overlap, name(index), &message);
            }
        }
        Ok(Header {
            text: self.text,
            tensors: self.tensors,
            metadata: self.metadata,
        })
    }
}

impl Sink for Walk {
    fn event(&mut self, event: Event<'_>) -> Result<(), Stopped> {
        self.take(event).map_err(|failure| {
            self.failure = Some(failure);
            Stopped
        })
    }
}

/// Parses and checks the header bytes that follow the length prefix of a
/// file of `file_size` bytes: the whole header must be JSON, then each
/// member is checked in order, then the payload layout.
pub fn parse(header: &[u8], file_size: u64) -> Result<Header, Failure> {
    let length = u64::try_from(header.len()).map_err(|_| Failure::Internal)?;
    if let Err((reason, message)) = layout::check_length(length, file_size) {
        return reject(reason, "", &[message]);
    }
    let mut walk = Walk::new(file_size - LENGTH_BYTES - length)?;
    match gufo_json::parse(header, &mut walk) {
        Ok(()) => walk.finish(),
        Err(gufo_json::Failure::Rejected(reason)) => {
            reject(Reason::Json, "", &["not JSON: ", reason])
        }
        Err(gufo_json::Failure::NoMemory) => Err(Failure::NoMemory),
        Err(gufo_json::Failure::Stopped) => Err(walk.failure.unwrap_or(Failure::Internal)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Checks `header` as the header of a file with `payload` bytes after it.
    fn check(header: &str, payload: u64) -> Result<Header, Failure> {
        parse(
            header.as_bytes(),
            LENGTH_BYTES + header.len() as u64 + payload,
        )
    }

    fn rejection(header: &str, payload: u64) -> (Reason, String, String) {
        match check(header, payload) {
            Err(Failure::Rejected(rejection)) => {
                (rejection.reason, rejection.name, rejection.message)
            }
            other => panic!("{header}: {other:?}"),
        }
    }

    const TWO: &str = r#"{"__metadata__":{"format":"pt","b":"2"},"b":{"dtype":"F32","shape":[2],"data_offsets":[2,10]},"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;

    #[test]
    fn reads_tensors_and_metadata_in_header_order() {
        let header = check(TWO, 10).unwrap();
        let tensors: Vec<_> = header
            .tensors()
            .map(|(name, entry)| {
                (
                    name,
                    entry.dtype,
                    entry.shape().to_vec(),
                    entry.begin,
                    entry.end,
                )
            })
            .collect();
        assert_eq!(
            tensors,
            [
                ("b", Dtype::F32, vec![2], 2, 10),
                ("a", Dtype::U8, vec![2], 0, 2)
            ]
        );
        let metadata: Vec<_> = header.metadata().collect();
        assert_eq!(metadata, [("format", "pt"), ("b", "2")]);
        let none = r#"{"__metadata__":null,"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        assert_eq!(check(none, 1).unwrap().metadata().count(), 0);
        // Padding with spaces after the object, as writers align the payload.
        let padded = format!("{TWO}   ");
        assert!(check(&padded, 10).is_ok());
        // Without metadata, with escaped names, and with members nothing reads.
        let header = check(
            r#"{"té":{"x":{"dtype":[1]},"dtype":"BOOL","y":[[1],{}],"shape":[],"data_offsets":[0,1]}}"#,
            1,
        )
        .unwrap();
        assert_eq!(header.metadata().count(), 0);
        assert_eq!(header.tensors().next().unwrap().0, "té");
    }

    #[test]
    fn rejects_bad_headers() {
        assert_eq!(rejection("", 1).0, Reason::Length);
        let not_json = rejection(r#"{"a":"#, 0);
        assert_eq!(not_json.0, Reason::Json);
        assert!(not_json.2.starts_with("not JSON: "), "{not_json:?}");
        for header in ["[]", "[{}]", "1", "\"x\"", "null"] {
            assert_eq!(
                rejection(header, 0),
                (
                    Reason::NotObject,
                    String::new(),
                    "not a JSON object".to_owned()
                ),
                "{header}"
            );
        }
        assert_eq!(
            rejection("{}", 0),
            (Reason::Empty, String::new(), "no tensors".to_owned())
        );
        assert_eq!(rejection(r#"{"__metadata__":{}}"#, 0).0, Reason::Empty);
        let duplicate = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        assert_eq!(rejection(duplicate, 2).0, Reason::Json);
        // A JSON error anywhere wins over a bad entry before it.
        assert_eq!(rejection(r#"{"a":1,"b":}"#, 0).0, Reason::Json);
    }

    #[test]
    fn rejects_entries_that_are_not_objects() {
        for value in ["[]", "[{}]", "1", "\"x\"", "null", "true"] {
            let header = format!(r#"{{"t":{value}}}"#);
            assert_eq!(
                rejection(&header, 0),
                (
                    Reason::Entry,
                    "t".to_owned(),
                    "tensor entry is not an object".to_owned()
                )
            );
        }
    }

    #[test]
    fn rejects_bad_metadata() {
        let entry = r#""t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}"#;
        let whole = "neither null nor an object";
        let value = "metadata value is not a string";
        for (metadata, name, message) in [
            ("[]", METADATA, whole),
            ("\"x\"", METADATA, whole),
            (r#"{"k":"v","n":1}"#, "n", value),
            (r#"{"o":{}}"#, "o", value),
            (r#"{"a":["x"],"b":2}"#, "a", value),
        ] {
            let header = format!(r#"{{"__metadata__":{metadata},{entry}}}"#);
            assert_eq!(
                rejection(&header, 1),
                (Reason::Metadata, name.to_owned(), message.to_owned())
            );
        }
    }

    #[test]
    fn names_the_failing_entry() {
        let header = r#"{"good":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"bad":{"dtype":"Q4","shape":[1],"data_offsets":[1,2]}}"#;
        assert_eq!(
            rejection(header, 2),
            (
                Reason::Dtype,
                "bad".to_owned(),
                r#"unsupported dtype "Q4""#.to_owned()
            )
        );
        let missing = r#"{"t":{"shape":[1],"data_offsets":[0,1]}}"#;
        assert_eq!(
            rejection(missing, 1),
            (Reason::Dtype, "t".to_owned(), "missing dtype".to_owned())
        );
        // The first fault in header order wins.
        let two = r#"{"a":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]},"__metadata__":[]}"#;
        let (reason, name, _) = rejection(two, 1);
        assert_eq!((reason, name.as_str()), (Reason::Dimension, "a"));
        let gap = r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"y":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#;
        assert_eq!(
            rejection(gap, 3),
            (
                Reason::Gap,
                "y".to_owned(),
                "data begins after a gap in the payload".to_owned()
            )
        );
        assert_eq!(rejection(gap, 4).0, Reason::Gap);
        let overlap = r#"{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        assert_eq!(
            rejection(overlap, 2),
            (
                Reason::Overlap,
                "y".to_owned(),
                "data overlaps that of x".to_owned()
            )
        );
        let trailing = r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        assert_eq!(
            rejection(trailing, 2),
            (
                Reason::Trailing,
                String::new(),
                "payload continues after the last tensor".to_owned()
            )
        );
    }

    #[test]
    fn places_tensors_without_elements_between_the_others() {
        let data = r#""a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}"#;
        let empty =
            |at: u64| format!(r#""e":{{"dtype":"F32","shape":[0,4],"data_offsets":[{at},{at}]}}"#);
        // Before, between and after the tensors with data, and sharing an
        // offset with another empty tensor.
        for at in [0, 4, 8] {
            let header = format!("{{{},{data}}}", empty(at));
            let read = check(&header, 8).unwrap();
            let (name, entry) = read.tensors().next().unwrap();
            assert_eq!((name, entry.begin, entry.end), ("e", at, at));
            assert_eq!(read.tensors().count(), 3);
        }
        // Inside the data of one.
        for (at, other) in [(1, "a"), (3, "a"), (6, "b")] {
            let header = format!("{{{data},{}}}", empty(at));
            let message = format!("offset lies inside the data of {other}");
            assert_eq!(
                rejection(&header, 8),
                (Reason::Overlap, "e".to_owned(), message)
            );
        }
        let two = format!(
            r#"{{{},{data},"f":{{"dtype":"U8","shape":[0],"data_offsets":[4,4]}}}}"#,
            empty(4)
        );
        assert_eq!(check(&two, 8).unwrap().tensors().count(), 4);
        // Past the payload.
        let past = format!("{{{data},{}}}", empty(9));
        let (reason, name, _) = rejection(&past, 8);
        assert_eq!((reason, name.as_str()), (Reason::Extent, "e"));
        // Only tensors without elements: the payload must be empty.
        let only = format!("{{{}}}", empty(0));
        assert_eq!(check(&only, 0).unwrap().tensors().count(), 1);
        assert_eq!(rejection(&only, 1).0, Reason::Trailing);
        // The others still tile the payload.
        let gap = format!(
            r#"{{{},"a":{{"dtype":"U8","shape":[4],"data_offsets":[1,5]}}}}"#,
            empty(0)
        );
        let (reason, name, _) = rejection(&gap, 5);
        assert_eq!((reason, name.as_str()), (Reason::Gap, "a"));
    }

    #[test]
    fn checks_the_file_size() {
        let rejected = |size| match parse(b"{}", size) {
            Err(Failure::Rejected(rejection)) => (rejection.reason, rejection.message),
            other => panic!("{other:?}"),
        };
        assert_eq!(
            rejected(9),
            (Reason::Length, "runs past the end of the file".to_owned())
        );
        assert_eq!(
            rejected(3),
            (
                Reason::File,
                "shorter than the 8-byte header length".to_owned()
            )
        );
    }

    #[test]
    fn reads_deep_values() {
        // Containers nested to the parser's limit, inside a member nothing
        // reads, need no more room than the walk reserved.
        let depth = MAX_DEPTH - 3;
        let deep = format!("{}{}", "[".repeat(depth), "]".repeat(depth));
        let header =
            format!(r#"{{"t":{{"x":{deep},"dtype":"U8","shape":[1],"data_offsets":[0,1]}}}}"#);
        assert!(check(&header, 1).is_ok());
    }
}
