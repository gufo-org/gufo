//! The frames a WebSocket client sends (RFC 6455, section 5): masked, at most
//! `max_message` bytes per message, text messages and close reasons in UTF-8.

/// A frame that breaks the protocol.
pub const PROTOCOL_ERROR: u16 = 1002;
/// A text message or close reason that is not UTF-8.
pub const INVALID_DATA: u16 = 1007;
/// A frame or message longer than `max_message`.
pub const TOO_BIG: u16 = 1009;
/// The payload could not be allocated.
pub const INTERNAL_ERROR: u16 = 1011;

/// Buffers above this capacity are released after use rather than kept for
/// the next message, so an idle connection holds little memory after a large
/// message.
const KEEP_CAPACITY: usize = 64 * 1024;

/// What `Reader::next` read.
#[derive(Debug, PartialEq, Eq)]
pub enum Event<'a> {
    /// `read` failed before a whole frame arrived.
    End,
    /// A complete text message.
    Message(&'a [u8]),
    /// A ping, to answer with a pong carrying its payload.
    Ping(&'a [u8]),
    /// The client's close frame, whose payload the reply echoes.
    Close(&'a [u8]),
    /// A protocol violation, to answer with a close frame with this code.
    Error(u16),
}

/// `Event` without its payload, while `Reader::next` still reads.
enum Step {
    End,
    Message,
    Ping,
    Close,
    Error(u16),
}

pub struct Reader {
    max_message: usize,
    /// The message being assembled from fragments.
    message: Vec<u8>,
    /// Whether the last data frame was not final.
    fragmented: bool,
    /// Whether `message` was handed out by the last call.
    delivered: bool,
    payload: Vec<u8>,
}

impl Reader {
    pub fn new(max_message: usize) -> Self {
        Self {
            max_message,
            message: Vec::new(),
            fragmented: false,
            delivered: false,
            payload: Vec::new(),
        }
    }

    /// Reads frames through `read`, which fills its whole buffer or returns
    /// false, until a message, a ping, a close frame, an error or the end.
    /// Pongs are skipped and fragments joined. After `Close`, `Error` or
    /// `End`, the connection is over.
    pub fn next(&mut self, read: &mut dyn FnMut(&mut [u8]) -> bool) -> Event<'_> {
        if std::mem::take(&mut self.delivered) {
            recycle(&mut self.message);
        }
        recycle(&mut self.payload);
        match self.step(read) {
            Step::End => Event::End,
            Step::Message => {
                self.delivered = true;
                Event::Message(&self.message)
            }
            Step::Ping => Event::Ping(&self.payload),
            Step::Close => Event::Close(&self.payload),
            Step::Error(code) => Event::Error(code),
        }
    }

    fn step(&mut self, read: &mut dyn FnMut(&mut [u8]) -> bool) -> Step {
        loop {
            let mut header = [0u8; 2];
            if !read(&mut header) {
                return Step::End;
            }
            let last = header[0] & 0x80 != 0;
            let opcode = header[0] & 0x0f;
            let control = opcode >= 8;
            // No extension bits, a mask, a known opcode, unfragmented control.
            if header[0] & 0x70 != 0
                || header[1] & 0x80 == 0
                || !matches!(opcode, 0 | 1 | 8 | 9 | 10)
                || (control && !last)
            {
                return Step::Error(PROTOCOL_ERROR);
            }
            let mut length = u64::from(header[1] & 0x7f);
            if length >= 126 {
                if control {
                    return Step::Error(PROTOCOL_ERROR);
                }
                let count = if length == 126 { 2 } else { 8 };
                let mut extended = [0u8; 8];
                if !read(&mut extended[..count]) {
                    return Step::End;
                }
                length = extended[..count]
                    .iter()
                    .fold(0, |length, &byte| length << 8 | u64::from(byte));
                // The shortest encoding, and a clear top bit.
                if (count == 8 && extended[0] & 0x80 != 0)
                    || (count == 2 && length < 126)
                    || (count == 8 && length <= 65_535)
                {
                    return Step::Error(PROTOCOL_ERROR);
                }
            }
            let room = self.max_message - self.message.len();
            let max = if control { self.max_message } else { room };
            let Some(length) = usize::try_from(length).ok().filter(|&length| length <= max) else {
                return Step::Error(TOO_BIG);
            };
            let mut mask = [0u8; 4];
            if !read(&mut mask) {
                return Step::End;
            }
            self.payload.clear();
            if self.payload.try_reserve(length).is_err() {
                return Step::Error(INTERNAL_ERROR);
            }
            self.payload.resize(length, 0);
            if !read(&mut self.payload) {
                return Step::End;
            }
            for chunk in self.payload.chunks_mut(4) {
                for (byte, mask) in chunk.iter_mut().zip(mask) {
                    *byte ^= mask;
                }
            }
            match opcode {
                8 => return close_step(&self.payload),
                9 => return Step::Ping,
                10 => continue,
                _ => {}
            }
            // A continuation continues a fragmented message; text starts one.
            if (opcode == 0) != self.fragmented {
                return Step::Error(PROTOCOL_ERROR);
            }
            if self.message.is_empty() {
                // The first or only frame becomes the message without a copy.
                std::mem::swap(&mut self.message, &mut self.payload);
            } else {
                if self.message.try_reserve(self.payload.len()).is_err() {
                    return Step::Error(INTERNAL_ERROR);
                }
                self.message.extend_from_slice(&self.payload);
            }
            self.fragmented = !last;
            if !last {
                continue;
            }
            if std::str::from_utf8(&self.message).is_err() {
                return Step::Error(INVALID_DATA);
            }
            return Step::Message;
        }
    }
}

/// Empties `buffer`, releasing its memory above `KEEP_CAPACITY`.
fn recycle(buffer: &mut Vec<u8>) {
    if buffer.capacity() > KEEP_CAPACITY {
        *buffer = Vec::new();
    } else {
        buffer.clear();
    }
}

/// A close frame: no payload, or a code a peer may send and a UTF-8 reason.
fn close_step(payload: &[u8]) -> Step {
    match payload {
        [] => Step::Close,
        [_] => Step::Error(PROTOCOL_ERROR),
        [high, low, reason @ ..] => {
            let code = u16::from_be_bytes([*high, *low]);
            if !(1000..5000).contains(&code)
                || matches!(code, 1004 | 1005 | 1006 | 1015)
                || (1016..3000).contains(&code)
            {
                Step::Error(PROTOCOL_ERROR)
            } else if std::str::from_utf8(reason).is_err() {
                Step::Error(INVALID_DATA)
            } else {
                Step::Close
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A client frame: final or not, an opcode, the payload masked with
    /// `mask`, in the shortest length encoding.
    fn frame(last: bool, opcode: u8, payload: &[u8]) -> Vec<u8> {
        let mask = [0x37, 0xfa, 0x21, 0x3d];
        let mut frame = vec![u8::from(last) << 7 | opcode];
        match payload.len() {
            len @ 0..126 => frame.push(0x80 | len as u8),
            len @ 126..=65_535 => {
                frame.push(0x80 | 126);
                frame.extend((len as u16).to_be_bytes());
            }
            len => {
                frame.push(0x80 | 127);
                frame.extend((len as u64).to_be_bytes());
            }
        }
        frame.extend(mask);
        frame.extend(payload.iter().zip(mask.iter().cycle()).map(|(b, m)| b ^ m));
        frame
    }

    /// Every event of `stream` up to the first that ends the connection.
    fn events(stream: &[u8], max_message: usize) -> Vec<String> {
        let mut reader = Reader::new(max_message);
        let mut offset = 0;
        let mut read = |buffer: &mut [u8]| {
            let Some(bytes) = stream.get(offset..offset + buffer.len()) else {
                return false;
            };
            buffer.copy_from_slice(bytes);
            offset += buffer.len();
            true
        };
        let mut seen = Vec::new();
        loop {
            let event = reader.next(&mut read);
            let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
            let (name, over) = match event {
                Event::End => ("end".to_string(), true),
                Event::Message(bytes) => (format!("message {}", text(bytes)), false),
                Event::Ping(bytes) => (format!("ping {}", text(bytes)), false),
                Event::Close(bytes) => (format!("close {bytes:?}"), true),
                Event::Error(code) => (format!("error {code}"), true),
            };
            seen.push(name);
            if over {
                return seen;
            }
        }
    }

    #[test]
    fn messages_fragments_and_control_frames() {
        let mut stream = frame(true, 1, b"hello");
        stream.extend(frame(false, 1, b"wor"));
        stream.extend(frame(true, 9, b"p"));
        stream.extend(frame(true, 10, b"ignored"));
        stream.extend(frame(true, 0, b"ld"));
        stream.extend(frame(true, 1, &[b'x'; 300]));
        stream.extend(frame(true, 8, &[0x03, 0xe8, b'o', b'k']));
        assert_eq!(
            events(&stream, 1 << 20),
            [
                "message hello",
                "ping p",
                "message world",
                &format!("message {}", "x".repeat(300)),
                "close [3, 232, 111, 107]",
            ]
        );
    }

    #[test]
    fn violations_close_with_their_codes() {
        let error = |stream: Vec<u8>| events(&stream, 100).pop().unwrap();
        assert_eq!(error(vec![0x81 | 0x40, 0x80, 0, 0, 0, 0]), "error 1002");
        assert_eq!(error(vec![0x81, 0x00]), "error 1002");
        assert_eq!(error(vec![0x83, 0x80, 0, 0, 0, 0]), "error 1002");
        assert_eq!(error(vec![0x09, 0x80, 0, 0, 0, 0]), "error 1002");
        assert_eq!(error(vec![0x89, 0x80 | 126, 0, 126]), "error 1002");
        assert_eq!(error(vec![0x81, 0x80 | 126, 0, 125]), "error 1002");
        assert_eq!(error(frame(true, 0, b"x")), "error 1002");
        assert_eq!(error(frame(true, 1, &[b'x'; 101])), "error 1009");
        assert_eq!(error(frame(true, 1, &[0xc3])), "error 1007");
        assert_eq!(error(frame(true, 8, &[3])), "error 1002");
        assert_eq!(error(frame(true, 8, &[0x03, 0xed])), "error 1002");
        assert_eq!(error(frame(true, 8, &[0x03, 0xe8, 0xff])), "error 1007");
        let mut stream = frame(false, 1, &[b'x'; 60]);
        stream.extend(frame(true, 0, &[b'x'; 41]));
        assert_eq!(error(stream), "error 1009");
        // A character split across fragments is whole once joined.
        let mut stream = frame(false, 1, &[0xc3]);
        stream.extend(frame(true, 0, &[0xa9]));
        assert_eq!(events(&stream, 100)[0], "message é");
    }

    #[test]
    fn truncation_ends_at_every_byte() {
        let stream = frame(true, 1, &[b'y'; 200]);
        for len in 0..stream.len() {
            assert_eq!(events(&stream[..len], 1000), ["end"]);
        }
    }

    #[test]
    fn large_buffers_are_released_after_their_message() {
        let large = vec![b'a'; 3 * KEEP_CAPACITY];
        let mut stream = frame(true, 1, &large);
        stream.extend(frame(false, 1, &large[..KEEP_CAPACITY + 1]));
        stream.extend(frame(true, 0, &large[..2 * KEEP_CAPACITY - 1]));
        stream.extend(frame(true, 1, b"small"));
        let mut reader = Reader::new(1 << 20);
        let mut offset = 0;
        let mut read = |buffer: &mut [u8]| {
            buffer.copy_from_slice(&stream[offset..offset + buffer.len()]);
            offset += buffer.len();
            true
        };
        assert_eq!(reader.next(&mut read), Event::Message(&large));
        // The fragments join to the same text.
        assert_eq!(reader.next(&mut read), Event::Message(&large));
        assert_eq!(reader.next(&mut read), Event::Message(b"small"));
        assert!(reader.message.capacity() <= KEEP_CAPACITY);
        assert!(reader.payload.capacity() <= KEEP_CAPACITY);
    }
}
