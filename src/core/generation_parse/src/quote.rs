//! Request-owned quoting state, as C++ `QuoteTracker` kept it. Completed
//! inline spans across nonblank lines and completed fences protect
//! documentation. Markers in unfinished spans are held during streaming
//! until a closing delimiter or the final parse decides.

use crate::text::NPOS;

#[derive(Debug, Clone, Copy)]
struct Span {
    begin: usize,
    end: usize,
}

#[derive(Debug, Clone)]
pub struct QuoteTracker {
    /// Bytes read so far, counting the blank origin of `reset`.
    len: usize,
    spans: Vec<Span>,
    unfinished_inline: Vec<Span>,
    bytes_scanned: usize,
    first_nonblank: usize,
    run_begin: usize,
    run_size: usize,
    run_char: u8,
    inline_begin: usize,
    inline_size: usize,
    fence_begin: usize,
    fence_size: usize,
    fence_char: u8,
    close_begin: usize,
    closing_fence: bool,
}

impl Default for QuoteTracker {
    fn default() -> Self {
        Self {
            len: 0,
            spans: Vec::new(),
            unfinished_inline: Vec::new(),
            bytes_scanned: 0,
            first_nonblank: NPOS,
            run_begin: 0,
            run_size: 0,
            run_char: 0,
            inline_begin: 0,
            inline_size: 0,
            fence_begin: 0,
            fence_size: 0,
            fence_char: 0,
            close_begin: 0,
            closing_fence: false,
        }
    }
}

impl QuoteTracker {
    /// Restarts with `text`, whose bytes before `origin` read as blanks: the
    /// quoting scope starts there.
    pub fn reset(&mut self, text: &[u8], origin: usize) {
        let origin = origin.min(text.len());
        let scanned = self.bytes_scanned;
        let mut spans = std::mem::take(&mut self.spans);
        let mut unfinished = std::mem::take(&mut self.unfinished_inline);
        spans.clear();
        unfinished.clear();
        *self = Self {
            len: origin,
            spans,
            unfinished_inline: unfinished,
            bytes_scanned: scanned,
            ..Self::default()
        };
        self.append(&text[origin..]);
    }

    pub fn append(&mut self, piece: &[u8]) {
        for &byte in piece {
            self.read(self.len, byte);
            self.len += 1;
        }
        self.bytes_scanned += piece.len();
    }

    pub fn quoted_at(&self, position: usize) -> bool {
        let next = self.spans.partition_point(|span| span.begin <= position);
        if next > 0 && position < self.spans[next - 1].end {
            return true;
        }
        // A closing delimiter can be the final bytes, without a following
        // newline or ordinary character to flush the pending run. This is
        // provisional: appending more text can make the last line cease to be
        // a closing fence.
        if self.fence_size > 0 {
            if self.closing_fence && position >= self.fence_begin && position < self.close_begin {
                return true;
            }
            if self.run_size >= self.fence_size
                && self.run_char == self.fence_char
                && self.at_line_start(self.run_begin)
                && position >= self.fence_begin
                && position < self.run_begin
            {
                return true;
            }
        } else if self.inline_size > 0
            && self.run_char == b'`'
            && self.run_size == self.inline_size
            && position >= self.inline_begin
            && position < self.run_begin
        {
            return true;
        }
        false
    }

    pub fn unclosed_at(&self, position: usize) -> bool {
        if self.quoted_at(position) {
            return false;
        }
        if (self.fence_size > 0 && position >= self.fence_begin)
            || (self.inline_size > 0 && position >= self.inline_begin)
        {
            return true;
        }
        let next = self
            .unfinished_inline
            .partition_point(|span| span.begin <= position);
        next > 0 && position < self.unfinished_inline[next - 1].end
    }

    /// Bytes read since the tracker was made, across resets.
    pub fn bytes_scanned(&self) -> usize {
        self.bytes_scanned
    }

    fn at_line_start(&self, position: usize) -> bool {
        self.first_nonblank == position
    }

    fn flush_run(&mut self) {
        if self.run_size == 0 {
            return;
        }
        if self.fence_size > 0 {
            if self.run_char == self.fence_char
                && self.run_size >= self.fence_size
                && self.at_line_start(self.run_begin)
            {
                self.closing_fence = true;
                self.close_begin = self.run_begin;
            }
        } else if self.inline_size == 0 && self.run_size >= 3 && self.at_line_start(self.run_begin)
        {
            self.fence_char = self.run_char;
            self.fence_size = self.run_size;
            self.fence_begin = self.run_begin + self.run_size;
        } else if self.run_char == b'`' {
            if self.inline_size == 0 {
                self.inline_size = self.run_size;
                self.inline_begin = self.run_begin + self.run_size;
            } else if self.run_size == self.inline_size {
                self.spans.push(Span {
                    begin: self.inline_begin,
                    end: self.run_begin,
                });
                self.inline_size = 0;
            }
        }
        self.run_size = 0;
    }

    fn read(&mut self, cursor: usize, byte: u8) {
        if self.first_nonblank == NPOS && !matches!(byte, b' ' | b'\t' | b'\r' | b'\n') {
            self.first_nonblank = cursor;
        }
        if byte == b'`' || byte == b'~' {
            if self.run_size != 0 && self.run_char != byte {
                self.flush_run();
            }
            if self.run_size == 0 {
                self.run_begin = cursor;
                self.run_char = byte;
            }
            self.run_size += 1;
            return;
        }
        self.flush_run();
        if byte == b'\n' {
            if self.fence_size > 0 && self.closing_fence {
                self.spans.push(Span {
                    begin: self.fence_begin,
                    end: self.close_begin,
                });
                self.fence_size = 0;
                self.fence_char = 0;
            }
            if self.inline_size > 0 && self.first_nonblank == NPOS {
                // A blank line ends an unfinished inline span. Retain its
                // range so possible calls still require the schema-checked
                // fallback.
                self.unfinished_inline.push(Span {
                    begin: self.inline_begin,
                    end: cursor,
                });
                self.inline_size = 0;
            }
            self.closing_fence = false;
            self.first_nonblank = NPOS;
        } else if byte != b' ' && byte != b'\t' && byte != b'\r' {
            self.closing_fence = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::QuoteTracker;
    use crate::text::find;

    fn tracker(text: &str) -> QuoteTracker {
        let mut tracker = QuoteTracker::default();
        tracker.reset(text.as_bytes(), 0);
        tracker
    }

    #[test]
    fn completed_spans_quote_and_bytewise_reading_agrees() {
        for (text, quoted) in [
            ("plain <marker>", false),
            ("`<marker>`", true),
            ("`<marker>", false),
            ("`<marker>\nnext", false),
            ("`before\n<marker>\nafter`", true),
            ("``before\n<marker>\nafter``", true),
            ("`before\n\n<marker>", false),
            ("`<marker>\n\nlater`", false),
            ("`before\n \t\r\n<marker>", false),
            ("`before <think>\n<marker>`", true),
            ("``<marker>``", true),
            ("``<marker>`", false),
            ("```python\n<marker>\n```", true),
            ("```python\n<marker>\n```\n", true),
            ("```python\n<marker>", false),
            ("~~~xml\n<marker>\n~~~", true),
            ("~~~xml\n<marker>\n```", false),
            ("````\n<marker>\n```", false),
            ("```\n<marker>\n```more text", false),
            ("```\n</think>\n<marker>", false),
            ("```\n</think>\n`<marker>`", false),
            ("`literal` and <marker>", false),
            ("```\nexample\n```\n<marker>", false),
        ] {
            let marker = find(text.as_bytes(), b"<marker>", 0);
            assert_eq!(tracker(text).quoted_at(marker), quoted, "{text}");
            let mut incremental = QuoteTracker::default();
            for byte in text.as_bytes() {
                incremental.append(std::slice::from_ref(byte));
            }
            assert_eq!(incremental.quoted_at(marker), quoted, "bytewise {text}");
        }
    }

    #[test]
    fn unfinished_fences_and_inline_spans_decide_later() {
        let mut quotes = QuoteTracker::default();
        quotes.append(b"```xml\n<marker>\n");
        assert!(!quotes.quoted_at(7));
        quotes.append(b"```\n");
        assert!(quotes.quoted_at(7));
        quotes.reset(b"`<marker>\n\n", 0);
        assert!(!quotes.quoted_at(1) && quotes.unclosed_at(1));
        quotes.reset(b"`<marker>`", 0);
        assert!(quotes.quoted_at(1) && !quotes.unclosed_at(1));
        quotes.reset(b"plain <marker>", 0);
        assert!(!quotes.quoted_at(6));
    }

    #[test]
    fn an_explicit_phase_boundary_starts_a_fresh_scope() {
        let text = b"```reasoning\n</think>\n<marker>";
        let mut quotes = QuoteTracker::default();
        quotes.reset(text, find(text, b"</think>", 0) + b"</think>".len());
        assert!(!quotes.quoted_at(find(text, b"<marker>", 0)));
        quotes.reset(b"`literal </think>\n<marker>`", 0);
        assert!(quotes.quoted_at(18));
    }

    #[test]
    fn reads_each_byte_once() {
        let mut quotes = QuoteTracker::default();
        let before = quotes.bytes_scanned();
        let units = 4000;
        for i in 0..units {
            quotes.append(b"<a>");
            assert!(!quotes.quoted_at(i * 3));
        }
        for i in (0..units).rev() {
            assert!(!quotes.quoted_at(i * 3));
        }
        assert_eq!(quotes.bytes_scanned() - before, units * 3);
    }
}
