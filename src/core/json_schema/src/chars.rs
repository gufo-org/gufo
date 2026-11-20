//! Sets of codepoints as sorted, disjoint, non-adjacent inclusive ranges, as
//! ICU's `UnicodeSet` keeps them, plus the multi-codepoint strings of
//! properties such as `\p{RGI_Emoji_Flag_Sequence}`: two sets are equal
//! exactly when their ranges and strings are. Strings never match a single
//! codepoint, but they keep a set nonempty and distinct, as in ICU.

use std::collections::BTreeSet;

pub const MAX_CODEPOINT: u32 = 0x10ffff;
pub const SURROGATES: (u32, u32) = (0xd800, 0xdfff);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Chars {
    ranges: Vec<(u32, u32)>,
    strings: BTreeSet<Vec<u32>>,
}

impl Chars {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn range(first: u32, last: u32) -> Self {
        let mut chars = Self::new();
        chars.add(first, last);
        chars
    }

    pub fn single(cp: u32) -> Self {
        Self::range(cp, cp)
    }

    /// Every Unicode scalar value: all codepoints except the surrogates.
    pub fn scalars() -> Self {
        let mut chars = Self::range(0, MAX_CODEPOINT);
        chars.remove(SURROGATES.0, SURROGATES.1);
        chars
    }

    pub fn ranges(&self) -> &[(u32, u32)] {
        &self.ranges
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty() && self.strings.is_empty()
    }

    /// The number of codepoints and strings, as ICU's `size()` counts them.
    pub fn size(&self) -> u64 {
        self.ranges
            .iter()
            .map(|&(first, last)| u64::from(last - first) + 1)
            .sum::<u64>()
            + self.strings.len() as u64
    }

    /// Adds a string, or its codepoint when it has one, as ICU does.
    pub fn add_string(&mut self, string: &[u32]) {
        match string {
            [cp] => self.add(*cp, *cp),
            _ => {
                self.strings.insert(string.to_vec());
            }
        }
    }

    /// The smallest codepoint.
    pub fn first(&self) -> Option<u32> {
        self.ranges.first().map(|&(first, _)| first)
    }

    pub fn contains(&self, cp: u32) -> bool {
        let index = self.ranges.partition_point(|&(_, last)| last < cp);
        self.ranges
            .get(index)
            .is_some_and(|&(first, _)| first <= cp)
    }

    /// Adds `[first, last]`, clamped to the codepoint range. An empty range
    /// adds nothing.
    pub fn add(&mut self, first: u32, last: u32) {
        let last = last.min(MAX_CODEPOINT);
        if first > last {
            return;
        }
        // ICU reports a set's ranges in order: append without searching.
        match self.ranges.last_mut() {
            None => {
                self.ranges.push((first, last));
                return;
            }
            Some(back) if first > back.1.saturating_add(1) => {
                self.ranges.push((first, last));
                return;
            }
            Some(back) if first >= back.0 => {
                back.1 = back.1.max(last);
                return;
            }
            _ => {}
        }
        // Ranges that overlap or touch [first, last] merge into it.
        let begin = self
            .ranges
            .partition_point(|&(_, end)| end.saturating_add(1) < first);
        let end = self
            .ranges
            .partition_point(|&(start, _)| start <= last.saturating_add(1));
        let mut merged = (first, last);
        if begin < end {
            merged.0 = merged.0.min(self.ranges[begin].0);
            merged.1 = merged.1.max(self.ranges[end - 1].1);
        }
        self.ranges.splice(begin..end, [merged]);
    }

    pub fn add_all(&mut self, other: &Chars) {
        for &(first, last) in &other.ranges {
            self.add(first, last);
        }
        self.strings.extend(other.strings.iter().cloned());
    }

    /// Removes `[first, last]`.
    pub fn remove(&mut self, first: u32, last: u32) {
        if first > last {
            return;
        }
        let mut kept = Vec::with_capacity(self.ranges.len() + 1);
        for &(start, end) in &self.ranges {
            if end < first || start > last {
                kept.push((start, end));
                continue;
            }
            if start < first {
                kept.push((start, first - 1));
            }
            if end > last {
                kept.push((last + 1, end));
            }
        }
        self.ranges = kept;
    }

    /// Complements within all codepoints, surrogates included, and keeps the
    /// strings, as ICU does.
    pub fn complement(&mut self) {
        let mut result = Vec::with_capacity(self.ranges.len() + 1);
        let mut next = 0u32;
        for &(first, last) in &self.ranges {
            if first > next {
                result.push((next, first - 1));
            }
            next = last + 1;
        }
        if next <= MAX_CODEPOINT {
            result.push((next, MAX_CODEPOINT));
        }
        self.ranges = result;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_merge_and_split() {
        let mut chars = Chars::range(10, 20);
        chars.add(21, 25);
        chars.add(0, 3);
        assert_eq!(chars.ranges(), [(0, 3), (10, 25)]);
        chars.add(4, 9);
        assert_eq!(chars.ranges(), [(0, 25)]);
        chars.remove(5, 7);
        assert_eq!(chars.ranges(), [(0, 4), (8, 25)]);
        assert!(chars.contains(4) && !chars.contains(5) && chars.contains(25));
        assert_eq!(chars.size(), 23);
        chars.complement();
        assert_eq!(chars.ranges(), [(5, 7), (26, MAX_CODEPOINT)]);
        assert_eq!(Chars::scalars().size(), 0x110000 - 0x800);
        let mut all = Chars::new();
        all.complement();
        assert_eq!(all, Chars::range(0, MAX_CODEPOINT));
        all.complement();
        assert!(all.is_empty());
    }

    #[test]
    fn strings_keep_sets_distinct() {
        let mut flags = Chars::new();
        flags.add_string(&[0x1f1e6, 0x1f1e8]);
        assert!(!flags.is_empty() && flags.first().is_none());
        assert_eq!(flags.size(), 1);
        assert!(!flags.contains(0x1f1e6));
        let mut negated = flags.clone();
        negated.complement();
        assert_ne!(negated, Chars::range(0, MAX_CODEPOINT));
        assert_eq!(negated.size(), 0x110000 + 1);
        let mut one = Chars::new();
        one.add_string(&[0x61]);
        assert_eq!(one, Chars::single(0x61));
    }

    #[test]
    fn append_in_order() {
        let mut chars = Chars::new();
        chars.add(5, 9);
        chars.add(7, 12);
        chars.add(13, 13);
        chars.add(20, 20);
        chars.add(20, 19);
        chars.add(18, 18);
        assert_eq!(chars.ranges(), [(5, 13), (18, 18), (20, 20)]);
        chars.add(0x10fff0, u32::MAX);
        assert_eq!(chars.ranges().last(), Some(&(0x10fff0, MAX_CODEPOINT)));
    }

    #[test]
    fn add_overlapping_many() {
        let mut chars = Chars::new();
        for i in (0..100).step_by(4) {
            chars.add(i, i + 1);
        }
        chars.add(2, 97);
        assert_eq!(chars.ranges(), [(0, 97)]);
    }
}
