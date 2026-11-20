//! JSON Schema `pattern` regular expressions (ECMA-262 syntax, without
//! backreferences, lookbehind or flags) compiled to an immutable Unicode DFA
//! for the intersection of patterns. Every admitted prefix has an accepting
//! continuation within the length bound.
//!
//! The expressions, their ids, the alphabet and the state numbering follow
//! the C++ compiler of Gufo 0.35.0 step for step, so the tables match it.

use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::chars::{Chars, SURROGATES};
use crate::error::{Result, invalid};

pub type Id = u32;
pub const DEAD: Id = u32::MAX;
const INFINITE: Id = u32::MAX;
const MAX_EXPRESSIONS: usize = 32768;
const MAX_STATES: usize = 4096;
const MAX_TRANSITIONS: usize = 262144;
const MAX_DERIVATIVES: usize = 1048576;
const MAX_PATTERN_BYTES: usize = 16384;
const MAX_NESTING: u32 = 32;
const MAX_PROPERTY_NAME: usize = 128;
/// What ICU's `char32At` returns past the end of the pattern.
const END: u32 = 0xffff;

/// Resolves a UnicodeSet pattern such as `[\p{L}]` with the pinned ICU, so
/// property names, aliases and the Unicode version stay ICU's.
pub trait UnicodeSets {
    /// The set, or `None` when ICU rejects the pattern.
    fn resolve(&mut self, pattern: &str) -> Result<Option<Chars>>;
}

fn word(cp: u32) -> bool {
    matches!(cp, 0x61..=0x7a | 0x41..=0x5a | 0x30..=0x39 | 0x5f)
}

fn words() -> Chars {
    let mut chars = Chars::range(u32::from(b'a'), u32::from(b'z'));
    chars.add(u32::from(b'A'), u32::from(b'Z'));
    chars.add(u32::from(b'0'), u32::from(b'9'));
    chars.add(u32::from(b'_'), u32::from(b'_'));
    chars
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Kind {
    Empty,
    Epsilon,
    Chars,
    Start,
    Boundary,
    Or,
    And,
    Not,
    Concat,
    Repeat,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Expression {
    kind: Kind,
    children: Vec<Id>,
    low: Id,
    high: Id,
}

impl Expression {
    fn leaf(kind: Kind) -> Self {
        Self::new(kind, Vec::new(), 0, 0)
    }

    fn new(kind: Kind, children: Vec<Id>, low: Id, high: Id) -> Self {
        Self {
            kind,
            children,
            low,
            high,
        }
    }
}

/// Regex derivatives preserve intersections and lookaheads, unlike testing
/// predicates separately against the already-generated prefix.
struct Expressions {
    maximum: Id,
    nodes: Vec<Expression>,
    minimums: Vec<u64>,
    ids: HashMap<Expression, Id>,
    derivatives: HashMap<(Id, u32, bool, bool), Id>,
    classes: Vec<Chars>,
    class_ids: HashMap<Chars, Id>,
    word_boundaries: bool,
    empty: Id,
    epsilon: Id,
    start: Id,
    all: Id,
}

impl Expressions {
    fn new(maximum: Id) -> Result<Self> {
        let mut e = Self {
            maximum,
            nodes: Vec::new(),
            minimums: Vec::new(),
            ids: HashMap::new(),
            derivatives: HashMap::new(),
            classes: Vec::new(),
            class_ids: HashMap::new(),
            word_boundaries: false,
            empty: 0,
            epsilon: 0,
            start: 0,
            all: DEAD,
        };
        e.empty = e.add(Expression::leaf(Kind::Empty))?;
        e.epsilon = e.add(Expression::leaf(Kind::Epsilon))?;
        e.start = e.add(Expression::leaf(Kind::Start))?;
        let any = e.chars(Chars::scalars())?;
        e.all = e.repeat(any, 0, INFINITE)?;
        Ok(e)
    }

    fn add(&mut self, node: Expression) -> Result<Id> {
        if let Some(&id) = self.ids.get(&node) {
            return Ok(id);
        }
        if self.nodes.len() >= MAX_EXPRESSIONS {
            return invalid("regex expression budget exceeded");
        }
        let infinite = u64::from(INFINITE);
        let minimum = match node.kind {
            Kind::Empty => infinite,
            Kind::Chars => 1,
            Kind::Or => node
                .children
                .iter()
                .map(|&c| self.minimums[c as usize])
                .fold(infinite, u64::min),
            Kind::Concat => node
                .children
                .iter()
                .fold(0, |sum, &c| infinite.min(sum + self.minimums[c as usize])),
            Kind::And => node
                .children
                .iter()
                .map(|&c| self.minimums[c as usize])
                .fold(0, u64::max),
            Kind::Repeat => {
                infinite.min(self.minimums[node.children[0] as usize] * u64::from(node.low))
            }
            _ => 0,
        };
        if !self.nodes.is_empty() && minimum > u64::from(self.maximum) {
            return Ok(self.empty);
        }
        let id = self.nodes.len() as Id;
        self.ids.insert(node.clone(), id);
        self.nodes.push(node);
        self.minimums.push(minimum);
        Ok(id)
    }

    fn boundary(&mut self, positive: bool) -> Result<Id> {
        self.word_boundaries = true;
        // Include word membership in the DFA alphabet.
        self.chars(words())?;
        self.add(Expression::new(
            Kind::Boundary,
            Vec::new(),
            u32::from(positive),
            0,
        ))
    }

    fn chars(&mut self, mut set: Chars) -> Result<Id> {
        set.remove(SURROGATES.0, SURROGATES.1);
        if set.is_empty() {
            return Ok(self.empty);
        }
        let index = match self.class_ids.get(&set) {
            Some(&index) => index,
            None => {
                let index = self.classes.len() as Id;
                self.class_ids.insert(set.clone(), index);
                self.classes.push(set);
                index
            }
        };
        self.add(Expression::new(Kind::Chars, Vec::new(), index, 0))
    }

    fn not(&mut self, child: Id) -> Result<Id> {
        if child == self.empty {
            return Ok(self.all);
        }
        if child == self.all {
            return Ok(self.empty);
        }
        let node = &self.nodes[child as usize];
        if node.kind == Kind::Not {
            return Ok(node.children[0]);
        }
        self.add(Expression::new(Kind::Not, vec![child], 0, 0))
    }

    fn combine(&mut self, kind: Kind, parts: Vec<Id>) -> Result<Id> {
        let mut flat: Vec<Id> = Vec::new();
        for part in parts {
            if kind == Kind::Concat {
                if part == self.empty {
                    return Ok(self.empty);
                }
                if part == self.epsilon {
                    continue;
                }
            } else {
                let (absorbing, identity) = if kind == Kind::Or {
                    (self.all, self.empty)
                } else {
                    (self.empty, self.all)
                };
                if part == absorbing {
                    return Ok(part);
                }
                if part == identity {
                    continue;
                }
            }
            let node = &self.nodes[part as usize];
            if node.kind == kind {
                flat.extend_from_slice(&node.children);
            } else if kind != Kind::Concat || part != self.all || flat.last() != Some(&self.all) {
                flat.push(part);
            }
        }
        if kind != Kind::Concat {
            flat.sort_unstable();
            flat.dedup();
            for &part in &flat {
                let node = &self.nodes[part as usize];
                if node.kind == Kind::Not && flat.binary_search(&node.children[0]).is_ok() {
                    return Ok(if kind == Kind::Or {
                        self.all
                    } else {
                        self.empty
                    });
                }
            }
        }
        match flat.len() {
            0 => Ok(match kind {
                Kind::Or => self.empty,
                Kind::And => self.all,
                _ => self.epsilon,
            }),
            1 => Ok(flat[0]),
            _ => self.add(Expression::new(kind, flat, 0, 0)),
        }
    }

    fn repeat(&mut self, mut child: Id, mut low: Id, mut high: Id) -> Result<Id> {
        if high < low {
            return invalid("invalid regex repetition");
        }
        if high == 0 || child == self.epsilon {
            return Ok(self.epsilon);
        }
        if child == self.empty {
            return Ok(if low != 0 { self.empty } else { self.epsilon });
        }
        if low == 1 && high == 1 {
            return Ok(child);
        }
        // Repetitions of a nullable language need only its nonempty members.
        // Missing mandatory copies can be filled with epsilon.
        if self.nullable(child, false, false, false) {
            let nonempty = self.not(self.epsilon)?;
            child = self.combine(Kind::And, vec![child, nonempty])?;
            low = 0;
        }
        let width = self.minimums[child as usize];
        if width != 0 && high != INFINITE {
            high = u64::from(high).min(u64::from(self.maximum) / width) as Id;
        }
        if high < low {
            return Ok(self.empty);
        }
        self.add(Expression::new(Kind::Repeat, vec![child], low, high))
    }

    fn nullable(&self, id: Id, at_start: bool, previous_word: bool, next_word: bool) -> bool {
        let node = &self.nodes[id as usize];
        let child = |c: &Id| self.nullable(*c, at_start, previous_word, next_word);
        match node.kind {
            Kind::Empty | Kind::Chars => false,
            Kind::Epsilon => true,
            Kind::Start => at_start,
            Kind::Boundary => (previous_word != next_word) == (node.low != 0),
            Kind::Not => !child(&node.children[0]),
            Kind::Or => node.children.iter().any(child),
            Kind::And | Kind::Concat => node.children.iter().all(child),
            Kind::Repeat => node.low == 0,
        }
    }

    fn derive(&mut self, id: Id, cp: u32, at_start: bool, previous_word: bool) -> Result<Id> {
        let key = (id, cp, at_start, previous_word);
        if let Some(&found) = self.derivatives.get(&key) {
            return Ok(found);
        }
        let node = self.nodes[id as usize].clone();
        let result = match node.kind {
            Kind::Empty | Kind::Epsilon | Kind::Start | Kind::Boundary => self.empty,
            Kind::Chars => {
                if self.classes[node.low as usize].contains(cp) {
                    self.epsilon
                } else {
                    self.empty
                }
            }
            Kind::Not => {
                let derivative = self.derive(node.children[0], cp, at_start, previous_word)?;
                self.not(derivative)?
            }
            Kind::Or | Kind::And => {
                let mut parts = Vec::with_capacity(node.children.len());
                for &child in &node.children {
                    parts.push(self.derive(child, cp, at_start, previous_word)?);
                }
                self.combine(node.kind, parts)?
            }
            Kind::Concat => {
                let mut choices = Vec::new();
                for (i, &child) in node.children.iter().enumerate() {
                    let mut suffix = vec![self.derive(child, cp, at_start, previous_word)?];
                    suffix.extend_from_slice(&node.children[i + 1..]);
                    choices.push(self.combine(Kind::Concat, suffix)?);
                    if !self.nullable(child, at_start, previous_word, word(cp)) {
                        break;
                    }
                }
                self.combine(Kind::Or, choices)?
            }
            Kind::Repeat => {
                let child = node.children[0];
                let derivative = self.derive(child, cp, at_start, previous_word)?;
                let high = if node.high == INFINITE {
                    INFINITE
                } else {
                    node.high - 1
                };
                let rest = self.repeat(child, node.low.saturating_sub(1), high)?;
                self.combine(Kind::Concat, vec![derivative, rest])?
            }
        };
        if self.derivatives.len() >= MAX_DERIVATIVES {
            return invalid("regex derivative budget exceeded");
        }
        self.derivatives.insert(key, result);
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SyntaxKind {
    Atom,
    Sequence,
    Alternative,
    Repeat,
    Positive,
    Negative,
    End,
}

#[derive(Debug)]
struct Syntax {
    kind: SyntaxKind,
    atom: Id,
    low: Id,
    high: Id,
    assertion: bool,
    children: Vec<Syntax>,
}

impl Syntax {
    fn new(kind: SyntaxKind) -> Self {
        Self {
            kind,
            atom: 0,
            low: 0,
            high: 0,
            assertion: false,
            children: Vec::new(),
        }
    }
}

struct Parser<'a> {
    e: &'a mut Expressions,
    sets: &'a mut dyn UnicodeSets,
    text: Vec<u32>,
    position: usize,
}

impl<'a> Parser<'a> {
    fn new(e: &'a mut Expressions, sets: &'a mut dyn UnicodeSets, pattern: &[u8]) -> Result<Self> {
        if pattern.len() > MAX_PATTERN_BYTES {
            return invalid("regex exceeds 16384 bytes");
        }
        // ICU decodes ill-formed UTF-8 to U+FFFD per maximal subpart, as this does.
        let decoded = String::from_utf8_lossy(pattern);
        let mut text = Vec::new();
        if text.try_reserve(decoded.len()).is_err() {
            return Err(crate::error::Failure::NoMemory);
        }
        text.extend(decoded.chars().map(u32::from));
        Ok(Self {
            e,
            sets,
            text,
            position: 0,
        })
    }

    fn parse(&mut self) -> Result<Id> {
        let syntax = self.alternatives(0)?;
        if self.position != self.text.len() {
            return invalid("unbalanced regex parentheses");
        }
        let all = self.e.all;
        let body = self.expand(&syntax, all)?;
        self.e.combine(Kind::Concat, vec![all, body])
    }

    fn at(&self, position: usize) -> u32 {
        self.text.get(position).copied().unwrap_or(END)
    }

    fn peek(&self) -> u32 {
        self.at(self.position)
    }

    fn take(&mut self) -> Result<u32> {
        if self.position >= self.text.len() {
            return invalid("truncated regex");
        }
        let cp = self.peek();
        self.position += 1;
        Ok(cp)
    }

    fn is(&self, c: u8) -> bool {
        self.peek() == u32::from(c)
    }

    fn digit(&self) -> bool {
        (u32::from(b'0')..=u32::from(b'9')).contains(&self.peek())
    }

    fn count(&mut self) -> Result<Id> {
        if !self.digit() {
            return invalid("invalid regex repetition");
        }
        let mut value: u64 = 0;
        while self.digit() {
            value = value * 10 + u64::from(self.take()? - u32::from(b'0'));
            if value >= u64::from(INFINITE) {
                return invalid("regex repetition is too large");
            }
        }
        Ok(value as Id)
    }

    fn hex(&mut self, count: u32) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..count {
            let cp = self.take()?;
            let digit = match char::from_u32(cp) {
                Some(c) => c.to_digit(16),
                None => None,
            };
            let Some(digit) = digit else {
                return invalid("invalid regex Unicode escape");
            };
            value = (value << 4) | digit;
        }
        Ok(value)
    }

    fn resolve(&mut self, pattern: &str) -> Result<Chars> {
        match self.sets.resolve(pattern)? {
            Some(chars) => Ok(chars),
            None => invalid("invalid regex Unicode property"),
        }
    }

    fn escape(&mut self, in_class: bool) -> Result<Chars> {
        let escape = self.take()?;
        let mut cp = escape;
        let set = match char::from_u32(escape).unwrap_or('\u{fffd}') {
            'd' | 'D' => Some(Chars::range(u32::from(b'0'), u32::from(b'9'))),
            // ECMA-262 WhiteSpace + LineTerminator, not ICU White_Space.
            's' | 'S' => Some(self.resolve("[\\p{Zs}\\u0009-\\u000d\\u2028\\u2029\\ufeff]")?),
            'w' | 'W' => Some(words()),
            'p' | 'P' => {
                if self.take()? != u32::from(b'{') {
                    return invalid("Unicode property escape requires braces");
                }
                let mut name = String::new();
                while !self.is(b'}') {
                    let c = self.take()?;
                    if c > 127 || name.len() >= MAX_PROPERTY_NAME {
                        return invalid("invalid Unicode property name");
                    }
                    name.push(char::from(c as u8));
                }
                self.position += 1;
                let sign = if escape == u32::from(b'p') { 'p' } else { 'P' };
                Some(self.resolve(&format!("[\\{sign}{{{name}}}]"))?)
            }
            'n' => {
                cp = u32::from(b'\n');
                None
            }
            'r' => {
                cp = u32::from(b'\r');
                None
            }
            't' => {
                cp = u32::from(b'\t');
                None
            }
            'f' => {
                cp = 0x0c;
                None
            }
            'v' => {
                cp = 0x0b;
                None
            }
            'b' => {
                if !in_class {
                    return invalid("word-boundary regex assertions are unsupported");
                }
                cp = 0x08;
                None
            }
            '0' => {
                if self.digit() {
                    return invalid("legacy octal regex escapes are unsupported");
                }
                cp = 0;
                None
            }
            'c' => {
                cp = self.take()?;
                if !char::from_u32(cp).is_some_and(|c| c.is_ascii_alphabetic()) {
                    return invalid("regex control escape requires an ASCII letter");
                }
                cp %= 32;
                None
            }
            'u' => {
                if self.is(b'{') {
                    self.position += 1;
                    cp = 0;
                    let mut digits = 0u32;
                    while !self.is(b'}') {
                        digits += 1;
                        cp = (cp << 4) | self.hex(1)?;
                        if cp > 0x10ffff {
                            return invalid("invalid regex Unicode code point");
                        }
                    }
                    self.position += 1;
                    if digits == 0 {
                        return invalid("empty regex Unicode code point");
                    }
                } else {
                    cp = self.hex(4)?;
                    if (0xd800..=0xdbff).contains(&cp) {
                        if self.take()? != u32::from(b'\\') || self.take()? != u32::from(b'u') {
                            return invalid("unpaired surrogate in regex");
                        }
                        let low = self.hex(4)?;
                        if !(0xdc00..=0xdfff).contains(&low) {
                            return invalid("unpaired surrogate in regex");
                        }
                        cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
                    }
                }
                None
            }
            'x' => {
                cp = self.hex(2)?;
                None
            }
            _ => {
                let syntax = cp <= 127 && "^$\\.*+?()[]{}|/".contains(char::from(cp as u8));
                if cp > 127 || !(syntax || (in_class && cp == u32::from(b'-'))) {
                    return invalid("unsupported regex escape or backreference");
                }
                None
            }
        };
        let Some(mut set) = set else {
            return Ok(Chars::single(cp));
        };
        if matches!(char::from_u32(escape), Some('D' | 'S' | 'W')) {
            set.complement();
        }
        Ok(set)
    }

    fn class(&mut self) -> Result<Chars> {
        // Parse range endpoints explicitly: whitespace is a literal here, and
        // regex shorthands differ from ICU UnicodeSet shorthand syntax.
        let mut result = Chars::new();
        let negate = self.is(b'^');
        if negate {
            self.position += 1;
        }
        while !self.is(b']') {
            let cp = self.take()?;
            let mut chars = if cp == u32::from(b'\\') {
                self.escape(true)?
            } else {
                Chars::single(cp)
            };
            if self.is(b'-') && self.at(self.position + 1) != u32::from(b']') {
                self.position += 1;
                let cp = self.take()?;
                let end = if cp == u32::from(b'\\') {
                    self.escape(true)?
                } else {
                    Chars::single(cp)
                };
                match (chars.first(), end.first()) {
                    (Some(first), Some(last))
                        if chars.size() == 1 && end.size() == 1 && first <= last =>
                    {
                        chars.add(first, last);
                    }
                    _ => return invalid("invalid regex character range"),
                }
            }
            result.add_all(&chars);
        }
        self.position += 1;
        if negate {
            result.complement();
        }
        Ok(result)
    }

    fn alternatives(&mut self, depth: u32) -> Result<Syntax> {
        if depth > MAX_NESTING {
            return invalid("regex nesting exceeds 32 levels");
        }
        let mut alternatives = Syntax::new(SyntaxKind::Alternative);
        loop {
            let mut sequence = Syntax::new(SyntaxKind::Sequence);
            while self.position < self.text.len() && !self.is(b')') && !self.is(b'|') {
                let cp = self.take()?;
                let mut atom = Syntax::new(SyntaxKind::Atom);
                match char::from_u32(cp).unwrap_or('\u{fffd}') {
                    '(' => {
                        let (mut positive, mut negative) = (false, false);
                        if self.is(b'?') {
                            self.position += 1;
                            let modifier = self.take()?;
                            positive = modifier == u32::from(b'=');
                            negative = modifier == u32::from(b'!');
                            if !positive && !negative && modifier != u32::from(b':') {
                                return invalid(
                                    "unsupported regex group, lookbehind or inline flag",
                                );
                            }
                        }
                        atom = self.alternatives(depth + 1)?;
                        if self.take()? != u32::from(b')') {
                            return invalid("unbalanced regex parentheses");
                        }
                        if positive || negative {
                            let mut assertion = Syntax::new(if positive {
                                SyntaxKind::Positive
                            } else {
                                SyntaxKind::Negative
                            });
                            assertion.assertion = true;
                            assertion.children.push(atom);
                            atom = assertion;
                        }
                    }
                    '[' => {
                        let chars = self.class()?;
                        atom.atom = self.e.chars(chars)?;
                    }
                    '^' => {
                        atom.atom = self.e.start;
                        atom.assertion = true;
                    }
                    '$' => {
                        atom.kind = SyntaxKind::End;
                        atom.assertion = true;
                        atom.atom = self.e.epsilon;
                    }
                    '\\' => {
                        if self.is(b'b') || self.is(b'B') {
                            let positive = self.take()? == u32::from(b'b');
                            atom.atom = self.e.boundary(positive)?;
                            atom.assertion = true;
                        } else {
                            let chars = self.escape(false)?;
                            atom.atom = self.e.chars(chars)?;
                        }
                    }
                    '.' => {
                        let mut chars = Chars::scalars();
                        chars.remove(0x0a, 0x0a);
                        chars.remove(0x0d, 0x0d);
                        chars.remove(0x2028, 0x2029);
                        atom.atom = self.e.chars(chars)?;
                    }
                    '*' | '+' | '?' | '{' | '}' => {
                        return invalid("regex repetition has no preceding atom");
                    }
                    _ => atom.atom = self.e.chars(Chars::single(cp))?,
                }
                let quantifier = self.peek();
                if [b'*', b'+', b'?', b'{']
                    .map(u32::from)
                    .contains(&quantifier)
                {
                    self.position += 1;
                    let mut repeated = Syntax::new(SyntaxKind::Repeat);
                    repeated.low = Id::from(quantifier == u32::from(b'+'));
                    repeated.high = if quantifier == u32::from(b'?') {
                        1
                    } else {
                        INFINITE
                    };
                    if quantifier == u32::from(b'{') {
                        repeated.low = self.count()?;
                        repeated.high = repeated.low;
                        if self.is(b',') {
                            self.position += 1;
                            repeated.high = if self.is(b'}') {
                                INFINITE
                            } else {
                                self.count()?
                            };
                        }
                        if self.take()? != u32::from(b'}') || repeated.high < repeated.low {
                            return invalid("invalid regex repetition");
                        }
                    }
                    if atom.assertion && (repeated.high == INFINITE || repeated.high > 32) {
                        return invalid(
                            "regex assertions in unbounded or large repetitions are unsupported",
                        );
                    }
                    repeated.assertion = atom.assertion;
                    repeated.children.push(atom);
                    atom = repeated;
                    if self.is(b'?') {
                        self.position += 1;
                    }
                }
                sequence.assertion |= atom.assertion;
                sequence.children.push(atom);
            }
            alternatives.assertion |= sequence.assertion;
            alternatives.children.push(sequence);
            if !self.is(b'|') {
                break;
            }
            self.position += 1;
        }
        Ok(alternatives)
    }

    fn expand(&mut self, node: &Syntax, mut tail: Id) -> Result<Id> {
        match node.kind {
            SyntaxKind::Atom => self.e.combine(Kind::Concat, vec![node.atom, tail]),
            SyntaxKind::Sequence => {
                for child in node.children.iter().rev() {
                    tail = self.expand(child, tail)?;
                }
                Ok(tail)
            }
            SyntaxKind::Alternative => {
                let mut choices = Vec::with_capacity(node.children.len());
                for child in &node.children {
                    choices.push(self.expand(child, tail)?);
                }
                self.e.combine(Kind::Or, choices)
            }
            SyntaxKind::Repeat => {
                if node.assertion {
                    let mut choices = Vec::new();
                    for count in 0..=node.high {
                        if count >= node.low {
                            choices.push(tail);
                        }
                        if count < node.high {
                            tail = self.expand(&node.children[0], tail)?;
                        }
                    }
                    return self.e.combine(Kind::Or, choices);
                }
                let epsilon = self.e.epsilon;
                let body = self.expand(&node.children[0], epsilon)?;
                let repeated = self.e.repeat(body, node.low, node.high)?;
                self.e.combine(Kind::Concat, vec![repeated, tail])
            }
            SyntaxKind::Positive | SyntaxKind::Negative => {
                let all = self.e.all;
                let mut condition = self.expand(&node.children[0], all)?;
                if node.kind == SyntaxKind::Negative {
                    condition = self.e.not(condition)?;
                }
                self.e.combine(Kind::And, vec![condition, tail])
            }
            SyntaxKind::End => self.e.combine(Kind::And, vec![node.atom, tail]),
        }
    }
}

/// The compiled DFA as flat tables. State 0 starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dfa {
    /// Each alphabet class: codepoints with the same transition everywhere.
    pub alphabet: Vec<Chars>,
    /// `states × classes`, row-major; `DEAD` where no accepted value continues.
    pub next: Vec<Id>,
    pub accepting: Vec<u8>,
    /// The fewest codepoints to an accepting state, or `DEAD`.
    pub distance: Vec<Id>,
    /// The distinct successors of each state, before pruning dead ones:
    /// `successors[successor_offsets[s]..successor_offsets[s + 1]]`.
    pub successor_offsets: Vec<Id>,
    pub successors: Vec<Id>,
    pub maximum_suffix: Id,
    /// Codepoint lookup: `lookup_classes[i]` holds for
    /// `[lookup_starts[i], lookup_starts[i + 1])`, `DEAD` for surrogates.
    /// The last start is 0x110000.
    pub lookup_starts: Vec<u32>,
    pub lookup_classes: Vec<Id>,
}

impl Dfa {
    pub fn states(&self) -> usize {
        self.accepting.len()
    }
}

pub fn compile(patterns: &[&[u8]], maximum: Id, sets: &mut dyn UnicodeSets) -> Result<Dfa> {
    let mut e = Expressions::new(maximum)?;
    let mut conditions = Vec::with_capacity(patterns.len());
    for pattern in patterns {
        conditions.push(Parser::new(&mut e, sets, pattern)?.parse()?);
    }
    let root = e.combine(Kind::And, conditions)?;

    // Partition Unicode once by membership in all character classes. Every
    // codepoint in a partition has the same derivative for every expression.
    let mut endpoints: Vec<u32> = vec![0, 0xd800, 0xe000, 0x110000];
    for chars in &e.classes {
        for &(first, last) in chars.ranges() {
            endpoints.push(first);
            endpoints.push(last + 1);
        }
    }
    endpoints.sort_unstable();
    endpoints.dedup();
    let mut partitions: HashMap<Vec<Id>, Id> = HashMap::new();
    let mut alphabet: Vec<Chars> = Vec::new();
    let mut lookup_starts: Vec<u32> = Vec::new();
    let mut lookup_classes: Vec<Id> = Vec::new();
    for pair in endpoints.windows(2) {
        let (first, last) = (pair[0], pair[1] - 1);
        let class = if (SURROGATES.0..=SURROGATES.1).contains(&first) {
            DEAD
        } else {
            let membership: Vec<Id> = (0..e.classes.len() as Id)
                .filter(|&c| e.classes[c as usize].contains(first))
                .collect();
            let next = alphabet.len() as Id;
            let class = *partitions.entry(membership).or_insert(next);
            if class == next {
                alphabet.push(Chars::new());
            }
            alphabet[class as usize].add(first, last);
            class
        };
        if lookup_classes.last() != Some(&class) {
            lookup_starts.push(first);
            lookup_classes.push(class);
        }
    }
    lookup_starts.push(0x110000);

    let mut states = States {
        ids: HashMap::new(),
        pending: Vec::new(),
        depths: Vec::new(),
        classes: alphabet.len(),
    };
    states.intern((root, true, false), 0)?;
    let representatives: Vec<u32> = alphabet
        .iter()
        .map(|chars| chars.first().unwrap_or(0))
        .collect();
    let mut accepting: Vec<u8> = Vec::new();
    let mut next: Vec<Id> = Vec::new();
    let mut i = 0;
    while i < states.pending.len() {
        let (expression, at_start, previous_word) = states.pending[i];
        let depth = states.depths[i];
        accepting.push(u8::from(e.nullable(
            expression,
            at_start,
            previous_word,
            false,
        )));
        i += 1;
        // BFS discovers each state at its minimum reachable length. Beyond the
        // schema's maxLength no transition can contribute to an accepted value.
        if depth >= maximum {
            next.extend(std::iter::repeat_n(DEAD, representatives.len()));
            continue;
        }
        for &cp in &representatives {
            let derivative = e.derive(expression, cp, at_start, previous_word)?;
            next.push(if derivative == e.empty {
                DEAD
            } else {
                states.intern(
                    (derivative, false, e.word_boundaries && word(cp)),
                    depth + 1,
                )?
            });
        }
    }

    let count = accepting.len();
    let classes = alphabet.len();
    let mut predecessors: Vec<Vec<Id>> = vec![Vec::new(); count];
    let mut distance: Vec<Id> = vec![DEAD; count];
    let mut successor_offsets: Vec<Id> = Vec::with_capacity(count + 1);
    let mut successors: Vec<Id> = Vec::new();
    let mut queue: VecDeque<Id> = VecDeque::new();
    successor_offsets.push(0);
    for state in 0..count {
        if accepting[state] != 0 {
            distance[state] = 0;
            queue.push_back(state as Id);
        }
        let unique: BTreeSet<Id> = next[state * classes..(state + 1) * classes]
            .iter()
            .copied()
            .filter(|&n| n != DEAD)
            .collect();
        for &successor in &unique {
            successors.push(successor);
            predecessors[successor as usize].push(state as Id);
        }
        successor_offsets.push(successors.len() as Id);
    }
    while let Some(state) = queue.pop_front() {
        for &previous in &predecessors[state as usize] {
            if distance[previous as usize] == DEAD {
                distance[previous as usize] = distance[state as usize] + 1;
                queue.push_back(previous);
            }
        }
    }
    for target in &mut next {
        if *target != DEAD && distance[*target as usize] == DEAD {
            *target = DEAD;
        }
    }
    let maximum_suffix = distance
        .iter()
        .copied()
        .filter(|&d| d != DEAD)
        .max()
        .unwrap_or(0);
    Ok(Dfa {
        alphabet,
        next,
        accepting,
        distance,
        successor_offsets,
        successors,
        maximum_suffix,
        lookup_starts,
        lookup_classes,
    })
}

/// DFA states by (expression, at start, previous codepoint is a word), in
/// order of discovery.
struct States {
    ids: HashMap<(Id, bool, bool), Id>,
    pending: Vec<(Id, bool, bool)>,
    depths: Vec<Id>,
    classes: usize,
}

impl States {
    fn intern(&mut self, key: (Id, bool, bool), depth: Id) -> Result<Id> {
        if let Some(&id) = self.ids.get(&key) {
            return Ok(id);
        }
        let id = self.ids.len() as Id;
        self.ids.insert(key, id);
        if self.ids.len() > MAX_STATES || self.ids.len() * self.classes > MAX_TRANSITIONS {
            return invalid("compiled regex exceeds the state budget");
        }
        self.pending.push(key);
        self.depths.push(depth);
        Ok(id)
    }
}
