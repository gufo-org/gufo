//! The C ABI. The C++ differential test compares the compiler with the
//! frozen C++ one; these cover the boundary and a few languages.

use super::*;

/// ICU stand-in: `\s`'s set, `\p{L}` as ASCII letters, `\p{Fail}` as a
/// callback failure; every other pattern is rejected.
unsafe extern "C" fn sets(
    _: *mut c_void,
    pattern: *const u8,
    len: usize,
    add: GufoJsonSchemaRangeFn,
    add_string: GufoJsonSchemaStringFn,
    sink: *mut c_void,
) -> c_int {
    // SAFETY: the compiler passes readable bytes.
    let pattern = unsafe { std::slice::from_raw_parts(pattern, len) };
    let ranges: &[(u32, u32)] = match pattern {
        b"[\\p{Zs}\\u0009-\\u000d\\u2028\\u2029\\ufeff]" => &[
            (0x09, 0x0d),
            (0x20, 0x20),
            (0xa0, 0xa0),
            (0x2028, 0x2029),
            (0xfeff, 0xfeff),
        ],
        b"[\\p{L}]" => &[(0x41, 0x5a), (0x61, 0x7a)],
        b"[\\p{Fail}]" => return GUFO_JSON_SCHEMA_SET_FAILED,
        b"[\\p{Flags}]" => {
            let flag = [0x1f1e6u32, 0x1f1e8];
            // SAFETY: `add_string` and `sink` come from the compiler for this
            // call; `flag` outlives it.
            unsafe { add_string(sink, flag.as_ptr(), flag.len()) };
            return GUFO_JSON_SCHEMA_SET_RESOLVED;
        }
        _ => return 1,
    };
    for &(first, last) in ranges {
        // SAFETY: `add` and `sink` come from the compiler for this call.
        unsafe { add(sink, first, last) };
    }
    0
}

unsafe extern "C" fn record(context: *mut c_void, data: *const u8, len: usize) {
    // SAFETY: every test passes a live `Vec<Vec<u8>>` and readable bytes.
    unsafe {
        (*context.cast::<Vec<Vec<u8>>>()).push(std::slice::from_raw_parts(data, len).to_vec());
    }
}

struct Compiled {
    handle: *mut GufoJsonSchemaRegex,
    tables: GufoJsonSchemaRegexTables,
}

impl Drop for Compiled {
    fn drop(&mut self) {
        // SAFETY: the handle came from the compiler and is freed once.
        unsafe { gufo_json_schema_regex_free(self.handle) };
    }
}

impl Compiled {
    fn table<T: Copy>(data: *const T, len: usize) -> &'static [T] {
        // SAFETY: the tables live as long as the handle, which outlives
        // every use in these tests.
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    fn advance(&self, state: u32, cp: u32) -> u32 {
        let t = &self.tables;
        if state == regex::DEAD || cp > 0x10ffff {
            return regex::DEAD;
        }
        let starts = Self::table(t.lookup_starts, t.lookup_len + 1);
        let classes = Self::table(t.lookup_classes, t.lookup_len);
        let class = classes[starts.partition_point(|&s| s <= cp) - 1];
        if class == regex::DEAD {
            return regex::DEAD;
        }
        let next = Self::table(t.next, (t.states * t.classes) as usize);
        next[(state * t.classes + class) as usize]
    }

    fn matches(&self, text: &str) -> bool {
        let mut state = 0;
        for c in text.chars() {
            state = self.advance(state, u32::from(c));
        }
        let accepting = Self::table(self.tables.accepting, self.tables.states as usize);
        state != regex::DEAD && accepting[state as usize] != 0
    }
}

fn compile(patterns: &[&str], maximum: u32) -> Result<Compiled, (GufoJsonSchemaStatus, String)> {
    let entries: Vec<GufoJsonSchemaBytes> = patterns
        .iter()
        .map(|p| GufoJsonSchemaBytes {
            data: p.as_ptr(),
            len: p.len(),
        })
        .collect();
    let mut errors: Vec<Vec<u8>> = Vec::new();
    let mut handle = std::ptr::null_mut();
    // SAFETY: the entries, `errors` and `handle` outlive the call.
    let status = unsafe {
        gufo_json_schema_regex_compile(
            entries.as_ptr(),
            entries.len(),
            maximum,
            Some(sets),
            Some(record),
            std::ptr::from_mut(&mut errors).cast(),
            &mut handle,
        )
    };
    if status != GufoJsonSchemaStatus::Complete {
        assert!(handle.is_null());
        let reason = errors.concat();
        return Err((status, String::from_utf8(reason).unwrap()));
    }
    let mut tables = std::mem::MaybeUninit::uninit();
    // SAFETY: `handle` is live and `tables` writable.
    let status = unsafe { gufo_json_schema_regex_tables(handle, tables.as_mut_ptr()) };
    assert_eq!(status, GufoJsonSchemaStatus::Complete);
    Ok(Compiled {
        handle,
        // SAFETY: the call above filled it.
        tables: unsafe { tables.assume_init() },
    })
}

fn rejected(pattern: &str) -> String {
    match compile(&[pattern], u32::MAX) {
        Err((GufoJsonSchemaStatus::Rejected, reason)) => reason,
        other => panic!("{pattern:?} gave {:?}", other.map(|_| ())),
    }
}

#[test]
fn languages() {
    let digits = compile(&["^[0-9]{2,3}$"], u32::MAX).unwrap();
    assert!(digits.matches("12") && digits.matches("123"));
    assert!(!digits.matches("1") && !digits.matches("1234") && !digits.matches("1a"));
    // Unanchored patterns match anywhere; the intersection must hold.
    let both = compile(&["a", "b"], u32::MAX).unwrap();
    assert!(both.matches("xaxbx") && !both.matches("aaa"));
    let words = compile(&["^\\p{L}+\\s\\w$"], u32::MAX).unwrap();
    assert!(words.matches("ab\u{a0}_") && !words.matches("a1 _"));
    let lookahead = compile(&["^(?=.*1)(?!.*2)[0-9]+$"], u32::MAX).unwrap();
    assert!(lookahead.matches("31") && !lookahead.matches("312") && !lookahead.matches("3"));
    let boundary = compile(&["^a\\b"], u32::MAX).unwrap();
    assert!(boundary.matches("a b") && !boundary.matches("ab"));
    let astral = compile(&["^\\u{1F600}\\ud83d\\ude00$"], u32::MAX).unwrap();
    assert!(astral.matches("\u{1f600}\u{1f600}"));
    // The bound caps repetitions; the lexeme counts the rest.
    let bounded = compile(&["^a{0,5}$"], 3).unwrap();
    assert!(bounded.matches("aaa") && !bounded.matches("aaaa"));
    let suffix = compile(&["^a{3}$"], 3).unwrap();
    assert_eq!(suffix.tables.maximum_suffix, 3);
    // States first reached at the bound have no transitions.
    let cut = compile(&["^(?:ab|ba)*$"], 1).unwrap();
    assert!(!cut.matches("ab"));
    // A set of strings matches no codepoint but is not empty: the class
    // stays distinct from `[^]`, as in ICU.
    let flags = compile(&["^[^\\p{Flags}]$"], u32::MAX).unwrap();
    assert!(flags.matches("a") && !flags.matches(""));
    let strings = compile(&["^\\p{Flags}?$"], u32::MAX).unwrap();
    assert!(strings.matches("") && !strings.matches("\u{1f1e6}"));
    assert_eq!(rejected("[\\p{Flags}-z]"), "invalid regex character range");
    let any = compile(&[], u32::MAX).unwrap();
    assert!(any.matches("") && any.matches("\u{10ffff}"));
    // The start state alone matches `^`.
    assert_eq!(any.tables.states, 2);
}

#[test]
fn rejections() {
    assert_eq!(rejected("(a"), "truncated regex");
    assert_eq!(rejected("a)"), "unbalanced regex parentheses");
    assert_eq!(rejected("*"), "regex repetition has no preceding atom");
    assert_eq!(rejected("a{2,1}"), "invalid regex repetition");
    assert_eq!(rejected("a{4294967295}"), "regex repetition is too large");
    assert_eq!(rejected("\\1"), "unsupported regex escape or backreference");
    assert_eq!(
        rejected("(?<=a)"),
        "unsupported regex group, lookbehind or inline flag"
    );
    assert_eq!(rejected("\\p{Nope}"), "invalid regex Unicode property");
    assert_eq!(rejected("\\pL"), "Unicode property escape requires braces");
    assert_eq!(rejected("[b-a]"), "invalid regex character range");
    assert_eq!(rejected("\\ud800"), "truncated regex");
    assert_eq!(rejected("\\ud800x"), "unpaired surrogate in regex");
    assert_eq!(rejected("\\ud800\\u0041"), "unpaired surrogate in regex");
    assert_eq!(rejected("\\u{110000}"), "invalid regex Unicode code point");
    assert_eq!(rejected("\\u{}"), "empty regex Unicode code point");
    assert_eq!(
        rejected("\\01"),
        "legacy octal regex escapes are unsupported"
    );
    assert_eq!(
        rejected("\\c1"),
        "regex control escape requires an ASCII letter"
    );
    assert_eq!(rejected("\\xg0"), "invalid regex Unicode escape");
    assert_eq!(
        rejected("(^)*"),
        "regex assertions in unbounded or large repetitions are unsupported"
    );
    assert_eq!(rejected(&"(".repeat(40)), "regex nesting exceeds 32 levels");
    assert_eq!(rejected(&"a".repeat(16385)), "regex exceeds 16384 bytes");
    assert_eq!(
        rejected("^[ab]*a[ab]{12}$"),
        "compiled regex exceeds the state budget"
    );
    assert!(matches!(
        compile(&["\\p{Fail}"], u32::MAX),
        Err((GufoJsonSchemaStatus::Stopped, _))
    ));
}

#[test]
fn string_and_format() {
    let keywords = [
        GufoJsonSchemaValue {
            kind: GUFO_JSON_SCHEMA_NUMBER,
            number: 1.0,
            text: b"1".as_ptr(),
            len: 1,
        },
        GufoJsonSchemaValue {
            kind: GUFO_JSON_SCHEMA_ABSENT,
            number: 0.0,
            text: std::ptr::null(),
            len: 0,
        },
        GufoJsonSchemaValue {
            kind: GUFO_JSON_SCHEMA_STRING,
            number: 0.0,
            text: b"x".as_ptr(),
            len: 1,
        },
        GufoJsonSchemaValue {
            kind: GUFO_JSON_SCHEMA_STRING,
            number: 0.0,
            text: b"hostname".as_ptr(),
            len: 8,
        },
    ];
    let (mut minimum, mut maximum) = (0, 0);
    let mut patterns: Vec<Vec<u8>> = Vec::new();
    // SAFETY: every buffer outlives the call.
    let status = unsafe {
        gufo_json_schema_string(
            keywords.as_ptr(),
            &mut minimum,
            &mut maximum,
            Some(record),
            None,
            std::ptr::from_mut(&mut patterns).cast(),
        )
    };
    assert_eq!(status, GufoJsonSchemaStatus::Complete);
    assert_eq!((minimum, maximum), (1, 253));
    assert_eq!(patterns.len(), 2);
    assert_eq!(patterns[0], b"x");
    let mut cap = 0;
    let mut texts: Vec<Vec<u8>> = Vec::new();
    // SAFETY: as above.
    let status = unsafe {
        gufo_json_schema_format(
            b"nope".as_ptr(),
            4,
            &mut cap,
            Some(record),
            Some(record),
            std::ptr::from_mut(&mut texts).cast(),
        )
    };
    assert_eq!(status, GufoJsonSchemaStatus::Rejected);
    assert_eq!(texts, [b"unsupported string format".to_vec()]);
    // Every format compiles.
    for name in [
        "date",
        "time",
        "date-time",
        "uuid",
        "ipv4",
        "ipv6",
        "hostname",
        "email",
        "duration",
    ] {
        let (pattern, cap) = string::format(name.as_bytes()).unwrap();
        let compiled = compile(&[&pattern], cap).unwrap();
        let sample = match name {
            "date" => "2024-02-29",
            "time" => "23:59:60.5Z",
            "date-time" => "2023-12-31T00:00:00+01:00",
            "uuid" => "123e4567-e89b-12d3-a456-426614174000",
            "ipv4" => "192.168.0.255",
            "ipv6" => "::ffff:10.0.0.1",
            "hostname" => "example.com.",
            "email" => "a@b",
            _ => "P1Y2MT3.5S",
        };
        assert!(compiled.matches(sample), "{name}");
    }
    assert!(
        !compile(&[&string::format(b"date").unwrap().0], u32::MAX)
            .unwrap()
            .matches("2023-02-29")
    );
}

#[test]
fn number_lexeme() {
    let absent = GufoJsonSchemaValue {
        kind: GUFO_JSON_SCHEMA_ABSENT,
        number: 0.0,
        text: std::ptr::null(),
        len: 0,
    };
    let mut keywords = [absent; 5];
    keywords[0] = GufoJsonSchemaValue {
        kind: GUFO_JSON_SCHEMA_NUMBER,
        number: 0.5,
        text: b"0.5".as_ptr(),
        len: 3,
    };
    let mut handle = std::ptr::null_mut();
    // SAFETY: every buffer outlives the call.
    let status = unsafe {
        gufo_json_schema_number_new(
            keywords.as_ptr(),
            1,
            16,
            None,
            std::ptr::null_mut(),
            &mut handle,
        )
    };
    assert_eq!(status, GufoJsonSchemaStatus::Complete);
    let check = |text: &[u8]| {
        let mut bits = 0xff;
        // SAFETY: `handle` is live; the buffers outlive the call.
        let status =
            unsafe { gufo_json_schema_number_check(handle, text.as_ptr(), text.len(), &mut bits) };
        assert_eq!(status, GufoJsonSchemaStatus::Complete);
        bits
    };
    assert_eq!(check(b"0"), 0);
    assert_eq!(
        check(b"1"),
        GUFO_JSON_SCHEMA_PREFIX | GUFO_JSON_SCHEMA_MATCH
    );
    assert_eq!(check(b"-"), 0);
    let mut accepted = 0xff;
    // SAFETY: as above.
    let status = unsafe {
        gufo_json_schema_number_accept(
            handle,
            b"2".as_ptr(),
            1,
            &mut accepted,
            None,
            std::ptr::null_mut(),
        )
    };
    assert_eq!((status, accepted), (GufoJsonSchemaStatus::Complete, 1));
    // SAFETY: freed once.
    unsafe { gufo_json_schema_number_free(handle) };

    let mut texts: Vec<Vec<u8>> = Vec::new();
    let two = GufoJsonSchemaValue {
        kind: GUFO_JSON_SCHEMA_NUMBER,
        number: 0.2,
        text: b"0.2".as_ptr(),
        len: 3,
    };
    let three = GufoJsonSchemaValue {
        kind: GUFO_JSON_SCHEMA_NUMBER,
        number: 0.03,
        text: b"0.03".as_ptr(),
        len: 4,
    };
    // SAFETY: as above.
    let status = unsafe {
        gufo_json_schema_multiple_of(
            &two,
            &three,
            Some(record),
            None,
            std::ptr::from_mut(&mut texts).cast(),
        )
    };
    assert_eq!(status, GufoJsonSchemaStatus::Complete);
    assert_eq!(texts, [b"6e-1".to_vec()]);
    let mut equal = 0xff;
    // SAFETY: as above.
    let status = unsafe {
        gufo_json_schema_number_equal(
            b"0.6".as_ptr(),
            3,
            b"6e-1".as_ptr(),
            4,
            &mut equal,
            None,
            std::ptr::null_mut(),
        )
    };
    assert_eq!((status, equal), (GufoJsonSchemaStatus::Complete, 1));
}

#[test]
fn caller_errors_are_internal() {
    let mut handle = std::ptr::null_mut();
    // SAFETY: the null inputs are rejected before any read.
    unsafe {
        assert_eq!(
            gufo_json_schema_regex_compile(
                std::ptr::null(),
                1,
                0,
                None,
                None,
                std::ptr::null_mut(),
                &mut handle
            ),
            GufoJsonSchemaStatus::Internal
        );
        assert_eq!(
            gufo_json_schema_regex_compile(
                std::ptr::null(),
                0,
                0,
                None,
                None,
                std::ptr::null_mut(),
                std::ptr::null_mut()
            ),
            GufoJsonSchemaStatus::Internal
        );
        let mut bits = 0;
        assert_eq!(
            gufo_json_schema_number_check(std::ptr::null(), std::ptr::null(), 0, &mut bits),
            GufoJsonSchemaStatus::Internal
        );
        gufo_json_schema_regex_free(std::ptr::null_mut());
        gufo_json_schema_number_free(std::ptr::null_mut());
    }
    assert!(handle.is_null());
    // An unknown value kind is a caller error.
    let mut keywords = [GufoJsonSchemaValue {
        kind: GUFO_JSON_SCHEMA_ABSENT,
        number: 0.0,
        text: std::ptr::null(),
        len: 0,
    }; 4];
    keywords[2].kind = 7;
    let (mut minimum, mut maximum) = (0, 0);
    // SAFETY: the buffers outlive the call.
    let status = unsafe {
        gufo_json_schema_string(
            keywords.as_ptr(),
            &mut minimum,
            &mut maximum,
            None,
            None,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(status, GufoJsonSchemaStatus::Internal);
    // A pattern needing a set without a resolver stops.
    let pattern = b"\\s";
    let entry = GufoJsonSchemaBytes {
        data: pattern.as_ptr(),
        len: pattern.len(),
    };
    // SAFETY: the buffers outlive the call.
    let status = unsafe {
        gufo_json_schema_regex_compile(&entry, 1, 0, None, None, std::ptr::null_mut(), &mut handle)
    };
    assert_eq!(status, GufoJsonSchemaStatus::Stopped);
}
