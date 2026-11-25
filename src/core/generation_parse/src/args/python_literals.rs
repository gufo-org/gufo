//! Qwen's XML-like calls look like Python keyword arguments, and the models
//! sometimes write Python's True, False and None where the schema asks for a
//! JSON boolean or null.

/// Rewrites those words outside JSON string literals.
pub fn to_json(value: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(value.len());
    let mut in_string = false;
    let mut i = 0;
    while i < value.len() {
        let c = value[i];
        if in_string {
            let length = if c == b'\\' && i + 1 < value.len() {
                2
            } else {
                1
            };
            out.extend_from_slice(&value[i..i + length]);
            in_string = c != b'"';
            i += length;
            continue;
        }
        if !c.is_ascii_alphabetic() {
            out.push(c);
            in_string = c == b'"';
            i += 1;
            continue;
        }
        let mut word_end = i;
        while word_end < value.len()
            && (value[word_end].is_ascii_alphanumeric() || value[word_end] == b'_')
        {
            word_end += 1;
        }
        let word = &value[i..word_end];
        out.extend_from_slice(match word {
            b"True" => b"true",
            b"False" => b"false",
            b"None" => b"null",
            _ => word,
        });
        i = word_end;
    }
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn rewrites_words_outside_strings() {
        assert_eq!(
            super::to_json(b"[True, \"True\", None_x, False]"),
            b"[true, \"True\", None_x, false]".to_vec()
        );
        assert_eq!(
            super::to_json(b"\"a\\\"None\" None"),
            b"\"a\\\"None\" null".to_vec()
        );
    }
}
