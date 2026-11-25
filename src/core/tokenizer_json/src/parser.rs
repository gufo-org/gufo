//! Hugging Face `tokenizer.json` parser.
//!
//! Extracts vocabulary, merges, added tokens, post-processor template IDs, and
//! model/normalizer/pre-tokenizer metadata for C++ tokenizers.

use gufo_json::{Limits, Number, Object, Value, ValueError, parse_value};

const MAX_BYTES: usize = 512 << 20;
const MAX_VALUES: usize = 16_000_000;
const MAX_POST_PROCESSOR_IDS: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    Invalid(String),
    Internal(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabEntry {
    pub token: String,
    pub id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeEntry {
    pub left: String,
    pub right: String,
    pub was_string: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedToken {
    pub content: String,
    pub id: u32,
    pub has_single_word: bool,
    pub single_word: bool,
    pub has_lstrip: bool,
    pub lstrip: bool,
    pub has_rstrip: bool,
    pub rstrip: bool,
    pub has_normalized: bool,
    pub normalized: bool,
    pub flags_valid: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Meta {
    pub truncation_is_null: bool,
    pub padding_is_null: bool,

    pub has_model: bool,
    pub has_vocab: bool,
    pub has_merges: bool,
    pub has_added_tokens: bool,

    pub model_type: String,
    pub has_dropout: bool,
    pub dropout_is_null: bool,
    pub continuing_subword_prefix: String,
    pub end_of_word_suffix: String,
    pub has_byte_fallback: bool,
    pub byte_fallback: bool,
    pub has_ignore_merges: bool,
    pub ignore_merges: bool,
    pub has_unk_token: bool,
    pub unk_token_is_null: bool,
    pub unk_token: String,

    pub has_normalizer: bool,
    pub normalizer_type: String,
    pub normalizer_pattern_string: String,
    pub normalizer_content: String,

    pub has_pre_tokenizer: bool,
    pub pre_tokenizer_type: String,
    pub pre_tokenizer_pattern_string: String,
    pub pre_tokenizer_behavior: String,
    pub has_pre_tokenizer_invert: bool,
    pub pre_tokenizer_invert: bool,

    pub pre_tokenizer_sub_count: usize,
    pub split_type: String,
    pub split_behavior: String,
    pub has_split_invert: bool,
    pub split_invert: bool,
    pub split_regex: String,
    pub byte_level_type: String,
    pub has_add_prefix_space: bool,
    pub add_prefix_space: bool,
    pub has_use_regex: bool,
    pub use_regex: bool,

    pub has_post_processor: bool,
    pub post_processor_valid: bool,
    pub post_processor_error: String,
}

#[derive(Debug, Clone)]
pub struct TokenizerJson {
    pub meta: Meta,
    pub vocab: Vec<VocabEntry>,
    pub merges: Vec<MergeEntry>,
    pub added_tokens: Vec<AddedToken>,
    pub prefix_ids: Vec<u32>,
    pub suffix_ids: Vec<u32>,
}

pub fn parse(data: &[u8], max_token_id: u32) -> Result<TokenizerJson, ParseError> {
    let limits = Limits {
        bytes: MAX_BYTES,
        values: MAX_VALUES,
    };
    let root_val = parse_value(data, limits).map_err(map_json_error)?;
    let Some(root) = root_val.as_object() else {
        return Err(ParseError::Invalid("expected a JSON object".to_string()));
    };

    let mut meta = Meta {
        truncation_is_null: root.get("truncation").is_none_or(is_null),
        padding_is_null: root.get("padding").is_none_or(is_null),
        ..Meta::default()
    };

    if let Some(norm_val) = root.get("normalizer")
        && let Some(norm) = norm_val.as_object()
    {
        meta.has_normalizer = true;
        meta.normalizer_type = member_str(norm, "type").to_string();
        meta.normalizer_pattern_string = string_pattern(norm);
        meta.normalizer_content = member_str(norm, "content").to_string();
    }

    if let Some(pre_val) = root.get("pre_tokenizer")
        && let Some(pre) = pre_val.as_object()
    {
        meta.has_pre_tokenizer = true;
        meta.pre_tokenizer_type = member_str(pre, "type").to_string();
        meta.pre_tokenizer_pattern_string = string_pattern(pre);
        meta.pre_tokenizer_behavior = member_str(pre, "behavior").to_string();
        if let Some(inv) = pre.get("invert").and_then(Value::as_bool) {
            meta.has_pre_tokenizer_invert = true;
            meta.pre_tokenizer_invert = inv;
        }
        if let Some(seq) = pre.get("pretokenizers").and_then(Value::as_array) {
            meta.pre_tokenizer_sub_count = seq.len();
            if seq.len() >= 2 {
                if let Some(split) = seq[0].as_object() {
                    meta.split_type = member_str(split, "type").to_string();
                    meta.split_behavior = member_str(split, "behavior").to_string();
                    if let Some(inv) = split.get("invert").and_then(Value::as_bool) {
                        meta.has_split_invert = true;
                        meta.split_invert = inv;
                    }
                    if let Some(pat) = split.get("pattern").and_then(Value::as_object) {
                        meta.split_regex = member_str(pat, "Regex").to_string();
                    }
                }
                if let Some(bl) = seq[1].as_object() {
                    meta.byte_level_type = member_str(bl, "type").to_string();
                    if let Some(aps) = bl.get("add_prefix_space").and_then(Value::as_bool) {
                        meta.has_add_prefix_space = true;
                        meta.add_prefix_space = aps;
                    }
                    if let Some(ur) = bl.get("use_regex").and_then(Value::as_bool) {
                        meta.has_use_regex = true;
                        meta.use_regex = ur;
                    }
                }
            }
        }
    }

    let mut vocab = Vec::new();
    let mut merges = Vec::new();
    if let Some(model_val) = root.get("model") {
        let Some(model) = model_val.as_object() else {
            return Err(ParseError::Invalid(
                "invalid model: expected an object".to_string(),
            ));
        };
        meta.has_model = true;
        meta.model_type = member_str(model, "type").to_string();
        if let Some(dropout) = model.get("dropout") {
            meta.has_dropout = true;
            meta.dropout_is_null = is_null(dropout);
        }
        meta.continuing_subword_prefix = member_str(model, "continuing_subword_prefix").to_string();
        meta.end_of_word_suffix = member_str(model, "end_of_word_suffix").to_string();
        if let Some(bf) = model.get("byte_fallback").and_then(Value::as_bool) {
            meta.has_byte_fallback = true;
            meta.byte_fallback = bf;
        }
        if let Some(im) = model.get("ignore_merges").and_then(Value::as_bool) {
            meta.has_ignore_merges = true;
            meta.ignore_merges = im;
        }
        if let Some(unk) = model.get("unk_token") {
            meta.has_unk_token = true;
            meta.unk_token_is_null = is_null(unk);
            if let Some(s) = unk.as_str() {
                meta.unk_token = s.to_string();
            }
        }

        if let Some(vocab_val) = model.get("vocab") {
            let Some(vocab_obj) = vocab_val.as_object() else {
                return Err(ParseError::Invalid(
                    "invalid model.vocab: expected an object".to_string(),
                ));
            };
            meta.has_vocab = true;
            vocab
                .try_reserve_exact(vocab_obj.len())
                .map_err(|_| ParseError::Internal("out of memory".to_string()))?;
            for (token, val) in vocab_obj.members() {
                let Some(id) = read_id(val, max_token_id) else {
                    return Err(ParseError::Invalid(format!(
                        "invalid model.vocab: id of '{token}' is not an integer in [0, {max_token_id})"
                    )));
                };
                vocab.push(VocabEntry {
                    token: token.clone(),
                    id,
                });
            }
        }

        if let Some(merges_val) = model.get("merges") {
            let Some(merges_arr) = merges_val.as_array() else {
                return Err(ParseError::Invalid(
                    "invalid model.merges: expected an array".to_string(),
                ));
            };
            meta.has_merges = true;
            merges
                .try_reserve_exact(merges_arr.len())
                .map_err(|_| ParseError::Internal("out of memory".to_string()))?;
            for (rank, entry) in merges_arr.iter().enumerate() {
                if let Some(s) = entry.as_str() {
                    let Some((left, right)) = s.split_once(' ') else {
                        return Err(ParseError::Invalid(format!(
                            "invalid model.merges: entry {rank} has no space"
                        )));
                    };
                    merges.push(MergeEntry {
                        left: left.to_string(),
                        right: right.to_string(),
                        was_string: true,
                    });
                } else if let Some(pair) = entry.as_array()
                    && pair.len() == 2
                    && let (Some(l), Some(r)) = (pair[0].as_str(), pair[1].as_str())
                {
                    merges.push(MergeEntry {
                        left: l.to_string(),
                        right: r.to_string(),
                        was_string: false,
                    });
                } else {
                    return Err(ParseError::Invalid(format!(
                        "invalid model.merges: entry {rank} is not a pair of strings"
                    )));
                }
            }
        }
    }

    let mut added_tokens = Vec::new();
    if let Some(added_val) = root.get("added_tokens")
        && !is_null(added_val)
    {
        let Some(added_arr) = added_val.as_array() else {
            return Err(ParseError::Invalid(
                "added_tokens must be an array".to_string(),
            ));
        };
        meta.has_added_tokens = true;
        added_tokens
            .try_reserve_exact(added_arr.len())
            .map_err(|_| ParseError::Internal("out of memory".to_string()))?;
        for entry in added_arr {
            let Some(obj) = entry.as_object() else {
                return Err(ParseError::Invalid(
                    "invalid added_tokens entry".to_string(),
                ));
            };
            let Some(content) = obj.get("content").and_then(Value::as_str) else {
                return Err(ParseError::Invalid(
                    "invalid added_tokens entry".to_string(),
                ));
            };
            let Some(id) = obj.get("id").and_then(|v| read_id(v, max_token_id)) else {
                return Err(ParseError::Invalid(
                    "invalid added_tokens entry".to_string(),
                ));
            };
            if content.is_empty() {
                return Err(ParseError::Invalid(
                    "invalid added_tokens entry".to_string(),
                ));
            }
            let (has_sw, sw, sw_ok) = read_opt_bool(obj, "single_word");
            let (has_ls, ls, ls_ok) = read_opt_bool(obj, "lstrip");
            let (has_rs, rs, rs_ok) = read_opt_bool(obj, "rstrip");
            let (has_nm, nm, nm_ok) = read_opt_bool(obj, "normalized");
            added_tokens.push(AddedToken {
                content: content.to_string(),
                id,
                has_single_word: has_sw,
                single_word: sw,
                has_lstrip: has_ls,
                lstrip: ls,
                has_rstrip: has_rs,
                rstrip: rs,
                has_normalized: has_nm,
                normalized: nm,
                flags_valid: sw_ok && ls_ok && rs_ok && nm_ok,
            });
        }
    }

    let mut prefix_ids = Vec::new();
    let mut suffix_ids = Vec::new();
    if let Some(pp_val) = root.get("post_processor")
        && !is_null(pp_val)
    {
        meta.has_post_processor = true;
        match parse_post_processor(pp_val, max_token_id, &mut prefix_ids, &mut suffix_ids) {
            Ok(()) => meta.post_processor_valid = true,
            Err(e) => {
                prefix_ids.clear();
                suffix_ids.clear();
                meta.post_processor_valid = false;
                meta.post_processor_error = e;
            }
        }
    }

    Ok(TokenizerJson {
        meta,
        vocab,
        merges,
        added_tokens,
        prefix_ids,
        suffix_ids,
    })
}

fn parse_post_processor(
    pp_val: &Value,
    max_token_id: u32,
    prefix_ids: &mut Vec<u32>,
    suffix_ids: &mut Vec<u32>,
) -> Result<(), String> {
    let Some(pp) = pp_val.as_object() else {
        return Err("post_processor must be TemplateProcessing".to_string());
    };
    let single = pp.get("single").and_then(Value::as_array);
    let specials = pp.get("special_tokens").and_then(Value::as_object);
    if member_str(pp, "type") != "TemplateProcessing" || single.is_none() || specials.is_none() {
        return Err("post_processor must be TemplateProcessing".to_string());
    }
    let single = single.unwrap();
    let specials = specials.unwrap();
    let mut seen_sequence = false;
    for item in single {
        let Some(obj) = item.as_object() else {
            return Err("unsupported post_processor template item".to_string());
        };
        if let Some(seq) = obj.get("Sequence").and_then(Value::as_object) {
            if seen_sequence || member_str(seq, "id") != "A" {
                return Err("post_processor single template must hold one sequence A".to_string());
            }
            seen_sequence = true;
        } else if let Some(special) = obj.get("SpecialToken").and_then(Value::as_object) {
            let entry = specials
                .get(member_str(special, "id"))
                .and_then(Value::as_object);
            let ids = entry.and_then(|e| e.get("ids")).and_then(Value::as_array);
            let Some(ids) = ids else {
                return Err("post_processor references an unknown special token".to_string());
            };
            let target = if seen_sequence {
                &mut *suffix_ids
            } else {
                &mut *prefix_ids
            };
            for id_val in ids {
                if target.len() >= MAX_POST_PROCESSOR_IDS {
                    return Err("post_processor template has too many special tokens".to_string());
                }
                let Some(id) = read_id(id_val, max_token_id) else {
                    return Err("post_processor special token id out of range".to_string());
                };
                target
                    .try_reserve(1)
                    .map_err(|_| "out of memory".to_string())?;
                target.push(id);
            }
        } else {
            return Err("unsupported post_processor template item".to_string());
        }
    }
    if !seen_sequence {
        return Err("post_processor single template lacks sequence A".to_string());
    }
    Ok(())
}

fn read_opt_bool(obj: &Object, key: &str) -> (bool, bool, bool) {
    match obj.get(key) {
        None => (false, false, true),
        Some(v) => match v.as_bool() {
            Some(b) => (true, b, true),
            None => (true, false, false),
        },
    }
}

fn read_id(value: &Value, limit: u32) -> Option<u32> {
    let n = value.as_number().map(Number::value)?;
    if !n.is_finite() || n < 0.0 || n != n.floor() || n >= f64::from(limit) {
        return None;
    }
    Some(n as u32)
}

fn string_pattern(obj: &Object) -> String {
    obj.get("pattern")
        .and_then(Value::as_object)
        .and_then(|p| p.get("String"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| "\x01".to_string())
}

fn member_str<'a>(obj: &'a Object, key: &str) -> &'a str {
    obj.get(key).and_then(Value::as_str).unwrap_or("")
}

fn is_null(value: &Value) -> bool {
    matches!(value, Value::Null)
}

fn map_json_error(err: ValueError) -> ParseError {
    match err {
        ValueError::Rejected(reason) => ParseError::Invalid(format!("JSON parse error: {reason}")),
        ValueError::TooLarge => {
            ParseError::Invalid("tokenizer.json exceeds size limit".to_string())
        }
        ValueError::NoMemory => ParseError::Internal("out of memory".to_string()),
        ValueError::Internal => ParseError::Internal("internal JSON parser error".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_GEMMA: &str = r#"{
        "truncation": null,
        "padding": null,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
        "added_tokens": [{"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false}],
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>"}}, {"Sequence": {"id": "A"}}],
            "special_tokens": {"<bos>": {"ids": [2]}}
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "byte_fallback": true,
            "ignore_merges": false,
            "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "a": 1, "<bos>": 2, "b": 3, "ab": 4},
            "merges": [["a", "b"]]
        }
    }"#;

    #[test]
    fn parses_gemma_style_tokenizer() {
        let tok = parse(SAMPLE_GEMMA.as_bytes(), 1 << 24).unwrap();
        assert!(tok.meta.truncation_is_null);
        assert!(tok.meta.padding_is_null);
        assert_eq!(tok.meta.normalizer_type, "Replace");
        assert_eq!(tok.meta.normalizer_content, "▁");
        assert_eq!(tok.meta.pre_tokenizer_type, "Split");
        assert_eq!(tok.meta.model_type, "BPE");
        assert!(tok.meta.byte_fallback);
        assert_eq!(tok.vocab.len(), 5);
        assert_eq!(tok.merges.len(), 1);
        assert_eq!(tok.merges[0].left, "a");
        assert_eq!(tok.merges[0].right, "b");
        assert!(!tok.merges[0].was_string);
        assert_eq!(tok.added_tokens.len(), 1);
        assert_eq!(tok.prefix_ids, vec![2]);
        assert!(tok.suffix_ids.is_empty());
    }

    #[test]
    fn rejects_every_truncation_prefix() {
        let bytes = SAMPLE_GEMMA.as_bytes();
        for len in 0..bytes.len() {
            assert!(
                parse(&bytes[..len], 1 << 24).is_err(),
                "truncation at {len} should fail"
            );
        }
    }

    #[test]
    fn rejects_mutated_ids_and_merges() {
        assert!(parse(br#"{"model":{"vocab":{"a":-1}}}"#, 100).is_err());
        assert!(parse(br#"{"model":{"vocab":{"a":1.5}}}"#, 100).is_err());
        assert!(parse(br#"{"model":{"vocab":{"a":100}}}"#, 100).is_err());
        assert!(parse(br#"{"model":{"merges":["nospace"]}}"#, 100).is_err());
        assert!(parse(br#"{"model":{"merges":[["onlyone"]]}}"#, 100).is_err());
        assert!(parse(br#"{"added_tokens":[{"id":0,"content":""}]}"#, 100).is_err());
    }
}
