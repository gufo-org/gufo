#!/usr/bin/env python3
"""Generates the Inkling 2 text goldens from the official artifacts.

  python -I gen_goldens.py OFFICIAL_DIR WIKI_RAW OUT_DIR

Needs: tokenizers, tiktoken, transformers (+jinja2). Tokenization expectations come
from the HF tokenizer.json (special tokens parsed) and tiktoken (plain text);
the script cross-checks the two and reports disagreements. Rendering expectations
come from transformers' apply_chat_template on the official chat_template.jinja,
except where the template's Python spelling of numbers differs from Thinking
Machines' renderer (tml-renderers), which Gufo, vLLM and SGLang follow: see
TML_SPELLED.
"""
import base64
import json
import random
import sys
import unicodedata

import tiktoken
from tokenizers import Tokenizer
from transformers import AutoTokenizer

official, wiki_path, out_dir = sys.argv[1:4]
rng = random.Random(20261121)

hf = Tokenizer.from_file(f"{official}/tokenizer.json")
ranks = {}
for line in open(f"{official}/tiktoken/tokenizer.model"):
    a, b = line.split()
    ranks[base64.b64decode(a)] = int(b)
specials = {t["content"]: t["id"] for t in json.load(open(f"{official}/tokenizer.json"))["added_tokens"]}
PAT = json.load(open(f"{official}/tokenizer.json"))["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
# tiktoken needs a pattern its regex engine accepts; (?i:...) is supported by `regex`/fancy-regex.
tt = tiktoken.Encoding(name="inkling2", pat_str=PAT,
                       mergeable_ranks={k: v for k, v in ranks.items() if v < 199998},
                       special_tokens=specials)

# ---------------- tokenization cases ----------------
cases = []
disagree = []


def add(text, special=True, note=""):
    if special:
        ids = hf.encode(text, add_special_tokens=False).ids
        if not any(s in text for s in specials):
            t = tt.encode_ordinary(text)
            if t != ids:
                disagree.append((note, text[:80]))
    else:
        ids = tt.encode_ordinary(text)
    cases.append({"text": text, "special": special, "ids": ids, "note": note})


wiki = [l for l in open(wiki_path, encoding="utf-8").read().split("\n") if l.strip()]
step = max(1, len(wiki) // 900)
for i in range(0, len(wiki), step):
    add(wiki[i][:4000], note="wiki")
raw = open(wiki_path, encoding="utf-8").read()
for i in range(40):
    s = rng.randrange(0, len(raw) - 3000)
    add(raw[s:s + rng.randrange(200, 3000)], note="wiki-chunk")  # crosses line breaks

words = ["Hello", "world", "naïve", "café", "Straße", "ΑΒΓ αβγ", "Привет мир", "שלום עולם", "مرحبا بالعالم",
         "สวัสดีชาวโลก", "नमस्ते दुनिया", "你好，世界", "こんにちは世界", "안녕하세요 세계", "ǅ", "ǆ", "ǈ", "ǋ", "ʰʱ", "ｈｅｌｌｏ"]
for w in words:
    add(w, note="unicode")
    add(" " + w + " ", note="unicode")
    add(w.upper() + w.lower(), note="unicode-case")
combining = ["é", "ä́", "क्ष", "กิ้", "각", "ÀB́",
             "́", "x⃝", "ःabc", "a‍b", "​‌‍", "﻿", "­", "é́́"]
for c in combining:
    add(c, note="marks")
    add("The " + c + " test", note="marks")
emoji = ["😀", "👍🏽", "👨‍👩‍👧‍👦", "🇬🇧🇫🇷", "❤️", "©️", "1️⃣", "🏳️‍🌈", "🧑🏿‍🦰", "☺", "✔️✨", "🫠🥹", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "𠮷野家", "𐍈"]
for e in emoji:
    add(e, note="emoji")
    add("ok " + e + e + " done!", note="emoji")
numbers = ["0", "7", "42", "123", "1234", "12345", "1234567890", "3.14159", "-273.15", "1,000,000", "1e-5", "0xDEADBEEF",
           "2026-11-21", "12:34:56", "٣٤٥", "१२३४", "１２３４", "①②③", "½¼¾", "Ⅷ", "x1y22z333w4444", "99999999999999999999",
           "1.5.2-rc.1", "+44 7700 900123", "$1,234.56", "50%", "10^9", "a123b"]
for n in numbers:
    add(n, note="numbers")
    add("n = " + n + ";", note="numbers")
ws = ["", " ", "  ", "   ", "\t", "\t\t", "\n", "\n\n", "\n\n\n", "\r\n", "\r\n\r\n", "\r", " \n", "\n ", " \n ", "  \n\n  ",
      "a \n b", "a  b", "a   b", "a\tb", "a\t\tb", "a \t b", "a\n\n\nb", "a\r\nb", "a \r\n b", " ", "a b", " x ",
      "a b", "a b", "\x0b\x0c", "a\x0bb", "　x　y", "\u0085", "a\u0085b", "x ​ y", "trailing   ", "   leading",
      "end\n", "end\n\n", "end \n", "a\n  b\n    c", "\n\n  x", " \n\n", "\t\n\t", "     "]
for w in ws:
    add(w, note="whitespace")
    add("x" + w + "y", note="whitespace")
    add("x" + w, note="whitespace")
    add(w + "y", note="whitespace")
contr = ["don't", "DON'T", "Don'T", "I'm", "I'M", "we're", "WE'RE", "you've", "they'll", "she'd", "it's", "IT'S", "rock'n'roll",
         "'s", "'", "''", "'tis", "a'b", "O'Reilly", "ſ", "it'ſ", "x'ſy", "’s", "don’t", "'re", "'ll'd'm'", "can't've", "y'all'd've",
         "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D", "'ReX", "'sa", "'tt"]
for c in contr:
    add(c, note="contraction")
    add("x " + c + " y", note="contraction")
code = [
    "def foo(x, y=None):\n    return x + (y or 0)  # comment\n",
    "#include <stdio.h>\nint main(void) {\n  printf(\"%d\\n\", 42);\n  return 0;\n}\n",
    "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"units\":[\"c\",\"f\"],\"n\":1.5e3}}",
    "<html><body class=\"a b\"><p>Hi &amp; bye</p></body></html>",
    "fn main() -> Result<(), Box<dyn Error>> { let v: Vec<u8> = vec![0; 16]; Ok(()) }",
    "SELECT a.id, COUNT(*) FROM t a JOIN u b ON a.id=b.id WHERE a.x >= 10 GROUP BY 1;",
    "camelCaseIdentifier snake_case_identifier SCREAMING_SNAKE_CASE kebab-case-id HTTPServerError XMLHttpRequest",
    "https://example.com/a/b?c=d&e=f#frag  user@example.org  /usr/local/bin/python3.12",
    "$$\\sum_{i=0}^{n} x_i^2 \\leq \\int_0^1 f(x)\\,dx$$",
    "    if (x) {\n\t\ty();\n    }\n\n\n    return;",
    "a" * 300, "ab" * 200, " " * 100 + "x", "\n" * 50, "=" * 80, "-" * 3 + ">" + "<" * 3, "x" + "?" * 20 + "!" * 20,
    "...!!!???---___***", "a/b/c//d///e", "x\n/\n/y", "! \n/ ?/\r\n", ".//\n", "a.\n/b",
    "word" + "é" * 50, "Z" * 70 + "z" * 70, "ÀÉÎÕÜ àéîõü", "ǅǅ", "ΣΣ σς", "İi ıI", "ǰ ß ẞ",
]
for c in code:
    add(c, note="code")
# special-token text
for s in specials:
    if "unused_" in s and s != "<|unused_200054|>":
        continue
    add(s, special=True, note="special")
    add("x" + s + "y", special=True, note="special")
    add("<|message_user|>" + s + "\n", special=True, note="special")
    add(s, special=False, note="special-plain")
    add("a " + s + " b", special=False, note="special-plain")
add("<|message_user|><|content_text|>Hi<|end_message|><|message_model|>", special=True, note="special-seq")
add("<|message_user|><|content_text|>Hi<|end_message|><|message_model|>", special=False, note="special-seq-plain")
add("<|end_message|<|end_message|>>", special=True, note="special-edge")
add("<|end_message", special=True, note="special-edge")
add("<|not_a_token|>", special=True, note="special-edge")
add("<|content_text|><|content_text|>", special=True, note="special-edge")
add("<<|end_message|>>", special=True, note="special-edge")
add("<|unused|><|unused_200053|><|unused_200055|>", special=True, note="special-edge")

# fuzz: structured pools
pools = {
    "lower": "abcdefghijklmnopqrstuvwxyz", "upper": "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "digit": "0123456789",
    "punct": "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'", "space": " \t\n\r  　\x0b\x0c\u0085 ",
    "latin": "éèêëàâäôöûüçñßøåæœÉÈÀÖÜÇÑ", "greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛ",
    "cyr": "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛ", "cjk": "你好世界中文漢字日本語한국어こんにちはカタカナ",
    "marks": "̧́̈̀ःा्ัิ‍‌⃝", "sym": "€£¥©®™°±×÷∑∏√∞≈≠≤≥←→↑↓★☆♥♦♠♣",
    "emoji": "😀😂🥲🔥👍🎉🚀🌍💯🤔🙏", "num": "٠١٢٣٤٥٦٧٨٩०१२३४५६७८९½¼¾²³¹ⅣⅤ①②",
    "apos": "'''’", "mix": "sSdDtTmMlLrReEvV",
}
keys = list(pools)
for i in range(700):
    n = rng.randrange(1, 40)
    s = ""
    for _ in range(n):
        k = rng.choice(keys)
        s += "".join(rng.choice(pools[k]) for _ in range(rng.randrange(1, 6)))
    add(s, note="fuzz-pool")
# fuzz: random code points of the whole assigned range
assigned = [c for c in range(0x20, 0x30000) if not (0xD800 <= c <= 0xDFFF) and unicodedata.category(chr(c)) not in ("Cn", "Co", "Cs", "Cc")]
for i in range(150):
    s = "".join(chr(rng.choice(assigned)) for _ in range(rng.randrange(1, 30)))
    add(s, note="fuzz-cp")
# random byte-level-ish: mixed words with random spacing from wiki
toks = raw.split()
for i in range(150):
    s = " ".join(rng.choice(toks) for _ in range(rng.randrange(1, 40)))
    if rng.random() < 0.3:
        s = s.replace(" ", rng.choice(["  ", "\n", " \n", "\t", ""]), rng.randrange(1, 6))
    add(s, note="fuzz-words")

# decode cases: random id sequences (tiktoken decode, bytes as hex)
dec = []
for i in range(300):
    n = rng.randrange(1, 40)
    ids = [rng.choice([rng.randrange(0, 256), rng.randrange(0, 199998), rng.randrange(199998, 200058)]) for _ in range(n)]
    b = b"".join(tt.decode_single_token_bytes(t) for t in ids)
    dec.append({"ids": ids, "hex": b.hex()})

json.dump({"pattern": PAT, "cases": cases, "decode_cases": dec}, open(f"{out_dir}/tokenization_cases.json", "w"), ensure_ascii=False)
print("tokenization cases:", len(cases), "decode cases:", len(dec), "hf/tiktoken disagreements:", len(disagree))
for d in disagree[:20]:
    print("  DISAGREE", d)

# ---------------- rendering cases ----------------
tok = AutoTokenizer.from_pretrained(official)
render = []


def fn(name, desc="Does a thing.", params=None, wrap=True, **extra):
    f = {"name": name, "description": desc, "parameters": params if params is not None else {"type": "object", "properties": {}}}
    f.update(extra)
    return {"type": "function", "function": f} if wrap else f


WEATHER = fn("get_weather", "Get the current weather for a city.",
             {"type": "object", "required": ["city"], "properties": {"unit": {"type": "string", "enum": ["c", "f"], "description": "Temperature unit"},
                                                                    "city": {"type": "string", "description": "City name, e.g. Paris"}}})
SEARCH = fn("web_search", "Search the web — café ☕ 你好", {"type": "object", "properties": {"q": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                                                                                    "filters": {"type": "object", "properties": {"z": {"type": "boolean"}, "a": {"type": "array", "items": {"type": "number"}}}}}})
U = lambda t: {"role": "user", "content": t}
S = lambda t: {"role": "system", "content": t}
A = lambda t, **k: {"role": "assistant", "content": t, **k}
TC = lambda id, name, args: {"id": id, "type": "function", "function": {"name": name, "arguments": args}}
T = lambda t, **k: {"role": "tool", "content": t, **k}
IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


# Cases whose expectation is tml-renderers' spelling, not the Jinja template's:
# floats positional for decimal exponents in [-5, 16), unpadded exponents with
# a sign only in tool declarations, and the effort quantized to hundredths.
# Checked against tml-renderers 0.5.0 Inkling2(), which refuses the integers
# over 64 bits these cases hold (checked with 64-bit stand-ins); Gufo, like the
# template, keeps them.
TML_SPELLED = {
    "tool_floats": "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"d\",\"name\":\"f\",\"parameters\":{\"properties\":{\"x\":{\"default\":1.0,\"maximum\":1e+21,\"minimum\":-0.5,\"multipleOf\":1e-7,\"t\":0.1,\"type\":\"number\",\"w\":123456789.12345679,\"z\":100000000000000000000}},\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>x<|end_message|>",
    "tool_call_arg_types": "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>x<|end_message|><|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":[1,2.5,\"s\",null,true,{\"y\":1,\"b\":[3]}],\"big\":12345678901234567890123,\"e\":\"\",\"exp\":1e-7,\"exp2\":1e22,\"f\":false,\"fl2\":1.2345678901234568e17,\"float\":0.1,\"i\":1000000000000000,\"m\":{\"k2\":\"v\",\"k1\":1.0},\"n\":null,\"neg\":-0.0,\"u\":\"héllo 你好 😀 \\\"q\\\" \\\\ \\n\\t \\u0001\",\"z\":1}}<|end_message|><|content_model_end_sampling|>",
    "effort_tiny": "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>hi<|end_message|>",
    "effort_long": "<|message_system|><|content_text|>Thinking effort level: 0.12<|end_message|><|message_user|><|content_text|>hi<|end_message|>"
}


def case(name, messages, tools=None, effort="__unset__", cpp_skip=False):
    kw = {}
    if effort != "__unset__":
        kw["reasoning_effort"] = effort
    exp = None
    err = False
    toks = None
    try:
        exp = tok.apply_chat_template(messages, tools=tools, tokenize=False, **kw)
        exp = TML_SPELLED.get(name, exp)
        toks = hf.encode(exp, add_special_tokens=False).ids
    except Exception as e:  # raise_exception from the template
        err = True
        exp = str(e)[:200]
    c = {"name": name, "messages": messages, "error": err, "expected": exp, "tokens": toks}
    if tools is not None:
        c["tools"] = tools
    if effort != "__unset__":
        c["reasoning_effort"] = effort
    if cpp_skip:
        c["cpp_skip"] = True
    if name in TML_SPELLED:
        c["spelling"] = "tml-renderers"
    render.append(c)


para = wiki[rng.randrange(len(wiki))][:600]
case("single_user", [U("Hello!")])
case("user_wiki", [U(para)])
case("system_user", [S("You are a terse assistant."), U("Hi")])
case("multi_system", [S("Rule one."), S("Rule two."), U("Go")])
case("system_after_user", [U("a"), S("late system"), U("b")])
case("system_only", [S("Only a system prompt.")])
case("empty_messages", [])
case("multi_turn", [S("Be brief."), U("What is 2+2?"), A("4"), U("And 3+3?"), A("6"), U("Thanks")])
case("assistant_reasoning", [U("Why is the sky blue?"), A("Rayleigh scattering.", reasoning_content="Short wavelengths scatter more.")])
case("assistant_reasoning_only", [U("hmm"), A("", reasoning_content="thinking only")])
case("assistant_reasoning_none_content", [U("hmm"), {"role": "assistant", "content": None, "reasoning_content": "t"}])
case("assistant_empty", [U("hi"), A("")])
case("assistant_none_content", [U("hi"), {"role": "assistant", "content": None}])
case("user_empty", [U("")])
case("assistant_name", [U("hi"), A("hello", name="Ada")])
case("tool_declare_one", [U("Weather in Paris?")], tools=[WEATHER])
case("tool_declare_two", [S("sys"), U("search or weather")], tools=[WEATHER, SEARCH])
case("tool_bare_function", [U("x")], tools=[fn("bare_fn", "bare", wrap=False)])
case("tool_no_description", [U("x")], tools=[{"type": "function", "function": {"name": "nodesc", "parameters": {"type": "object", "properties": {"b": {"type": "string"}, "a": {"type": "number"}}}}}])
case("tool_no_parameters", [U("x")], tools=[{"type": "function", "function": {"name": "noparams", "description": "d"}}])
case("tool_empty_description_params", [U("x")], tools=[{"type": "function", "function": {"name": "e", "description": "", "parameters": {}}}])
case("tool_unicode_escape", [U("x")], tools=[fn("u", "quote \" backslash \\ tab \t nl \n bell \u0007 emoji 😀 del \u007f", {"type": "object", "properties": {"é": {"type": "string", "description": "ü"}}})])
case("tool_empty_list", [U("x")], tools=[])
case("tool_floats", [U("x")], tools=[fn("f", "d", {"type": "object", "properties": {"x": {"type": "number", "default": 1.0, "minimum": -0.5, "maximum": 1e21, "multipleOf": 1e-7, "z": 100000000000000000000, "t": 0.1, "w": 123456789.123456789}}})])
case("tool_call_basic", [U("Weather?"), A(None, tool_calls=[TC("c1", "get_weather", {"city": "Paris", "unit": "c"})]), T("18C and sunny", tool_call_id="c1"), A("It is 18C.")], tools=[WEATHER])
case("tool_call_with_content", [U("Weather?"), A("Let me check.", tool_calls=[TC("c1", "get_weather", {"city": "Paris"})])], tools=[WEATHER])
case("tool_call_with_reasoning", [U("Weather?"), A("", reasoning_content="I should call the tool.", tool_calls=[TC("c1", "get_weather", {"city": "Paris"})]), T("rain", tool_call_id="c1")], tools=[WEATHER])
case("tool_call_empty_string_content", [U("Weather?"), A("", tool_calls=[TC("c1", "get_weather", {"city": "Rome"})])], tools=[WEATHER])
case("tool_call_parallel", [U("both"), A(None, tool_calls=[TC("a", "get_weather", {"city": "A"}), TC("b", "web_search", {"q": "x"})]), T("wa", tool_call_id="a"), T("wb", tool_call_id="b")], tools=[WEATHER, SEARCH])
case("tool_call_arg_types", [U("x"), A(None, tool_calls=[TC("c", "f", {"z": 1, "a": [1, 2.5, "s", None, True, {"y": 1, "b": [3]}], "m": {"k2": "v", "k1": 1.0}, "e": "", "n": None, "f": False,
                                                                          "big": 12345678901234567890123, "neg": -0.0, "exp": 1e-7, "exp2": 1e22, "u": "héllo 你好 😀 \"q\" \\ \n\t \u0001", "float": 0.1, "i": 10 ** 15, "fl2": 123456789012345680.0})])])
case("tool_call_key_sort", [U("x"), A(None, tool_calls=[TC("c", "f", {"b": 1, "B": 2, "a": 3, "A": 4, "_": 5, "é": 6, "z": 7, "10": 8, "2": 9, "ab": 10, "a b": 11})])])
case("tool_call_empty_args", [U("x"), A(None, tool_calls=[TC("c", "noargs", {})])])
case("tool_call_missing_args", [U("x"), A(None, tool_calls=[{"id": "c", "type": "function", "function": {"name": "noargs"}}])])
case("tool_call_name_escape", [U("x"), A(None, tool_calls=[TC("c", 'we"ird\\name', {"k": 1})])])
case("tool_call_string_args_error", [U("x"), A(None, tool_calls=[TC("c", "f", '{"a":1}')])], cpp_skip=True)
case("tool_call_name_mismatch", [U("x"), A(None, name="Bob", tool_calls=[TC("c", "f", {})])])
case("tool_call_name_match", [U("x"), A(None, name="f", tool_calls=[TC("c", "f", {"a": 1})])])
case("tool_result_name", [U("x"), T("res", name="my_tool")])
case("tool_result_unknown_id", [U("x"), T("res", tool_call_id="nope")])
case("tool_result_no_id", [U("x"), T("res")])
case("tool_result_name_over_id", [U("x"), A(None, tool_calls=[TC("c", "f", {})]), T("res", name="other", tool_call_id="c")])
case("tool_result_dup_ids", [U("x"), A(None, tool_calls=[TC("c", "first", {})]), A(None, tool_calls=[TC("c", "second", {})]), T("res", tool_call_id="c")])
case("tool_result_json", [U("x"), A(None, tool_calls=[TC("c", "f", {})]), T(json.dumps({"temp": 18, "ok": True}), tool_call_id="c")])
case("image_user", [U([{"type": "text", "text": "What is this?"}, IMG])])
case("image_first", [U([IMG, {"type": "text", "text": "Describe."}])])
case("image_only", [U([IMG])])
case("image_two", [U([{"type": "text", "text": "a"}, IMG, {"type": "text", "text": "b"}, IMG])])
case("image_input_image_type", [U([{"type": "input_image", "image_url": "x"}, {"type": "input_text", "text": "t"}])])
case("image_type_image", [U([{"type": "image"}])])
case("image_in_tool", [U("x"), A(None, tool_calls=[TC("c", "shot", {})]), T([{"type": "text", "text": "screenshot"}, IMG], tool_call_id="c")])
case("image_in_assistant_error", [U("x"), A([IMG])])
case("image_in_system_error", [S([IMG]), U("x")])
case("audio_error", [U([{"type": "input_audio", "input_audio": {}}])])
case("unsupported_part_error", [U([{"type": "video"}])])
case("parts_plain_strings", [U(["one", "two"])])
case("parts_no_type", [U([{"text": "no type key"}])])
case("parts_text_not_string", [U([{"type": "text", "text": 5}])])
case("parts_empty_list", [U([])])
case("parts_assistant_text", [U("q"), A([{"type": "text", "text": "ans"}])])
case("unknown_role", [{"role": "developer", "content": "x"}])
case("unknown_role_after", [U("a"), {"role": "function", "content": "x"}])
for e in ["none", "minimal", "low", "medium", "high", "xhigh", "max"]:
    case("effort_" + e, [U("hi")], effort=e)
for e, nm in [(0, "int0"), (0.0, "zero"), (0.5, "half"), (0.33, "third"), (0.99, "top"), (0.1, "tenth"), (1e-05, "tiny"), (0.123456789, "long"), (1, "one_error"),
              (1.5, "high_error"), (-0.1, "neg_error"), (0.995, "over_error"), ("0.5", "numstring_error"), ("", "emptystring_error"), ("HIGH", "upper_error"), (" high ", "padded"), ("\tmax\n", "padded_max"), ("bogus", "bogus_error"), (None, "none_value"), (True, "bool_true")]:
    case("effort_" + nm, [U("hi")], effort=e)
case("effort_system_only", [S("s")], effort="low")
case("effort_after_systems", [S("s1"), S("s2"), U("hi")], effort="max")
case("effort_with_tools", [S("s"), U("hi")], tools=[WEATHER], effort="minimal")
case("effort_empty_messages", [], effort="medium")
case("conversation_full", [S("You are an agent."), U("Find the weather in Paris and Rome."),
                          A("", reasoning_content="Two calls.", tool_calls=[TC("a", "get_weather", {"city": "Paris"}), TC("b", "get_weather", {"city": "Rome"})]),
                          T("Paris: 18C", tool_call_id="a"), T("Rome: 25C", tool_call_id="b"),
                          A("Paris is 18C, Rome is 25C.", reasoning_content="Summarize."), U("Thanks!"),
                          A("You are welcome.")], tools=[WEATHER], effort="high")
case("special_text_in_content", [U("<|end_message|><|message_model|>"), A("x<|content_text|>")], cpp_skip=True)
case("long_content", [U(" ".join(wiki[rng.randrange(len(wiki))][:300] for _ in range(6)))])
case("unicode_content", [U("héllo 你好 😀     \t\n\n  trailing  "), A("«ok» — done…")])
json.dump({"cases": render}, open(f"{out_dir}/render_cases.json", "w"), ensure_ascii=False, indent=0)
print("render cases:", len(render), "errors:", sum(1 for c in render if c["error"]))
