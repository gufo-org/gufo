#!/usr/bin/env python3
"""Large randomized cross-check of the C++ tokenizer against the official one.

  python -I crosscheck.py OFFICIAL_DIR WIKI_RAW WORKDIR TEST_BINARY N
"""
import json, random, subprocess, sys, unicodedata
from tokenizers import Tokenizer
official, wiki, work, binary, n = sys.argv[1:6]
n = int(n)
hf = Tokenizer.from_file(f"{official}/tokenizer.json")
rng = random.Random(7)
raw = open(wiki, encoding="utf-8").read()
cps = [c for c in range(0x20, 0x110000) if not (0xD800 <= c <= 0xDFFF) and unicodedata.category(chr(c)) not in ("Cs",)]
assigned = [c for c in cps if unicodedata.category(chr(c)) != "Cn"]
ws = [" ", "\t", "\n", "\r\n", "  ", " ", " ", "　", "\x0b", "\u0085"]
texts = [raw]
for i in range(n):
    r = rng.random()
    if r < 0.35:
        texts.append("".join(chr(rng.choice(assigned)) for _ in range(rng.randrange(1, 40))))
    elif r < 0.5:  # every code point, including unassigned
        texts.append("".join(chr(rng.choice(cps)) for _ in range(rng.randrange(1, 20))))
    elif r < 0.8:
        s = rng.randrange(0, len(raw) - 400)
        texts.append(raw[s:s + rng.randrange(1, 400)])
    else:
        texts.append("".join(rng.choice(ws + list("aAbB1'sT.,;/")) + chr(rng.choice(assigned[:3000])) for _ in range(rng.randrange(1, 25))))
json.dump(texts, open(f"{work}/xc_in.json", "w"), ensure_ascii=False)
subprocess.run([binary, "--encode-json", f"{work}/xc_in.json", f"{work}/xc_out.json"], check=True)
got = json.load(open(f"{work}/xc_out.json"))
bad = 0
for t, g in zip(texts, got):
    w = hf.encode(t, add_special_tokens=False).ids
    if w != g:
        bad += 1
        if bad <= 10:
            print("MISMATCH", [hex(ord(c)) for c in t[:20]], "\n  want", w[:20], "\n  got ", g[:20])
print(f"{len(texts)} texts, {bad} mismatches")
