"""Chat-template facts that bound how much of a generated turn history reuses.

Reuse is an exact token prefix. Where a model's own template renders a turn
differently from the tokens the model generated, the next request diverges at
that point and resumes from the stable boundary before it.
"""

# Gemma 4 (tokenizer.chat_template of the Unsloth GGUFs): with thinking off the
# generation prompt opens an empty thought, `<|turn>model\n<|channel>thought\n
# <channel|>`, but a model turn in history renders a thought only when it has
# text. The next request reuses up to the stable boundary before these seven
# tokens of the previous prompt, not the generated turn.
THINKING_OFF_OPENING_TOKENS = {"gemma4": 7}

# Gemma 4 renders a turn's reasoning only after the last user message (unless
# preserve_thinking keeps it on tool calls): a later user turn reuses the
# previous prompt, not the reasoning generated after it.
DROPS_EARLIER_REASONING = {"gemma4"}


# Gufo's Gemma 4 template shows a string `const` as a one-value `enum` (the
# official template drops `const`, and Gemma then deliberated far longer), so a
# control that renames `const` must show the same `enum`.
SHOWS_STRING_CONST_AS_ENUM = {"gemma4"}

# Gemma 4 calls carry typed values (`call:NAME{key:value}`), so a root `const`
# is enforced as that value; Qwen and DeepSeek tags declare no parameters for
# a root that is not a plain object, as llama.cpp's parsers do.
TYPED_NATIVE_CALLS = {"gemma4"}
