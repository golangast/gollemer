#!/usr/bin/env python3
"""Fixed eval suite for the gocode domain.

Two kinds of cases:
  code: prompt must produce the required code substrings, and the output
        must not contain chatter markers (no blended "I am func doing (x)").
  chat: conversational prompt must get a clean chat reply — no code markers
        anywhere in the output (mode separation).
"""
import json

# (prompt, required substrings) for code cases
CODE_CASES = [
    ("write a function that adds two ints",
     ["func add", "return a + b"]),
    ("give me a go function to subtract b from a",
     ["func sub", "return a - b"]),
    ("i need a multiply function in go",
     ["func mul", "return a * b"]),
    ("write a function returning the larger of two ints",
     ["func max", "return a", "return b"]),
    ("write a function checking if a number is even",
     ["func isEven", "n % 2 == 0"]),
    ("write a hello world program in go",
     ["func main", "hello world"]),
    ("write a function summing a slice of ints",
     ["func sum", "range nums", "return total"]),
    ("write a recursive factorial function in go",
     ["func factorial", "factorial(n-1)"]),
    ("write a rectangle struct in go",
     ["type rect struct", "float64"]),
    ("write a divide function returning an error on zero",
     ["func divide", "errors.New", "divide by zero"]),
    # unseen paraphrases (not in training inputs)
    ("code me an adder for integers",
     ["func add", "return a + b"]),
    ("how do i check even numbers in go",
     ["func isEven", "% 2 == 0"]),
]

# conversational prompts: must get a clean chat reply, never code
CHAT_CASES = [
    "hello",
    "how are you",
    "thank you",
    "tell me a joke",
    "good morning",
]

# Chatter that must never appear in a code answer (word-boundary matched).
CHATTER_FORBIDDEN = [
    "i am", "i'm", "here is", "here's", "sure", "of course",
    "hope this helps", "let me know", "you're welcome", "happy to help",
]

# Code markers that must never appear in a chat answer.
CODE_FORBIDDEN = ["func", "{", "}", ":=", "package", "struct"]

cases = (
    [{"prompt": p, "mode": "code", "need": n,
      "forbid": CHATTER_FORBIDDEN, "forbid_kind": "chatter"}
     for p, n in CODE_CASES]
    + [{"prompt": p, "mode": "chat", "need": [],
        "forbid": CODE_FORBIDDEN, "forbid_kind": "code"}
       for p in CHAT_CASES]
)

with open("/home/hatch/workspace/gollemer/scripts/gocode_eval_cases.json", "w") as f:
    json.dump(cases, f, indent=1)
print(f"wrote {len(cases)} eval cases ({len(CODE_CASES)} code, {len(CHAT_CASES)} chat)")
