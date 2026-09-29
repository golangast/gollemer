#!/usr/bin/env python3
"""Targeted 10-case probe for the social model (kept at 10/10).

Reconstructs the original regression set (no scripted file survived):
  - 2 sentiment contrast on TRAINING inputs (happy / sad) with a
    polarity guard: the sad reply must not contain happy-words and
    vice versa (catches sentiment flips like "that's wonderful!" to
    "i am sad today").
  - 2 time-of-day (morning / evening)
  - 2 jokes on training inputs: must be joke-shaped ("?" + "!" +
    >= 6 words) - rejects word salad - plus a mode-separation guard.
  - 2 identity answers (name Gollemer, what it is)
  - 2 unseen paraphrases: mild rephrasings the pre-expansion model
    already answered coherently ("i am feeling down",
    "know any funny jokes"). Harsher unseen prompts ("share
    something humorous", "i am feeling joyful") produce word salad
    on the pre-expansion model too, so they are not regressions.

Every prompt runs in a FRESH chat session (one process per prompt),
matching how the original 10/10 was measured: the multi-turn session
memory feeds prior turns back into the model, which degrades later
turns and is covered separately by make eval-social-multiturn.
"""
import re, subprocess, sys, os

ROOT = "/home/hatch/workspace/gollemer"

CODE_MARKERS = [
    "func ", "package main", "package ", "import (", ":=", "go mod",
    "go run", "go build", "gofmt", "make train", "make real-chat",
    "{", "}", "func(", "// ",
]

HAPPY_WORDS = ["wonderful", "great", "fantastic", "amazing", "congratulations",
               "happy", "glad", "love", "thrilled"]
SAD_WORDS = ["sorry", "sad", "tough", "here for you", "talk about it", "listen"]


def joke_shaped(a):
    return ("?" in a) and a.rstrip().endswith("!") and len(a.split()) >= 6


CASES = [
    # sentiment contrast (training inputs) with polarity guards
    {"prompt": "i am happy!",
     "need": HAPPY_WORDS, "forbid": ["sorry"],
     "kind": "happy-contrast"},
    {"prompt": "i am sad today",
     "need": SAD_WORDS, "forbid": HAPPY_WORDS,
     "kind": "sad-contrast"},
    # time-of-day
    {"prompt": "good morning!",
     "need": ["morning"], "forbid": [],
     "kind": "morning"},
    {"prompt": "good evening!",
     "need": ["evening"], "forbid": [],
     "kind": "evening"},
    # jokes: must be joke-shaped, never word salad
    {"prompt": "tell me a joke.",
     "need": [], "forbid": [], "kind": "joke", "shape": joke_shaped},
    {"prompt": "do you know a joke?",
     "need": [], "forbid": [], "kind": "joke", "shape": joke_shaped},
    # identity
    {"prompt": "what is your name",
     "need": ["gollemer"], "forbid": [],
     "kind": "identity-name"},
    {"prompt": "what exactly are you?",
     "need": ["gollemer"], "forbid": [],
     "kind": "identity-what"},
    # unseen paraphrases (mild; pre-expansion model handled these)
    {"prompt": "i am feeling down",
     "need": SAD_WORDS, "forbid": HAPPY_WORDS,
     "kind": "paraphrase-sad"},
    {"prompt": "know any funny jokes",
     "need": [], "forbid": [], "kind": "paraphrase-joke", "shape": joke_shaped},
]


def run_one(prompt):
    env = dict(os.environ)
    env["PATH"] = env.get("PATH", "") + ":/home/hatch/golang/bin"
    env["GOEXPERIMENT"] = "simd"
    env["CGO_ENABLED"] = "1"
    p = subprocess.run(
        ["go", "run", "main.go", "-real-chat", "-domain", "social"],
        input=prompt + "\n/quit\n", capture_output=True, text=True,
        timeout=600, cwd=ROOT, env=env,
    )
    if p.returncode != 0:
        print("chat run failed:")
        print(p.stderr[-3000:])
        sys.exit(1)
    answers = re.findall(r"gollemer(?: \[[^\]]+\])?> ([^\n]*)", p.stdout)
    return answers[-1] if answers else ""


def main():
    answers = [run_one(c["prompt"]) for c in CASES]
    fails = []
    for c, a in zip(CASES, answers):
        low = a.lower()
        code = [m for m in CODE_MARKERS if m.lower() in low]
        bad_pol = [w for w in c["forbid"] if w in low]
        if "shape" in c:
            ok = bool(c["shape"](a))
            detail = "" if ok else "not joke-shaped"
        else:
            ok = any(n in low for n in c["need"])
            detail = "" if ok else f"missing one of {c['need']}"
        if bad_pol:
            ok = False
            detail += f" WRONG-POLARITY:{bad_pol}"
        if code:
            ok = False
            detail += f" CODE-MARKERS:{code}"
        mark = "PASS" if ok else "FAIL"
        if not ok:
            fails.append(c["kind"])
        print(f"[{mark}][{c['kind']}] {c['prompt']}{(' ' + detail) if detail else ''}\n       -> {a[:110]}")

    print(f"\ntargeted probe: {len(CASES) - len(fails)}/{len(CASES)}")
    if fails:
        print("misses:", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
