#!/usr/bin/env python3
"""Multi-turn conversation eval for the social model.

Exercises the real chat loop so the history prefix (SocialInput) is applied
exactly as in production: turn 1 is sent, its reply recorded, then turn 2 is
sent and the reply must show it used the context (keyword presence,
case-insensitive). Single-turn sanity cases guard against regression.
Recall cases run through unified chat and require exact transcript quotes.
"""
import json, re, subprocess, sys, os

GO = ["go", "run", "main.go"]
ENV = dict(os.environ)
ENV.update({"PATH": "/home/hatch/golang/bin:" + ENV.get("PATH", ""), "GOEXPERIMENT": "simd", "CGO_ENABLED": "1"})

NEURAL_CASES = [
    # multi-turn: turn 2 needs turn 1's context
    {"turn1": "i love rainy days", "turn2": "why do you like them", "keywords": ["rain"]},
    {"turn1": "i got a new puppy", "turn2": "her name is bella", "keywords": ["bella"]},
    {"turn1": "i am going on vacation", "turn2": "to the beach", "keywords": ["beach"]},
    {"turn1": "i made cookies", "turn2": "chocolate chip", "keywords": ["chocolate"]},
    {"turn1": "do you like music", "turn2": "what kind of music", "keywords": ["music", "song"]},
    {"turn1": "i am sad today", "turn2": "i just miss my friend", "keywords": ["friend", "miss"]},
    # single-turn sanity (no regression)
    {"turn1": None, "turn2": "hello", "keywords": ["hello", "hey", "hi"]},
    {"turn1": None, "turn2": "what is your name", "keywords": ["gollemer"]},
]

RECALL_CASES = [
    {"turns": ["hello there", "what did I just say"], "want": 'you said: "hello there"'},
    {"turns": ["hello there", "what did you just say"], "want_prefix": "i said: "},
]


def run_chat(domain, lines):
    p = subprocess.run(
        GO + ["-real-chat", "-domain", domain],
        input="\n".join(lines + ["/quit"]) + "\n",
        capture_output=True, text=True, env=ENV, cwd="/home/hatch/workspace/gollemer",
        timeout=300,
    )
    return re.findall(r"gollemer(?: \[[^\]]+\])?> ([^\n]*)", p.stdout)


def main():
    fails = []
    # neural multi-turn + single-turn through the social model
    for c in NEURAL_CASES:
        lines = ([c["turn1"]] if c["turn1"] else []) + [c["turn2"]]
        answers = run_chat("social", lines)
        reply = answers[-1].lower() if answers else ""
        if not any(k in reply for k in c["keywords"]):
            fails.append((c["turn2"], reply, c["keywords"]))
            print(f"[FAIL] {c['turn2']!r} -> {reply!r} (want one of {c['keywords']})")
        else:
            print(f"[PASS] {c['turn2']!r}")
    # deterministic recall through unified chat
    for c in RECALL_CASES:
        answers = run_chat("unified", c["turns"])
        reply = answers[-1] if answers else ""
        ok = (reply == c["want"]) if "want" in c else reply.startswith(c["want_prefix"])
        if not ok:
            fails.append((c["turns"][-1], reply, c.get("want", c.get("want_prefix"))))
            print(f"[FAIL] recall {c['turns'][-1]!r} -> {reply!r}")
        else:
            print(f"[PASS] recall {c['turns'][-1]!r}")
    print(f"\nmultiturn: {len(NEURAL_CASES) + len(RECALL_CASES) - len(fails)}/{len(NEURAL_CASES) + len(RECALL_CASES)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
