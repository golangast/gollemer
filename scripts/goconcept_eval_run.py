#!/usr/bin/env python3
"""Run the fixed go-concept eval suite through -real-chat non-interactively.

Scores one mode:
  concept: every required key term present AND no code-syntax markers
           (the go model explains in prose; it must never emit code).
"""
import json, subprocess, re, sys, os

ROOT = "/home/hatch/workspace/gollemer"
with open(os.path.join(ROOT, "scripts", "goconcept_eval_cases.json")) as f:
    cases = json.load(f)

stdin_text = "\n".join(c["prompt"] for c in cases) + "\n/quit\n"
env = dict(os.environ)
env["PATH"] = env.get("PATH", "") + ":/home/hatch/golang/bin"
env["GOEXPERIMENT"] = "simd"
env["CGO_ENABLED"] = "1"
p = subprocess.run(
    ["go", "run", "main.go", "-real-chat", "-domain", "go"],
    input=stdin_text, capture_output=True, text=True, timeout=900,
    cwd=ROOT, env=env,
)
if p.returncode != 0:
    print("chat run failed:")
    print(p.stderr[-3000:])
    sys.exit(1)

answers = re.findall(r"gollemer> (.*?)(?:\nyou> |\Z)", p.stdout, re.S)
if len(answers) > len(cases):
    answers = answers[-len(cases):]
if len(answers) != len(cases):
    print(f"answer/case mismatch: {len(answers)} answers for {len(cases)} cases")
    print(p.stdout[-2000:])
    sys.exit(1)


def norm(s):
    return re.sub(r"\s+", "", s.lower())


def has_forbidden(text, words):
    low = text.lower()
    return [w for w in words
            if re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low)]


ok_count = 0
for c, a in zip(cases, answers):
    need_ok = all(norm(n) in norm(a) for n in c["need"])
    non_empty = norm(a) != ""
    bad = has_forbidden(a, c["forbid"])
    ok = need_ok and non_empty and not bad
    ok_count += ok
    mark = "PASS" if ok else "FAIL"
    detail = ""
    if not need_ok:
        detail += " MISSING-need"
    if bad:
        detail += f" FORBIDDEN({c['forbid_kind']}):{bad}"
    if not non_empty:
        detail += " EMPTY"
    print(f"[{mark}][{c['mode']}] {c['prompt']}{detail}\n       -> {a[:110]}")

print(f"\nSCORE: {ok_count}/{len(cases)}")
