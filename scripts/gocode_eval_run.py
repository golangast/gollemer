#!/usr/bin/env python3
"""Run the fixed gocode eval suite through -real-chat non-interactively.

Scores two modes:
  code: every required substring present AND no chatter markers (no blended
        "I am func doing (x)" output).
  chat: conversational prompt gets a clean chat reply — no code markers.
"""
import json, subprocess, re, sys, os

ROOT = "/home/hatch/workspace/gollemer"
with open(os.path.join(ROOT, "scripts", "gocode_eval_cases.json")) as f:
    cases = json.load(f)

stdin_text = "/thoughts\n" + "\n".join(c["prompt"] for c in cases) + "\n/quit\n"
env = dict(os.environ)
env["PATH"] = env.get("PATH", "") + ":/home/hatch/golang/bin"
env["GOEXPERIMENT"] = "simd"
env["CGO_ENABLED"] = "1"
p = subprocess.run(
    ["go", "run", "main.go", "-real-chat", "-domain", "gocode"],
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


score = {"code": [0, 0], "chat": [0, 0]}
for c, a in zip(cases, answers):
    mode = c["mode"]
    need_ok = all(norm(n) in norm(a) for n in c["need"])
    non_empty = norm(a) != ""
    bad = has_forbidden(a, c["forbid"])
    if mode == "code":
        ok = need_ok and not bad
    else:
        ok = non_empty and not bad
    score[mode][1] += 1
    score[mode][0] += ok
    mark = "PASS" if ok else "FAIL"
    detail = ""
    if not need_ok:
        detail += " MISSING-need"
    if bad:
        detail += f" FORBIDDEN({c['forbid_kind']}):{bad}"
    if mode == "chat" and not non_empty:
        detail += " EMPTY"
    print(f"[{mark}][{mode}] {c['prompt']}{detail}\n       -> {a[:110]}")

tot_ok = score["code"][0] + score["chat"][0]
tot_n = score["code"][1] + score["chat"][1]
print(f"\nSCORE: {tot_ok}/{tot_n}  (code {score['code'][0]}/{score['code'][1]}, "
      f"chat {score['chat'][0]}/{score['chat'][1]})")
