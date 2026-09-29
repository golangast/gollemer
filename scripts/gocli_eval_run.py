#!/usr/bin/env python3
"""Run the fixed Go-CLI eval suite through -real-chat non-interactively.

Scores exact command match: the gocli model must emit the bare command
and nothing else. Whitespace/case normalized; any chatter fails.
Reports seen (memorization) and paraphrase (generalization) separately.
"""
import json, subprocess, re, sys, os

ROOT = "/home/hatch/workspace/gollemer"
with open(os.path.join(ROOT, "scripts", "gocli_eval_cases.json")) as f:
    cases = json.load(f)

stdin_text = "\n".join(c["prompt"] for c in cases) + "\n/quit\n"
env = dict(os.environ)
env["PATH"] = env.get("PATH", "") + ":/home/hatch/golang/bin"
env["GOEXPERIMENT"] = "simd"
env["CGO_ENABLED"] = "1"
p = subprocess.run(
    ["go", "run", "main.go", "-real-chat", "-domain", "gocli"],
    input=stdin_text, capture_output=True, text=True, timeout=900,
    cwd=ROOT, env=env,
)
if p.returncode != 0:
    print("chat run failed:")
    print(p.stderr[-3000:])
    sys.exit(1)

# Each reply is "gollemer> <command>" followed by the optional thought
# trace on later lines; only the first line is the model's answer.
answers = re.findall(r"gollemer> ([^\n]*)", p.stdout)
if len(answers) > len(cases):
    answers = answers[-len(cases):]
if len(answers) != len(cases):
    print(f"answer/case mismatch: {len(answers)} answers for {len(cases)} cases")
    print(p.stdout[-2000:])
    sys.exit(1)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


seen_ok = seen_n = par_ok = par_n = 0
for c, a in zip(cases, answers):
    ok = norm(a) == norm(c["expect"])
    mark = "PASS" if ok else "FAIL"
    if c["kind"] == "seen":
        seen_n += 1
        seen_ok += ok
    else:
        par_n += 1
        par_ok += ok
    print(f"[{mark}] ({c['kind']}) {c['prompt']!r}")
    if not ok:
        print(f"       got: {a!r}  want: {c['expect']!r}")

print(f"\nseen: {seen_ok}/{seen_n}   paraphrase: {par_ok}/{par_n}   total: {seen_ok + par_ok}/{seen_n + par_n}")
