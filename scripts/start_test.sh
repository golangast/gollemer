#!/bin/bash
# Tests for the make start / make guide split (scripts/start.sh):
#
#   bash scripts/start.sh        make start — the chat, TTY only
#   bash scripts/start.sh guide  make guide — the guide, on demand
#
# Run: bash scripts/start_test.sh   (the PTY case needs python3)
set -u
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fail=0
ok()   { echo "ok   - $1"; }
bad()  { echo "FAIL - $1"; fail=1; }

# 1. Piped guide == the golden, byte for byte, with no ANSI in it.
bash scripts/start.sh guide </dev/null > "$TMP/guide.out" 2>&1
if cmp -s "$TMP/guide.out" scripts/start_guide.golden.txt; then
  ok "piped guide matches scripts/start_guide.golden.txt verbatim"
else
  bad "piped guide differs from the golden (diff below)"
  diff scripts/start_guide.golden.txt "$TMP/guide.out" | head -20
fi
if grep -q $'\033' "$TMP/guide.out"; then
  bad "piped guide contains ANSI escapes"
else
  ok "piped guide is escape-free"
fi

# 2. Piped start: exactly one short line, no guide, no chat.
bash scripts/start.sh </dev/null > "$TMP/start.out" 2>&1
printf '%s\n' "make start opens the chat and needs a real terminal — for the guide, run: make guide" > "$TMP/start.want"
if cmp -s "$TMP/start.out" "$TMP/start.want"; then
  ok "piped start prints only the terminal-needed line"
else
  bad "piped start printed something else:"
  cat "$TMP/start.out"
fi

# 3. The guide's brain count is the registry's brain count.
reg=$(awk '/^var brainRegistry/,/^}/' internal/ai/training/chat/brains.go | grep -cP '^\t\{"')
said=$(grep -oE 'One chat, [0-9]+ brains' "$TMP/guide.out" | grep -oE '[0-9]+')
if [ -n "$reg" ] && [ "$reg" = "$said" ]; then
  ok "guide brain count ($said) matches brainRegistry ($reg)"
else
  bad "guide says $said brains, brainRegistry has $reg"
fi

# 4. Every make target the guide names exists in the Makefile,
#    and make guide is wired to this script.
missing=0
# ("make commands" in the try-lines is prose — the makefile brain's
# description, "makefile: make commands" — not a target reference.)
for t in $(grep -oE 'make [a-z][a-z-]*' "$TMP/guide.out" | awk '{print $2}' | sort -u | grep -vx commands); do
  if ! grep -qE "^$t:" Makefile; then
    echo "     guide names missing target: make $t"
    missing=1
  fi
done
if [ "$missing" = 0 ]; then ok "every make target named in the guide exists"; else bad "guide names a missing make target"; fi
if awk '/^guide:/,/^$/' Makefile | grep -q 'start.sh guide'; then
  ok "make guide runs scripts/start.sh guide"
else
  bad "make guide is not wired to scripts/start.sh guide"
fi

# 5. PTY: make start launches the chat directly — start screen with
#    NO guide text before it, a answered line, a clean exit.
if command -v python3 >/dev/null 2>&1; then
  cat > "$TMP/drive.py" << 'PYEOF'
import os, pty, select, struct, subprocess, sys, termios, time, fcntl
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
p = subprocess.Popen(["bash", "scripts/start.sh"], stdin=slave, stdout=slave,
                     stderr=slave, close_fds=True)
os.close(slave)
os.set_blocking(master, False)
buf = b""
def pump(seconds):
    global buf
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([master], [], [], 0.2)
        if master in r:
            try:
                d = os.read(master, 65536)
            except OSError:
                d = b""
            if d:
                buf += d
        if p.poll() is not None:
            return
def wait_for(needle, seconds):
    end = time.time() + seconds
    while time.time() < end:
        if needle.encode() in buf:
            return True
        pump(0.3)
    return needle.encode() in buf
ok = True
if not wait_for("brains ready", 300):
    print("chat start screen never appeared"); ok = False
if b"START HERE" in buf:
    print("guide text preceded the chat"); ok = False
os.write(master, b"brains\r")
if not wait_for("THE CAST", 60):
    print("chat did not answer 'brains'"); ok = False
os.write(master, b"exit\r")
end = time.time() + 60
while p.poll() is None and time.time() < end:
    pump(0.3)
if p.poll() is None:
    print("chat did not exit"); p.kill(); ok = False
elif p.returncode != 0:
    print("chat exit status", p.returncode); ok = False
sys.exit(0 if ok else 1)
PYEOF
  if python3 "$TMP/drive.py"; then
    ok "PTY start: chat screen first (no guide), answers, clean exit 0"
  else
    bad "PTY start test failed"
  fi
else
  echo "skip - PTY start test (no python3)"
fi

if [ "$fail" = 0 ]; then echo "ALL PASS"; else echo "FAILURES ABOVE"; fi
exit "$fail"
