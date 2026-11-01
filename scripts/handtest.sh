#!/usr/bin/env bash
# handtest.sh — the scripted hand test for the gollemer chat.
#
#   make handtest
#
# Drives the REAL chat binary (built once from this checkout)
# through scripted sessions against a throwaway project and
# prints a per-case PASS/FAIL scoreboard. It covers the
# composer builds a person would otherwise re-type by hand
# after every pull: folder creation and bare files, the
# folder-default declarations (handlers/, middleware/,
# security/, database/), the webserver + serve-an-html-file
# recipe, the security rows in a program file, line-number
# placement, the gorqlite dependency gate, and the basics +
# cheatsheet statement rows.
#
# How it works:
#   - Everything happens under handtest/ at the repo root,
#     created fresh each run (the previous run is wiped). The
#     chat is started with handtest/ as its working folder, so
#     handtest/ is the project it edits; a symlink
#     handtest/data -> ../data lets it find this checkout's
#     models and book exactly as a normal start would. Nothing
#     outside handtest/ is written — the repo's own files,
#     jim/ included, are never touched.
#   - A session is a file of lines (asks and "y" gate answers)
#     piped to the chat; every gate consumes the next line.
#     Sessions run under `timeout`, so a stuck case fails
#     instead of hanging the run.
#   - Assertions are real: files and content fragments on
#     disk, transcript fragments for answer-only cases, and a
#     final `go build ./...` over the scratch project. Exit 0
#     only when every case passes.
#
# Needs: bash, go on PATH, coreutils timeout. Nothing else.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
SCRATCH="$ROOT/handtest"
MARKER=".handtest-owned"
BIN="$SCRATCH/.bin/gollemer"
SESSION_TIMEOUT="${HANDTEST_TIMEOUT:-300}"

PASS=0
FAIL=0

say() { printf '%s\n' "$*"; }
ok()  { PASS=$((PASS + 1)); printf '  PASS  %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  FAIL  %s  (%s)\n' "$1" "$2"; }
die() { say "handtest: $*" >&2; exit 1; }

# --- safety: the scratch is ours, and only ours -----------------

[ -f "$ROOT/go.mod" ] && [ -f "$ROOT/data/concepts.json" ] ||
	die "run me from the gollemer checkout (scripts/handtest.sh or make handtest)"
command -v go >/dev/null 2>&1 || die "go is not on PATH"
command -v timeout >/dev/null 2>&1 || die "timeout (coreutils) is not on PATH"

if [ -L "$SCRATCH" ]; then
	die "handtest/ is a symlink — I won't write through it. Remove it and run again."
fi
if [ -e "$SCRATCH" ] && [ ! -f "$SCRATCH/$MARKER" ]; then
	die "handtest/ already exists and isn't a handtest scratch — move it away and run again."
fi
rm -rf -- "$SCRATCH"
mkdir -p "$SCRATCH/.bin" "$SCRATCH/transcripts" "$SCRATCH/.inputs" "$SCRATCH/app"
touch "$SCRATCH/$MARKER"

# --- build the chat once, from this checkout --------------------

say "handtest: building the chat from this checkout…"
# The Makefile's env, for the chat build only. It must NOT leak
# into the sessions: the chat verifies every edit with its own
# `go build` inside the scratch module, whose go directive the
# local toolchain may handle without the toolchain switch the
# repo's go.mod triggers — an experiment the switched toolchain
# knows (simd) can be unknown to the unswitched one, and every
# edit would fail its build check. Sessions therefore run with
# GOEXPERIMENT stripped (see run_session and the final build).
(cd "$ROOT" && GOEXPERIMENT="${GOEXPERIMENT:-simd}" CGO_ENABLED="${CGO_ENABLED:-1}" go build -o "$BIN" .) || die "go build failed"

# --- fixtures ----------------------------------------------------
# handtest/ is its own module and its own project. The data
# symlink is how the chat finds the checkout's models/book
# while its edits land in the scratch.

ln -s ../data "$SCRATCH/data"

cat > "$SCRATCH/go.mod" <<'EOF'
module handtest

go 1.21
EOF

# The program file: line numbers are load-bearing for the
# placement case — line 9 is fmt.Println("one"), line 10 is
# fmt.Println("two"). Login is the handler the idor case aims at.
cat > "$SCRATCH/app/main.go" <<'EOF'
package main

import (
	"fmt"
	"net/http"
)

func main() {
	fmt.Println("one")
	fmt.Println("two")
}

func Login(w http.ResponseWriter, r *http.Request) {
}
EOF

# The pieces file: one stub function per row group, so the
# statement rows' sample variable names (ages, names, text…)
# never collide inside one function — the same reason two
# rqlite statements can't share a function.
cat > "$SCRATCH/pieces.go" <<'EOF'
package main

func main() {
}

func alpha() {
}

func beta() {
}

func gamma() {
}

func delta() {
}

func epsilon() {
}

func zeta() {
}

func eta() {
}
EOF

# --- session driver ------------------------------------------------

# run_session <name> — reads the session's lines from
# .inputs/<name>.txt, runs the chat with the scratch as its
# working folder, and appends the output to the session's
# transcript and the combined handtest/transcript.txt.
run_session() {
	local name="$1"
	local out="$SCRATCH/transcripts/$name.txt"
	: > "$out"
	(cd "$SCRATCH" && env -u GOEXPERIMENT timeout "$SESSION_TIMEOUT" "$BIN" -real-chat -domain unified) \
		< "$SCRATCH/.inputs/$name.txt" >> "$out" 2>&1
	local rc=$?
	{
		printf '===== session: %s =====\n' "$name"
		cat "$out"
		printf '\n'
	} >> "$SCRATCH/transcript.txt"
	if [ "$rc" -eq 124 ]; then
		bad "session-$name" "timed out after ${SESSION_TIMEOUT}s"
	fi
}

# --- assertions ------------------------------------------------------

case_file_contains() { # <case> <path-under-scratch> <fragment>…
	local id="$1" rel="$2"
	shift 2
	local f
	for f in "$@"; do
		if [ ! -f "$SCRATCH/$rel" ] || ! grep -qF -- "$f" "$SCRATCH/$rel"; then
			bad "$id" "missing '$f' in handtest/$rel"
			return
		fi
	done
	ok "$id"
}

case_transcript_contains() { # <case> <session> <fragment>…
	local id="$1" sess="$2"
	shift 2
	local f
	for f in "$@"; do
		if ! grep -qF -- "$f" "$SCRATCH/transcripts/$sess.txt"; then
			bad "$id" "missing '$f' in the $sess transcript"
			return
		fi
	done
	ok "$id"
}

case_transcript_lacks() { # <case> <session> <fragment>
	local id="$1" sess="$2" f="$3"
	if grep -qF -- "$f" "$SCRATCH/transcripts/$sess.txt"; then
		bad "$id" "unexpected '$f' in the $sess transcript"
		return
	fi
	ok "$id"
}

case_dir_absent() { # <case> <path-under-scratch>
	local id="$1" rel="$2"
	if [ -e "$SCRATCH/$rel" ]; then
		bad "$id" "handtest/$rel exists but shouldn't (yet)"
		return
	fi
	ok "$id"
}

line_no() { grep -nF -- "$2" "$1" | head -1 | cut -d: -f1; }

case_between() { # <case> <path-under-scratch> <first> <middle> <last>
	local id="$1" rel="$2" a="$3" b="$4" c="$5"
	local path="$SCRATCH/$rel"
	local na nb nc
	na="$(line_no "$path" "$a")"
	nb="$(line_no "$path" "$b")"
	nc="$(line_no "$path" "$c")"
	if [ -n "$na" ] && [ -n "$nb" ] && [ -n "$nc" ] && [ "$na" -lt "$nb" ] && [ "$nb" -lt "$nc" ]; then
		ok "$id"
	else
		bad "$id" "'$b' did not land between '$a' and '$c' in handtest/$rel"
	fi
}

say "handtest: running scripted sessions (transcripts in handtest/transcripts/)…"

# --- session: create (folders and bare files) ----------------------

cat > "$SCRATCH/.inputs/create.txt" <<'EOF'
create a folder jane
y
create jane/main.go
y
create notes.go in folder jane
y
/quit
EOF
run_session create

case_file_contains create-folder-jane jane/main.go "package main"
case_file_contains bare-file-path jane/main.go "package main" "func main()"
case_file_contains bare-file-in-folder jane/notes.go "package main"

# --- session: webserver + serve an html file ------------------------

cat > "$SCRATCH/.inputs/webserver.txt" <<'EOF'
create a folder webtest
y
create a file webtest/main.go with a webserver
y
serve an html file to webtest/main.go
y
serve an html file to webtest/main.go
/quit
EOF
run_session webserver

case_file_contains webserver-file webtest/main.go "ListenAndServe"
case_file_contains serve-html webtest/main.go "http.ServeFile"
case_file_contains serve-html-page webtest/index.html "TODO"
case_transcript_contains serve-html-repeat webserver "already serves" "nothing to add"

# --- session: line placement (pristine program file) -----------------

cat > "$SCRATCH/.inputs/lines.txt" <<'EOF'
add a print line in main to app/main.go after line 9
y
/quit
EOF
run_session lines

case_between line-placement app/main.go 'fmt.Println("one")' 'fmt.Println("")' 'fmt.Println("two")'
case_transcript_contains line-anchor-quoted lines "after line 9:" 'line 9: `fmt.Println("one")`'

# --- session: security rows in the program file -----------------------

cat > "$SCRATCH/.inputs/security.txt" <<'EOF'
add a jwt token in main to app/main.go
y
add an idor check in Login to app/main.go
y
/quit
EOF
run_session security

case_file_contains jwt-in-main app/main.go "JWT_SECRET"
case_file_contains idor-in-handler app/main.go "StatusForbidden"

# --- session: a named file wins over the folder default ----------------

cat > "$SCRATCH/.inputs/namedwins.txt" <<'EOF'
add a cors middleware named Cors to app/main.go
y
/quit
EOF
run_session namedwins

case_file_contains named-file-wins app/main.go "Access-Control-Allow-Origin"
case_dir_absent named-file-wins-no-folder security

# --- session: folder-default declarations --------------------------------

cat > "$SCRATCH/.inputs/folderdefaults.txt" <<'EOF'
add a form handler named SignUp
y
add a middleware named Logger
y
add a cors middleware named Cors
y
add a database connection function named Connect
y
/quit
EOF
run_session folderdefaults

case_file_contains folder-handlers handlers/signup.go "package handlers" "func SignUp("
case_file_contains folder-middleware middleware/logger.go "package middleware" "func Logger("
case_file_contains folder-security security/cors.go "package security" "Access-Control-Allow-Origin"
case_file_contains folder-database database/connect.go "package database" "func Connect() (*sql.DB, error)" "sql.Open"

# --- session: the gorqlite dependency gate --------------------------------
# The scratch go.mod does not list gorqlite, so the ask must be
# answered with the go get line and NO plan, and "installed"
# must be answered with the not-in-go.mod-yet check. The
# positive path (dependency actually installed -> the original
# plan re-offered behind the gate) needs a real `go get`, so
# it stays pinned by the Go test suite instead:
# TestGorqliteGateAbsentThenInstalled in the chat package.

cat > "$SCRATCH/.inputs/gorqlitegate.txt" <<'EOF'
add a gorqlite connection named Connect
installed
/quit
EOF
run_session gorqlitegate

case_transcript_contains gorqlite-gate-line gorqlitegate "go get github.com/rqlite/gorqlite"
case_transcript_lacks gorqlite-no-plan gorqlitegate "Apply this edit?"
case_transcript_contains gorqlite-installed-not-yet gorqlitegate "isn't in its require list yet"
if grep -rqF --include='*.go' "gorqlite" "$SCRATCH/app" "$SCRATCH/handlers" "$SCRATCH/middleware" "$SCRATCH/security" "$SCRATCH/database" "$SCRATCH/webtest" "$SCRATCH/jane" "$SCRATCH/pieces.go" 2>/dev/null; then
	bad gorqlite-nothing-written "a scratch .go file mentions gorqlite"
else
	ok gorqlite-nothing-written
fi

# --- sessions: basics + cheatsheet statement rows -------------------------
# One stub function per row group (see pieces.go): the rows'
# sample names never collide inside a function.

cat > "$SCRATCH/.inputs/pieces1.txt" <<'EOF'
add a string builder in main to pieces.go
y
add an env var in main to pieces.go
y
add a while loop in alpha to pieces.go
y
add an infinite loop in alpha to pieces.go
y
add a sleep in alpha to pieces.go
y
add a type conversion in alpha to pieces.go
y
add swap values in alpha to pieces.go
y
add a delete from a map in beta to pieces.go
y
add a sum of a slice in beta to pieces.go
y
add a join strings in beta to pieces.go
y
/quit
EOF
run_session pieces1

cat > "$SCRATCH/.inputs/pieces2.txt" <<'EOF'
add a clear map in gamma to pieces.go
y
add a repeat string in gamma to pieces.go
y
add read input in gamma to pieces.go
y
add the keys of a map in delta to pieces.go
y
add a string contains in delta to pieces.go
y
add a hex encoding in delta to pieces.go
y
add an insert into a slice in epsilon to pieces.go
y
add a file exists check in epsilon to pieces.go
y
add a print to stderr in epsilon to pieces.go
y
/quit
EOF
run_session pieces2

cat > "$SCRATCH/.inputs/pieces3.txt" <<'EOF'
add a remove a value from a slice in zeta to pieces.go
y
add a delete file in zeta to pieces.go
y
add a rename file in zeta to pieces.go
y
add a reverse slice in eta to pieces.go
y
/quit
EOF
run_session pieces3

# basics sweep
case_file_contains basics-string-builder pieces.go "strings.Builder"
case_file_contains basics-env-var pieces.go "os.Getenv"
# cheatsheet sweep — the 21 rows
case_file_contains row-while-loop pieces.go "for count < 3"
case_file_contains row-infinite-loop pieces.go "a loop with no condition"
case_file_contains row-type-conversion pieces.go "float64(whole)"
case_file_contains row-sleep pieces.go "time.Sleep"
case_file_contains row-delete-from-map pieces.go "delete(ages,"
case_file_contains row-clear-map pieces.go "clear(ages)"
case_file_contains row-map-keys pieces.go "keys = append(keys, key)"
case_file_contains row-read-input pieces.go "bufio.NewScanner(os.Stdin)"
case_file_contains row-swap-values pieces.go "first, second = second, first"
case_file_contains row-sum pieces.go "total += number"
case_file_contains row-join-strings pieces.go "strings.Join"
case_file_contains row-repeat-string pieces.go "strings.Repeat"
case_file_contains row-string-contains pieces.go "strings.Contains"
case_file_contains row-file-exists pieces.go "os.Stat"
case_file_contains row-delete-file pieces.go "os.Remove"
case_file_contains row-rename-file pieces.go "os.Rename"
case_file_contains row-hex-encoding pieces.go "hex.EncodeToString"
case_file_contains row-insert-into-slice pieces.go 'names[1] = "bo"'
case_file_contains row-remove-value-from-slice pieces.go "kept = append(kept, name)"
case_file_contains row-reverse-slice pieces.go "names[i], names[j] = names[j], names[i]"
case_file_contains row-print-stderr pieces.go "fmt.Fprintln(os.Stderr"

# --- final build over the whole scratch project ---------------------------

if (cd "$SCRATCH" && env -u GOEXPERIMENT timeout "$SESSION_TIMEOUT" go build ./...) > "$SCRATCH/transcripts/final-build.txt" 2>&1; then
	ok final-build
else
	bad final-build "go build ./... failed in handtest/ (see transcripts/final-build.txt)"
fi

# --- scoreboard -------------------------------------------------------------

TOTAL=$((PASS + FAIL))
say ""
say "handtest scoreboard: $PASS passed, $FAIL failed, $TOTAL total"
say "scratch project and transcripts are in handtest/ — read them, then just run it again; it rebuilds its scratch every run."
if [ "$FAIL" -ne 0 ]; then
	exit 1
fi
exit 0
