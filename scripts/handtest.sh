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
# security/, database/), the ambiguous-piece guidance for
# the bare handler and database asks, the webserver + serve-an-html-file
# recipe, the security rows in a program file, line-number
# placement, the gorqlite dependency gate, running your own
# programs (quick ones, a live server with visits in its log,
# and the pin that "run the program X" never becomes the
# project's `make start`), the basic Linux commands (pwd, ls,
# cd, cat, echo, clear, date, whoami — gated, confined, with
# their refusal shapes), and the basics + cheatsheet
# statement rows.
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

# A Makefile with a start target, on purpose: the scratch
# project's own `make start` must never be what "run the
# program <file>.go" runs (John's 2026-11-01 screenshot: the
# chat ran the HOST project's make start and narrated a
# success that never happened). If this target ever executes
# during the run, its marker lands in a transcript and the
# run-vs-make case fails.
printf 'start:\n\t@echo SCRATCH-MAKE-START-RAN\n' > "$SCRATCH/Makefile"

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

# --- sessions: the ambiguous-piece guidance ---------------------------------
# The bare asks get guidance, never a write: "add handler
# Jim" names the two handler pieces and spells both exact
# sentences (with the target carried into both when one is
# given), and "add a database to jake" names the connection
# function and the inline piece instead of silently landing
# sql.Open in jake's main. These sessions run before
# folderdefaults on purpose: the nothing-written checks need
# handlers/ and database/ to not exist yet.

cat > "$SCRATCH/.inputs/handlerguide.txt" <<'EOF'
add handler Jim
add a handler named Jim to app/main.go
/quit
EOF
run_session handlerguide

case_transcript_contains handler-guide-two-kinds handlerguide '"add a form handler named Jim"' '"add a session handler named Jim"' "handlers/"
case_transcript_contains handler-guide-target-carried handlerguide '"add a form handler named Jim to app/main.go"' '"add a session handler named Jim to app/main.go"'
case_transcript_lacks handler-guide-no-gate handlerguide "Apply this edit?"
case_dir_absent handler-guide-nothing-written handlers

cat > "$SCRATCH/.inputs/dbguide1.txt" <<'EOF'
create a folder jake
y
create file jake/main.go
y
add a webserver to jake/main.go
y
/quit
EOF
run_session dbguide1

cp "$SCRATCH/jake/main.go" "$SCRATCH/.inputs/jake-main-before.txt"

cat > "$SCRATCH/.inputs/dbguide.txt" <<'EOF'
add a database to jake
/quit
EOF
run_session dbguide

case_transcript_contains db-guide-two-forms dbguide '"add a database connection function named Connect to jake"' '"add a database in main to jake/main.go"'
if cmp -s "$SCRATCH/.inputs/jake-main-before.txt" "$SCRATCH/jake/main.go"; then
	ok db-guide-file-unchanged
else
	bad db-guide-file-unchanged "the guidance changed jake/main.go"
fi
case_dir_absent db-guide-no-folder database

# --- session: a named connection resolves to the declaration ----------------
# John's setup from his 2026-11-01 screenshot: jake/db.go
# already holds a Connect function and no main, beside the
# webserver main the dbguide session built. His line —
# "add a database connection named Connet to jake/db.go" —
# used to match the inline database statement, drop the
# name, and fabricate a main to hold the snippet. A name
# belongs to a declaration: the ask now plans the
# connection FUNCTION for Connet in that file.
cat > "$SCRATCH/jake/db.go" <<'EOF'
package main

import (
	"context"
	"database/sql"
)

func Connect() (*sql.DB, error) {
	db, err := sql.Open("TODO: driver name", "TODO: connection string")
	if err != nil {
		return nil, err
	}
	if err := db.PingContext(context.Background()); err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(10)
	return db, nil
}
EOF

cat > "$SCRATCH/.inputs/namedconn.txt" <<'EOF'
add a database connection named Connet to jake/db.go
y
/quit
EOF
run_session namedconn

case_transcript_contains namedconn-plan namedconn "Add the database connect Connet to jake/db.go" "built the project and checked"
case_file_contains namedconn-decl jake/db.go "func Connet() (*sql.DB, error)" "func Connect() (*sql.DB, error)"
if grep -qF "func main(" "$SCRATCH/jake/db.go"; then
	bad namedconn-no-fake-main "jake/db.go grew a fabricated main"
else
	ok namedconn-no-fake-main
fi

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

# --- sessions: run your own programs ---------------------------------------
# quick/ is a lone-file program that prints one line and
# exits; broken/ is a lone file that does not compile. Both
# are single-file packages on purpose (the form rule: a lone
# file runs as itself). They are removed before the final
# build — broken/ would fail it by design.

mkdir -p "$SCRATCH/quick" "$SCRATCH/broken"
cat > "$SCRATCH/quick/main.go" <<'EOF'
package main

import "fmt"

func main() {
	fmt.Println("hello from the quick program")
}
EOF
cat > "$SCRATCH/broken/main.go" <<'EOF'
package main

func main() {
	println(nopeUndefined())
}
EOF

cat > "$SCRATCH/.inputs/runprog.txt" <<'EOF'
top
run the program quick/main.go
y
run the program broken/main.go
y
/quit
EOF
run_session runprog

case_transcript_contains run-top-empty runprog "No programs are running right now"
case_transcript_contains run-quick-program runprog "Run it? go run quick/main.go" "hello from the quick program" "It ran and finished (exit code 0)"
case_transcript_lacks run-not-make-start runprog "SCRATCH-MAKE-START-RAN"
case_transcript_contains run-compile-error runprog "It didn't run" "doesn't compile yet" "undefined: nopeUndefined"

# --- session: a live server, run in the background ---------------------------
# This session can't be a plain input file: the visit has to
# happen BETWEEN chat lines, while the server runs. A FIFO
# feeds the chat; the script answers the gate, waits for the
# server to answer a real HTTP fetch, then asks for the log
# (the visit must be in it) and stops the program.

RUNFIFO="$SCRATCH/.inputs/runserver.fifo"
mkfifo "$RUNFIFO"
RSOUT="$SCRATCH/transcripts/runserver.txt"
: > "$RSOUT"
(cd "$SCRATCH" && env -u GOEXPERIMENT timeout "$SESSION_TIMEOUT" "$BIN" -real-chat -domain unified) \
	< "$RUNFIFO" >> "$RSOUT" 2>&1 &
RSCHAT=$!
exec 7> "$RUNFIFO"
printf 'run the program webtest/main.go\ny\n' >&7
served=""
for _ in $(seq 1 90); do
	if curl --noproxy '*' -sf -o /dev/null "http://localhost:8080/"; then
		served=1
		break
	fi
	sleep 1
done
printf 'show visits\n' >&7
sleep 1
printf 'top\n' >&7
sleep 1
printf 'stop the program\n' >&7
sleep 1
printf 'top\n' >&7
sleep 1
printf '/quit\n' >&7
exec 7>&-
wait "$RSCHAT" 2>/dev/null || true
{
	printf '===== session: runserver =====\n'
	cat "$RSOUT"
	printf '\n'
} >> "$SCRATCH/transcript.txt"

if [ -n "$served" ]; then
	ok run-server-visit-served
else
	bad run-server-visit-served "the webtest server never answered http://localhost:8080/"
fi
case_transcript_contains run-server-background runserver "is running in the background" "http://localhost:8080"
if grep -qF "Opening http://localhost:8080 in your browser." "$RSOUT" || grep -qF "couldn't open your browser" "$RSOUT"; then
	ok run-server-browser-step
else
	bad run-server-browser-step "neither the browser announcement nor its honest failure is in the runserver transcript"
fi
case_transcript_contains run-server-visit-logged runserver "visit: GET /"
case_transcript_contains run-top-server runserver "Programs this chat is running" "webtest/main.go — running for" "1 visit"
case_transcript_contains run-server-stopped runserver "Stopped webtest/main.go"
case_transcript_contains run-top-after-stop runserver "It's no longer running."
if curl --noproxy '*' -sf -o /dev/null --max-time 2 "http://localhost:8080/"; then
	bad run-server-port-free "http://localhost:8080/ still answers after stop the program"
else
	ok run-server-port-free
fi

rm -rf -- "$SCRATCH/quick" "$SCRATCH/broken"

# --- session: basic Linux commands -------------------------------------------
# The terminal family: gated like every run, confined to the
# scratch. pwd prints the scratch root (the physical path —
# the chat resolves symlinks), ls/cat/echo/date/whoami do
# their one job, cd moves the working folder and pwd follows
# it, and the refusal shapes (rm, mkdir, a metachar line, a
# path escape, cd .. at the top) answer without any y —
# refusals are never gated, because nothing runs.

cat > "$SCRATCH/.inputs/linux.txt" <<'EOF'
pwd
y
ls
y
ls app
y
cat app/main.go
y
echo hello from handtest
y
date
y
whoami
y
clear
y
cd app
y
pwd
y
cd
y
cd ..
rm app/main.go
mkdir newfolder
ls app; cat app/main.go
ls ../outside
/quit
EOF
run_session linux

EXPECT_PWD="$(cd "$SCRATCH" && pwd -P)"
if grep -qFx -- "$EXPECT_PWD" "$SCRATCH/transcripts/linux.txt"; then
	ok linux-pwd-root
else
	bad linux-pwd-root "pwd did not print the scratch root $EXPECT_PWD"
fi
case_transcript_contains linux-ls-root linux "app/" "pieces.go" "go.mod"
if grep -qFx -- "main.go" "$SCRATCH/transcripts/linux.txt"; then
	ok linux-ls-folder
else
	bad linux-ls-folder "ls app did not list main.go on its own line"
fi
case_transcript_contains linux-cat-file linux 'fmt.Println("one")'
if grep -qFx -- "hello from handtest" "$SCRATCH/transcripts/linux.txt"; then
	ok linux-echo
else
	bad linux-echo "echo did not print its arguments"
fi
if grep -Eq -- '[A-Z][a-z]{2} [A-Z][a-z]{2} +[0-9]+ [0-9]{2}:[0-9]{2}:[0-9]{2}' "$SCRATCH/transcripts/linux.txt"; then
	ok linux-date
else
	bad linux-date "date did not print a standard date line"
fi
if grep -qFx -- "$(whoami)" "$SCRATCH/transcripts/linux.txt"; then
	ok linux-whoami
else
	bad linux-whoami "whoami did not print $(whoami)"
fi
case_transcript_contains linux-clear-gated linux "Run it? clear"
case_transcript_contains linux-cd-announce linux "Now working in app/"
if grep -qFx -- "$EXPECT_PWD/app" "$SCRATCH/transcripts/linux.txt"; then
	ok linux-cd-pwd
else
	bad linux-cd-pwd "pwd after cd app did not print $EXPECT_PWD/app"
fi
case_transcript_contains linux-cd-home linux "back at the folder the chat started in"
case_transcript_contains linux-cd-stop-at-root linux "cd .. stops here"
case_transcript_contains linux-rm-refused linux "doesn't delete files"
case_file_contains linux-rm-did-not-run app/main.go "package main"
case_transcript_contains linux-mkdir-refused linux "create a folder <name>"
case_dir_absent linux-mkdir-did-not-run newfolder
case_transcript_contains linux-meta-refused linux "no shell behind it"
case_transcript_contains linux-escape-refused linux "climbs out of the project folder"

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
