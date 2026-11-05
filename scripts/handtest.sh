#!/usr/bin/env bash
# handtest.sh — the scripted hand test for the gollemer chat.
#
#   make handtest
#
# Drives the REAL chat binary (built once from this checkout)
# through scripted sessions against a throwaway project and
# prints a per-case PASS/FAIL scoreboard. It covers the
# composer builds a person would otherwise re-type by hand
# after every pull: folder creation and bare files (including
# the "add the file" spelling and a statement aimed at a
# function-less file), the
# folder-default declarations (handlers/, middleware/,
# security/, database/), declarations creating a named file
# that doesn't exist yet (and the missing-folder refusal),
# the ambiguous-piece guidance for
# the bare handler and database asks (including John's jim/db journey: the
# guidance's own sentences followed to their landings in a folder with no
# Go files), the webserver + serve-an-html-file
# recipe, the handler-wiring recipe (a placed handler
# registered with the webserver — named, bare, and the two
# already-registered answers), the page family (six named
# page kinds as page+handler pairs in one program — the
# homepage taking over the webserver's starter reply — plus
# the no-server follow-up journey and the never-overwrite
# refusal), the security rows in a program file, line-number
# placement, the gorqlite dependency gate, running your own
# programs (quick ones, a live server with visits in its log,
# and the pin that "run the program X" never becomes the
# project's `make start`), the basic Linux commands (pwd, ls,
# cd, cat, echo, clear, date, whoami — gated, confined, with
# their refusal shapes), the basics + cheatsheet
# statement rows, the one-ask make-and-call (a new
# function in one file, called from another, one gate), and
# the session working context (John's database-package
# journey: bare follow-ups resolved against the work in
# progress, each reading printed with its full path), and
# the clarifying questions (an ask missing one piece of
# information asks once; a name, number, or word answer
# completes the original ask).
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

case_file_lacks() { # <case> <path-under-scratch> <fragment>
	local id="$1" rel="$2" f="$3"
	if [ -f "$SCRATCH/$rel" ] && grep -qF -- "$f" "$SCRATCH/$rel"; then
		bad "$id" "unexpected '$f' in handtest/$rel"
		return
	fi
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

# --- session: "add the file" creates the file itself (D1) ------------
# The add-verb spelling of a bare file: store/ already holds
# a package-store file, and "add the file store/db.go" lands
# db.go beside it with that package — behind the create plan
# and its gate, not the write boundary's refusal. The repeat
# ask meets the exists refusal and the file is byte-identical.

cat > "$SCRATCH/.inputs/addfile.txt" <<'EOF'
create a folder store
y
create a file store.go in folder store
y
add the file store/db.go
y
add the file store/db.go
/quit
EOF
run_session addfile

case_file_contains add-file-lands store/db.go "package store"
case_transcript_contains add-file-already-there addfile "store/db.go is already there — I never overwrite what's yours"
printf 'package store\n' > "$SCRATCH/.inputs/store-db-expected.txt"
if cmp -s "$SCRATCH/.inputs/store-db-expected.txt" "$SCRATCH/store/db.go"; then
	ok add-file-repeat-unchanged
else
	bad add-file-repeat-unchanged "store/db.go is not exactly the file the first ask created"
fi

# --- session: webserver + serve an html file ------------------------

cat > "$SCRATCH/.inputs/webserver.txt" <<'EOF'
create a folder webtest
y
create a file webtest/main.go with a webserver
y
serve home.html to webtest/main.go
y
serve home.html to webtest/main.go
serve an html file to webtest/main.go
/quit
EOF
run_session webserver

case_file_contains webserver-file webtest/main.go "ListenAndServe"
case_file_contains serve-html webtest/main.go "http.ServeFile"
case_transcript_contains serve-html-repeat webserver "already serves" "nothing to add"
case_file_contains serve-html-named-page webtest/home.html "TODO"
case_file_contains serve-html-rooted-path webtest/main.go 'http.ServeFile(w, r, "webtest/home.html")'
# The bare "serve an html file" ask names the default index
# page, and an index page can only be served at its folder's
# address — which the webserver's own "/" handler already
# owns. The honest answer is the route conflict, and no
# index page is created (index serving itself — a standalone
# program answering at "/" — is fetched over HTTP by the
# analyze suite's serve tests).
case_transcript_contains serve-html-index-conflict webserver "already answers http://localhost:8080/ with a different handler"
case_dir_absent serve-html-index-not-created webtest/index.html

# --- session: wire a handler to the webserver -----------------------
# John's 2026-11-03 journey, end to end: the form handler
# Jake lands in jim/main.go (which already runs a webserver),
# and the sentence that used to fall through every claim —
# "connect the handler to the webserver" — registers it. The
# named form lands Jake first; the repeats pin the honest
# answers (already registered at the asked path; already
# registered at ANOTHER path when asked at /signup); then a
# second handler lands and the BARE form resolves to it —
# the fixture's Login is not a candidate, living as it does
# in app/, a program of its own that jim's server cannot
# reach into.

cat > "$SCRATCH/.inputs/wire.txt" <<'EOF'
create a folder jim
y
create a file jim/main.go with a webserver
y
add a form handler named Jake to jim/main.go
y
connect the Jake handler to the webserver
y
connect the Jake handler to the webserver
connect the Jake handler to the webserver at /signup
add a form handler named Pete to jim/main.go
y
connect the handler to the webserver
y
/quit
EOF
run_session wire

case_file_contains wire-jake-registered jim/main.go 'http.HandleFunc("/jake", Jake)'
case_between wire-jake-before-serve jim/main.go 'func main() {' 'http.HandleFunc("/jake", Jake)' 'http.ListenAndServe'
case_transcript_contains wire-duplicate-answer wire "Jake is already registered at /jake" "nothing to add"
case_transcript_contains wire-other-path-refusal wire "already registered at /jake" "/signup" "won't register it at /signup as well"
case_file_contains wire-bare-pete jim/main.go 'http.HandleFunc("/pete", Pete)'

# --- session: clarifying questions (2026-11-05) ----------------------
# An ask missing one piece of information asks ONE question
# instead of ending in guidance prose; the next line — a
# name, a number, a distinguishing word — completes the
# ORIGINAL ask behind the usual gate. Two handlers wait
# unregistered, so the bare wiring ask asks which one; the
# number "1" completes it for Zeke. A database ask is then
# interrupted by a full different ask (which lands, and the
# question is gone — no completion happens behind it), and
# finally answered by the one word "connection"; an unnamed
# struct is completed by its one-word name answer.

cat > "$SCRATCH/.inputs/clarify.txt" <<'EOF'
create a folder quest
y
create a file quest/main.go with a webserver
y
add a form handler named Zeke to quest/main.go
y
add a form handler named Yara to quest/main.go
y
connect the handler to the webserver
1
y
add a database to quest/main.go
add a struct named Profile to quest/main.go
y
add a database to quest/main.go
connection
y
add a struct to quest/main.go
Widget
y
/quit
EOF
run_session clarify

case_transcript_contains clarify-wire-question clarify "More than one handler isn't registered yet" "1. Zeke" "2. Yara" "Answer with the name or the number"
case_file_contains clarify-number-answer quest/main.go 'http.HandleFunc("/zeke", Zeke)'
case_file_lacks clarify-question-wrote-nothing quest/main.go 'http.HandleFunc("/yara", Yara)'
case_transcript_contains clarify-database-question clarify '"database" names two different pieces here' 'Answer "connection" or "inline"'
case_file_contains clarify-fallthrough-landed quest/main.go "type Profile struct"
case_file_contains clarify-word-answer quest/main.go "func Connect() (*sql.DB, error)"
case_file_contains clarify-name-answer quest/main.go "type Widget struct"

# --- session: the phrasing-gap widenings (2026-11-05) ---------------
# The beginner's own spellings of asks the features already
# owned: "new" in the creation slot, a server that RETURNS a
# page (John's live miss — the serve grammar knew only
# serves/serving/to serve), the split phrasal "wire Gus up",
# and the boundary answering copy like its sibling verbs.

cat > "$SCRATCH/.inputs/gaps.txt" <<'EOF'
create a new folder gaptest
y
create a new file in folder gaptest with a webserver
y
build a server that returns home.html to gaptest/main.go
y
add a form handler named Gus to gaptest/main.go
y
wire Gus up to the webserver
y
copy the file gaptest/home.html to gaptest/welcome.html
/quit
EOF
run_session gaps

case_file_contains gaps-returns-serves gaptest/main.go "http.ServeFile"
case_file_contains gaps-returns-page gaptest/home.html "TODO"
case_file_contains gaps-wire-split-phrasal gaptest/main.go 'http.HandleFunc("/gus", Gus)'
case_transcript_contains gaps-copy-boundary gaps "I can't copy files or folders on disk." "cp gaptest/home.html gaptest/welcome.html"

# --- sessions: the page family ---------------------------------------
# A named page kind is a matched pair in ONE gated plan: the
# starter page, the handler that serves it, and the route
# registration composed onto the program's webserver. The
# homepage takes over the webserver component's own starter
# reply at "/" (that line is replaced, nothing else in the
# server changes); the contact page's form asks for exactly
# the fields its handler reads. The repeat about ask meets
# the never-overwrite refusal — an existing page is never
# rewritten.

# --- session: a page with no server — the follow-up journey ----------
# The lonely program has no webserver: the about page and its
# handler still land, the plan names the finishing sentence,
# and feeding that sentence back (after adding the webserver
# it points at) registers the handler. Placement first,
# wiring as its own gated step. The handler is named Solo
# because the scratch holds several programs by now, and two
# of them can each have an About of their own — the wiring
# sentence resolves a name, so this journey gives it a name
# only one program has.
# ORDER MATTERS: this journey runs BEFORE the site journey
# below. The wiring claim reads route conflicts project-wide
# (its shipped rule: one path, one handler, per project), so
# /about must still be unregistered anywhere when Solo wires
# up. The site journey then proves the page family's own
# composition is per-program: site composes its own About at
# /about right after, in its own binary, without a murmur.

cat > "$SCRATCH/.inputs/pagesnoserver.txt" <<'EOF'
create a folder lonely
y
create a file lonely/main.go
y
create an about page named Solo in lonely
y
add a webserver to lonely/main.go
y
connect the Solo handler to the webserver at /about
y
/quit
EOF
run_session pagesnoserver

case_file_contains pages-noserver-page lonely/about.html "About us"
case_transcript_contains pages-noserver-followup pagesnoserver 'connect the Solo handler to the webserver at /about'
case_file_contains pages-noserver-wired lonely/main.go 'http.HandleFunc("/about", Solo)'

cat > "$SCRATCH/.inputs/pages.txt" <<'EOF'
create a folder site
y
create a file site/main.go with a webserver
y
create a homepage in site
y
create an about page in site
y
create a contact page in site
y
/quit
EOF
run_session pages

cat > "$SCRATCH/.inputs/pages2.txt" <<'EOF'
create an item list page in site
y
create a details page in site
y
create a form page in site
y
create an about page in site
/quit
EOF
run_session pages2

case_file_contains pages-home-page site/index.html "TODO: your homepage heading"
case_file_contains pages-home-wired site/main.go "func Home(" 'http.HandleFunc("/", Home)'
if grep -qF "hello — your webserver is running" "$SCRATCH/site/main.go"; then
	bad pages-starter-replaced "the webserver's starter line is still in site/main.go"
else
	ok pages-starter-replaced
fi
case_between pages-about-before-serve site/main.go "func main() {" 'http.HandleFunc("/about", About)' "http.ListenAndServe"
case_file_contains pages-contact-page site/contact.html 'name="email"' 'name="subject"' 'name="message"' 'action="/contact"'
case_file_contains pages-contact-handler site/main.go "func Contact(" 'r.FormValue("email")' 'http.ServeFile(w, r, "site/contact.html")'
case_file_contains pages-list-page site/items.html "TODO: an item"
case_file_contains pages-list-wired site/main.go "func Items(" 'http.HandleFunc("/items", Items)'
case_file_contains pages-details-page site/details.html "TODO: the item's name"
case_file_contains pages-details-wired site/main.go "func Details(" 'http.HandleFunc("/details", Details)'
case_file_contains pages-form-page site/form.html 'name="email"' 'action="/form"'
case_file_contains pages-form-handler site/main.go "func Form(" 'http.HandleFunc("/form", Form)'
case_transcript_contains pages-repeat-refused pages2 "about.html is already there — I never overwrite what's yours"


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

# --- session: a statement aimed at a function-less file (D2) ------------
# ledger/ledger.go holds only its package clause — no
# functions at all. "add database connection to
# ledger/ledger.go" can't name a host function (there is
# none to name), so the answer spells the one piece that
# fits a file like this — the database connection function,
# a function of its own — and the file stays byte-identical.

mkdir -p "$SCRATCH/ledger"
cat > "$SCRATCH/ledger/ledger.go" <<'EOF'
package ledger
EOF
cp "$SCRATCH/ledger/ledger.go" "$SCRATCH/.inputs/ledger-before.txt"

cat > "$SCRATCH/.inputs/nohost.txt" <<'EOF'
add database connection to ledger/ledger.go
/quit
EOF
run_session nohost

case_transcript_contains nohost-guidance nohost "ledger/ledger.go declares no functions, so there's nowhere to put the database — that piece lives inside a function." 'The piece that fits a file like this is the database connection function, a function of its own: say "add a database connection function named Connect to ledger/ledger.go"'
case_transcript_lacks nohost-no-gate nohost "Apply this edit?"
if cmp -s "$SCRATCH/.inputs/ledger-before.txt" "$SCRATCH/ledger/ledger.go"; then
	ok nohost-file-unchanged
else
	bad nohost-file-unchanged "the guidance changed ledger/ledger.go"
fi

# --- session: declarations create their named file -------------------------
# A declaration that names a file creates it when the file
# doesn't exist yet (its folder must): the struct lands in a
# new app/profile.go and the database connection function in
# a new app/store.go, both package main read from app/'s own
# files. A missing FOLDER refuses with the real first step —
# "create a folder nowhere" — and no gate.

cat > "$SCRATCH/.inputs/declcreate.txt" <<'EOF'
add a struct named Profile to nowhere/profile.go
add a struct named Profile to app/profile.go
y
add a database connection function named Store to app/store.go
y
/quit
EOF
run_session declcreate

case_transcript_contains declcreate-missing-folder declcreate "there's no folder nowhere in this project" 'say "create a folder nowhere" first'
case_transcript_lacks declcreate-no-webserver-remedy declcreate "with a webserver"
case_dir_absent declcreate-no-nowhere nowhere
case_transcript_contains declcreate-plan-creates declcreate "Create app/profile.go"
case_file_contains declcreate-struct app/profile.go "package main" "type Profile struct"
case_file_contains declcreate-connect app/store.go "package main" "func Store() (*sql.DB, error)"

# --- sessions: the database folder guidance (John's jim/db journey) --------
# John's live state (2026-11-03): jim/db holds only an empty
# database.db — no Go file at all. The bare ask's guidance
# suggests two sentences; in this state BOTH used to die in
# the webserver claim's refusals ("has no main program to
# add a server to… create a file jim/db/main.go with a
# webserver"). The declaration sentence now lands Connect as
# jim/db/connect.go (package db), and the statement sentence
# answers with its own two-step — create the file, ask
# again — which the second session follows to its landing.

mkdir -p "$SCRATCH/jim/db"
: > "$SCRATCH/jim/db/database.db"

cat > "$SCRATCH/.inputs/dbfolder1.txt" <<'EOF'
add database to jim/db
add a database connection function named Connect to jim/db
y
/quit
EOF
run_session dbfolder1

case_transcript_contains dbfolder-guide dbfolder1 '"add a database connection function named Connect to jim/db"' '"add a database in main to jim/db/main.go"'
case_transcript_lacks dbfolder-no-server-words dbfolder1 "with a webserver"
case_file_contains dbfolder-connect-lands jim/db/connect.go "package db" "func Connect() (*sql.DB, error)"

mkdir -p "$SCRATCH/jim/db2"
: > "$SCRATCH/jim/db2/database.db"

cat > "$SCRATCH/.inputs/dbfolder2.txt" <<'EOF'
add a database in main to jim/db2/main.go
create a file jim/db2/main.go
y
add a database in main to jim/db2/main.go
y
/quit
EOF
run_session dbfolder2

case_transcript_contains dbfolder2-two-step dbfolder2 "there's no jim/db2/main.go in this project — a database lives inside a function in a file that exists" 'Say "create a file jim/db2/main.go" first'
case_transcript_lacks dbfolder2-no-server-words dbfolder2 "with a webserver"
case_file_contains dbfolder2-lands jim/db2/main.go "sql.Open"

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
case_transcript_contains gorqlite-gate-grammar gorqlitegate "That ask — a gorqlite connection named Connect (it would be created as database/connect.go) — needs the gorqlite client"
case_transcript_lacks gorqlite-gate-no-broken-frame gorqlitegate "That a gorqlite"
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
# The serve-html contract, fetched: the page the recipe wired
# into webtest must come back over HTTP from the program the
# chat's runner started at the project root — a bare relative
# ServeFile path 404s here, which is how the defect showed.
# The fetch sits after the first 'top' on purpose: that check
# counts the visits, and this fetch is one more.
PAGEOUT="$SCRATCH/.inputs/pagefetch.out"
pagecode=""
if [ -n "$served" ]; then
	pagecode="$(curl --noproxy '*' -s -o "$PAGEOUT" -w '%{http_code}' "http://localhost:8080/home.html")"
fi
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
if [ "$pagecode" = "200" ] && grep -qF "Your Go program is serving this file" "$PAGEOUT"; then
	ok serve-html-page-fetched
else
	bad serve-html-page-fetched "GET /home.html through the chat's runner = '$pagecode' (want 200 and the starter page's words)"
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

# --- session: one ask, two files (make-and-call) -----------------------------
# The one-ask recipe (2026-11-02): a new function declared in
# one file and called from another, in ONE plan behind ONE
# gate. The named-caller case creates the caller's file first
# (the calling side goes into a file that exists — the uses
# route's standing rule), then one line lands both halves;
# the bare case hosts the call in app/main.go's main.

cat > "$SCRATCH/.inputs/makecallnamed.txt" <<'EOF'
create app/report.go
y
make a function named Validate in app/validate.go and add a function named Report in app/report.go that calls it
y
/quit
EOF
run_session makecallnamed

case_transcript_contains makecall-named-plan makecallnamed "Make Validate in app/validate.go and call it in app/report.go" "One plan covers both files"
case_file_contains makecall-named-callee app/validate.go "func Validate() {" "// TODO: write the body"
case_file_contains makecall-named-caller app/report.go "func Report() {" "Validate()"

cat > "$SCRATCH/.inputs/makecallbare.txt" <<'EOF'
make a function named Greet in app/greet.go and call it in app/main.go
y
/quit
EOF
run_session makecallbare

case_file_contains makecall-bare-callee app/greet.go "func Greet() {"
case_file_contains makecall-bare-call app/main.go "Greet()"

# --- session: the working context (John's database journey) ------------------
# The session-context build (2026-11-02): four lines building
# one database package, the last two aimed by bare name only.
# The chat carries the work — the file it just created, the
# folder it is working in — and prints each reading with the
# full path, so the connection lands in jake/database/ and
# never at the project root, where a bare name points in a
# cold session.

cat > "$SCRATCH/.inputs/sessionctx.txt" <<'EOF'
create file jake/database/database.db
y
create file jake/database/database.go
y
add a database connection function named Connect to database.go
y
add the file db.go
y
/quit
EOF
run_session sessionctx

case_transcript_contains ctx-journey-note sessionctx 'Reading "database.go" as jake/database/database.go — the file you just created.'
case_file_contains ctx-journey-connect jake/database/database.go "func Connect() (*sql.DB, error)"
case_transcript_contains ctx-journey-folder-note sessionctx 'Reading "db.go" as jake/database/db.go — the folder you'\''re working in.'
case_file_contains ctx-journey-dbfile jake/database/db.go "package database"
case_dir_absent ctx-journey-no-root-database database.go
case_dir_absent ctx-journey-no-root-db db.go

# --- session: the talk seat -------------------------------------------------
# The local talk seat (2026-11-03): lines no deterministic
# claim owns get one composed, read-only answer under the
# gollemer [talk]> tag — project facts from the index, honest
# not-here answers for names the project doesn't have, and
# plain conversation with no context at all. It never writes
# and never gates, so these sessions carry no "y" lines.

cat > "$SCRATCH/.inputs/talk.txt" <<'EOF'
what does Login take?
where is Frobnicate defined?
/quit
EOF
run_session talk

case_transcript_contains talk-fact-login talk "Login" "http.ResponseWriter"
case_transcript_contains talk-marker talk "gollemer [talk]>"
case_transcript_contains talk-trap-honest talk "Frobnicate" "don't see"

cat > "$SCRATCH/.inputs/talkchat.txt" <<'EOF'
hello
how are you?
/quit
EOF
run_session talkchat

case_transcript_contains talk-conversation-reply talkchat "gollemer"
case_transcript_lacks talk-conversation-no-error talkchat "[error:"
case_transcript_lacks talk-conversation-no-invention talkchat "Frobnicate"

# --- session: memory + system state awareness ------------------------------
# Two sessions sharing one isolated HOME (the store lives in
# ~/.gollemer_memory.json — under the scratch home here, and
# never inside the project). Session A captures a preference
# explicitly, asks its branch, and lands a piece; its /quit
# saves the session summary. Session B must open with the
# one continuity line, recall the preference and the piece,
# answer the changed-files question with its honest baseline
# caveat, and forget exactly what the replies confirm. A
# minimal .git/HEAD gives the branch arm a fixture branch.

mkdir -p "$SCRATCH/.git" "$SCRATCH/.home"
printf 'ref: refs/heads/handtest-memory\n' > "$SCRATCH/.git/HEAD"

run_session_with_home() {
	local name="$1"
	local out="$SCRATCH/transcripts/$name.txt"
	: > "$out"
	(cd "$SCRATCH" && env -u GOEXPERIMENT HOME="$SCRATCH/.home" timeout "$SESSION_TIMEOUT" "$BIN" -real-chat -domain unified) \
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

cat > "$SCRATCH/.inputs/memory1.txt" <<'EOF'
remember: I prefer tabs over spaces
what branch am I on?
create file memoryprobe.go
y
add a struct named MemoryProbe to memoryprobe.go
y
what have we made?
what do you remember
/quit
EOF
run_session_with_home memory1

case_transcript_contains memory-capture-echo memory1 "I'll remember: I prefer tabs over spaces."
case_transcript_contains memory-branch memory1 "You're on branch handtest-memory."
case_transcript_contains memory-made memory1 "MemoryProbe" "memoryprobe.go"
case_transcript_contains memory-recall-first memory1 "I prefer tabs over spaces" "For this project"
case_file_contains memory-session-saved .home/.gollemer_memory.json "I prefer tabs over spaces" "MemoryProbe"

cat > "$SCRATCH/.inputs/memory2.txt" <<'EOF'
what do you remember
what files changed?
forget my tabs over spaces
forget this project
/quit
EOF
run_session_with_home memory2

case_transcript_contains memory-continuity memory2 "Last time here:" "MemoryProbe"
case_transcript_contains memory-changed-caveat memory2 "don't have a project-read baseline" "won't call that a git status"
case_transcript_contains memory-forget memory2 'Forgotten: your preference "I prefer tabs over spaces"' "Forgotten: this project's notes and session summaries"

# --- sessions: the teach seat ----------------------------------------------
# The chat learns a piece kind from ONE example in the project
# and keeps it as the user's own data: the store lives in the
# same isolated HOME as the memory store
# (~/.gollemer_taught.json), never in the project. Session A
# lands a form handler named Jake with the book's own line,
# teaches it as a "signup form", and assembles one in another
# folder under a new name — the plan carries the example's
# provenance. Session B (same HOME, fresh chat) still knows
# the kind — taught pieces are the user's, across sessions —
# lists it, forgets it, and then meets the honest
# unknown-kind answer for the same ask.

cat > "$SCRATCH/.inputs/teach1.txt" <<'EOF'
create a folder teachjim
y
create a file teachjim/main.go
y
add a form handler named Jake to teachjim/main.go
y
learn this as a signup form from Jake in teachjim/main.go
create a folder teachsite
y
create a file teachsite/site.go
y
add a signup form named Signup to teachsite/site.go
y
what have I taught you
/quit
EOF
run_session_with_home teach1

case_transcript_contains teach-learned teach1 "Here's what I learned from Jake in teachjim/main.go" 'The kind: "signup form"' "the code stays exactly as in Jake"
case_file_contains teach-store-kept .home/.gollemer_taught.json '"signup form"' '"Jake"' 'teachjim/main.go'
case_dir_absent teach-store-not-in-project .gollemer_taught.json
case_file_contains teach-use-landed teachsite/site.go "func Signup(" 'r.FormValue("email")'
case_transcript_contains teach-use-provenance teach1 "From your example Jake in teachjim/main.go, taught"
case_transcript_contains teach-listed teach1 '"signup form" — from your example Jake (function) in teachjim/main.go'

cat > "$SCRATCH/.inputs/teach2.txt" <<'EOF'
what have I taught you
forget the signup form
add a signup form named Signup to teachsite/site.go
/quit
EOF
run_session_with_home teach2

case_transcript_contains teach-forgotten teach2 'Forgotten: your "signup form" — from your example Jake in teachjim/main.go'
case_transcript_contains teach-after-forget teach2 "the concept book has no piece by that name"

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
