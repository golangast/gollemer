#!/usr/bin/env bash
# paces.sh — the multi-file end-to-end workout for the gollemer chat.
#
#   make paces
#
# Plays a beginner building ONE real multi-file project through
# the REAL chat binary, in sequence, the way a user actually
# uses it: folder + webserver, handlers, middleware, security
# pieces, a named database connection, the gorqlite dependency
# gate, a struct defined in one file and used from others,
# cross-file writes, a created-target compound, serving an html
# page, running the program and watching its log, the terminal
# basics, the ambiguous-piece guidance, a cross-file rename,
# the rm refusal, the one-ask make-and-call (a new function
# in one file, called from another, in a single gated plan),
# and the handler wiring (two placed handlers registered with
# the webserver — one in its own file, one across packages),
# and a homepage that takes over the webserver's starter
# reply (the page family: page + handler + registration in
# one ask), and a clarifying question answered in one word
# (the bare wiring ask with two handlers waiting asks which
# one, and the name completes it), a follow-up that repeats
# the last piece (its name asked the same way), and a taught
# name for a file used across two chats (its sessions run
# with HOME isolated under the scratch, so the vocabulary
# store it writes is the scratch's, never the developer's).
# After every landing step it greps the
# disk for what the chat claimed, and builds every package in
# the scratch project — a step that "succeeds" in chat but
# leaves the tree unbuildable is recorded as a failure.
#
# This is the 2026-11-01 paces workout, kept: its first run
# scored 46/58 and traced three real defects (declarations
# refusing files that don't exist yet, pages 404ing from the
# project root, and the named-function-plus-"that uses"
# hybrid falling between two grammars). The fixes landed;
# the stages now run the direct forms those fixes earned.
#
# Everything happens under paces/ at the repo root (wiped and
# rebuilt each run, marker-guarded), the handtest pattern:
# the chat is built once from THIS checkout and finds its
# data through a symlink while its edits land in the scratch
# project. Results land in paces/results.txt; transcripts in
# paces/transcripts/.
#
# Needs: bash, go on PATH, curl, coreutils timeout.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
SCRATCH="$ROOT/paces"
BASE="$SCRATCH"
PROJ="$SCRATCH/project"
BIN="$SCRATCH/.bin/gollemer"
TRANSCRIPTS="$SCRATCH/transcripts"
RESULTS="$SCRATCH/results.txt"
MARKER=".paces-owned"
SESSION_TIMEOUT="${PACES_TIMEOUT:-300}"

PASSN=0
FAILN=0

say() { printf '%s\n' "$*"; }
rec() { printf '%s\n' "$*" >> "$RESULTS"; }
ok()  { PASSN=$((PASSN + 1)); say "  PASS  $1"; rec "PASS  $1"; }
bad() { FAILN=$((FAILN + 1)); say "  FAIL  $1  ($2)"; rec "FAIL  $1  ($2)"; }
die() { say "paces: $*" >&2; exit 1; }

step() { say ""; say "== $1 =="; rec ""; rec "== $1 =="; }

# --- safety: the scratch is ours, and only ours --------------

[ -f "$ROOT/go.mod" ] && [ -f "$ROOT/data/concepts.json" ] ||
	die "run me from the gollemer checkout (scripts/paces.sh or make paces)"
command -v go >/dev/null 2>&1 || die "go is not on PATH"
command -v timeout >/dev/null 2>&1 || die "timeout (coreutils) is not on PATH"
command -v curl >/dev/null 2>&1 || die "curl is not on PATH"

if [ -L "$SCRATCH" ]; then
	die "paces/ is a symlink — I won't write through it. Remove it and run again."
fi
if [ -e "$SCRATCH" ] && [ ! -f "$SCRATCH/$MARKER" ]; then
	die "paces/ already exists and isn't a paces scratch — move it away and run again."
fi
rm -rf -- "$SCRATCH"
mkdir -p "$PROJ" "$SCRATCH/.bin" "$TRANSCRIPTS" "$SCRATCH/.inputs"
touch "$SCRATCH/$MARKER" "$PROJ/$MARKER"
: > "$RESULTS"

BASE_HASH="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
say "paces: base (checkout HEAD) = $BASE_HASH"
rec "base: $BASE_HASH"

# --- build the chat once, from this checkout ------------------
# The Makefile's env, for the chat build only — the sessions
# below strip it again (the handtest rule: the scratch module
# must verify its edits without the experiment).

say "paces: building the chat from this checkout…"
(cd "$ROOT" && GOEXPERIMENT="${GOEXPERIMENT:-simd}" CGO_ENABLED="${CGO_ENABLED:-1}" go build -o "$BIN" .) || die "go build failed"

# --- the scratch project ---------------------------------------
# Its own module. The data symlink is how the chat finds the
# clone's models/book while its edits land in the scratch. The
# Makefile carries a start target on purpose: "run the program
# in api/main.go" must never become this project's make start.

ln -s "$ROOT/data" "$PROJ/data"

cat > "$PROJ/go.mod" <<'EOF'
module paces

go 1.21
EOF

printf 'start:\n\t@echo PACES-MAKE-START-RAN\n' > "$PROJ/Makefile"

# --- session drivers --------------------------------------------

# run_session <name> — static input file .inputs/<name>.txt,
# handtest-style: every gate consumes the next piped line.
run_session() {
	local name="$1"
	local out="$TRANSCRIPTS/$name.txt"
	: > "$out"
	(cd "$PROJ" && env -u GOEXPERIMENT timeout "$SESSION_TIMEOUT" "$BIN" -real-chat -domain unified) \
		< "$BASE/.inputs/$name.txt" >> "$out" 2>&1
	local rc=$?
	if [ "$rc" -eq 124 ]; then
		bad "session-$name" "timed out after ${SESSION_TIMEOUT}s (hang)"
	fi
}

# fifo_open <name> / fifo_send <line> / fifo_waitfor <name> <fragment> <secs>
# fifo_close — for the two sessions that need the outside world
# mid-conversation (the gorqlite install, the HTTP fetch).
FIFO=""
FIFO_OUT=""
fifo_open() {
	FIFO="$BASE/.inputs/$1.fifo"
	FIFO_OUT="$TRANSCRIPTS/$1.txt"
	rm -f "$FIFO"
	mkfifo "$FIFO"
	: > "$FIFO_OUT"
	(cd "$PROJ" && env -u GOEXPERIMENT timeout "$SESSION_TIMEOUT" "$BIN" -real-chat -domain unified) \
		< "$FIFO" >> "$FIFO_OUT" 2>&1 &
	FIFO_PID=$!
	exec 7> "$FIFO"
}
fifo_send() { printf '%s\n' "$1" >&7; }
fifo_waitfor() { # <fragment> <secs>
	local frag="$1" secs="${2:-60}" i
	for i in $(seq 1 "$secs"); do
		if grep -qF -- "$frag" "$FIFO_OUT"; then
			return 0
		fi
		if ! kill -0 "$FIFO_PID" 2>/dev/null; then
			return 1
		fi
		sleep 1
	done
	return 1
}
fifo_close() {
	exec 7>&-
	wait "$FIFO_PID" 2>/dev/null || true
	FIFO=""
}

# --- assertions ---------------------------------------------------

has() { # <case> <path-under-proj> <fragment>…
	local id="$1" rel="$2"
	shift 2
	local f
	for f in "$@"; do
		if [ ! -f "$PROJ/$rel" ] || ! grep -qF -- "$f" "$PROJ/$rel"; then
			bad "$id" "missing '$f' in project/$rel"
			return
		fi
	done
	ok "$id"
}

hasnt() { # <case> <path-under-proj> <fragment>
	local id="$1" rel="$2" f="$3"
	if [ -f "$PROJ/$rel" ] && grep -qF -- "$f" "$PROJ/$rel"; then
		bad "$id" "unexpected '$f' in project/$rel"
		return
	fi
	ok "$id"
}

absent() { # <case> <path-under-proj>
	local id="$1" rel="$2"
	if [ -e "$PROJ/$rel" ]; then
		bad "$id" "project/$rel exists but shouldn't"
		return
	fi
	ok "$id"
}

told() { # <case> <session> <fragment>…
	local id="$1" sess="$2"
	shift 2
	local f
	for f in "$@"; do
		if ! grep -qF -- "$f" "$TRANSCRIPTS/$sess.txt"; then
			bad "$id" "missing '$f' in the $sess transcript"
			return
		fi
	done
	ok "$id"
}

notold() { # <case> <session> <fragment>
	local id="$1" sess="$2" f="$3"
	if grep -qF -- "$f" "$TRANSCRIPTS/$sess.txt"; then
		bad "$id" "unexpected '$f' in the $sess transcript"
		return
	fi
	ok "$id"
}

builds() { # <case>
	# Per-package builds with the output thrown away. A bare
	# `go build ./...` is NOT a safe check here: while the
	# project has exactly ONE package and it is a main package
	# in api/, go writes the binary to ./api and fails with
	# 'build output "api" already exists and is a directory'
	# even though the code compiles (that artifact cost
	# build-step1 a false FAIL in the first recorded run).
	local id="$1"
	local log="$TRANSCRIPTS/build-$id.txt"
	local pkg fail=0
	: > "$log"
	while IFS= read -r pkg; do
		(cd "$PROJ" && env -u GOEXPERIMENT timeout "$SESSION_TIMEOUT" go build -o /dev/null "$pkg") >> "$log" 2>&1 || fail=1
	done < <(cd "$PROJ" && env -u GOEXPERIMENT go list ./... 2>> "$log")
	if [ "$fail" -eq 0 ]; then
		ok "build-$id"
	else
		bad "build-$id" "per-package go build failed (see transcripts/build-$id.txt)"
	fi
}

no_stray_main() { # <case>
	local id="$1" hits
	hits="$(grep -rl --include='*.go' -F 'func main()' "$PROJ" | grep -v "^$PROJ/api/main.go$" || true)"
	if [ -n "$hits" ]; then
		bad "$id" "func main() found outside api/main.go: $hits"
		return
	fi
	ok "$id"
}

say "paces: running the journey (transcripts in transcripts/)…"

# ================================================================
step "step 1: folder + webserver program"
# ================================================================

cat > "$BASE/.inputs/s1.txt" <<'EOF'
create a folder api
y
create a file api/main.go with a webserver
y
/quit
EOF
run_session s1

has step1-webserver api/main.go "ListenAndServe" "package main" "func main()"
builds step1

# ================================================================
step "step 2: form handler (folder default) + session handler (named file)"
# ================================================================

cat > "$BASE/.inputs/s2.txt" <<'EOF'
add a form handler named Signup
y
add a session handler named Session to api/main.go
y
/quit
EOF
run_session s2

has step2-form-handler handlers/signup.go "package handlers" "func Signup("
has step2-session-handler api/main.go "func Session("
builds step2

# ================================================================
step "step 3: middleware + jwt token + cors"
# ================================================================

cat > "$BASE/.inputs/s3.txt" <<'EOF'
add a middleware named Logger
y
add a jwt token in main to api/main.go
y
add a cors middleware named Cors
y
/quit
EOF
run_session s3

has step3-middleware middleware/logger.go "package middleware" "func Logger("
has step3-jwt api/main.go "JWT_SECRET"
has step3-cors security/cors.go "package security" "Access-Control-Allow-Origin"
builds step3

# ================================================================
step "step 4: named database connection (the declaration, not a snippet, not a main)"
# ================================================================

cat > "$BASE/.inputs/s4.txt" <<'EOF'
add a database connection named Connect
y
/quit
EOF
run_session s4

has step4-connect database/connect.go "package database" "func Connect() (*sql.DB, error)" "sql.Open"
hasnt step4-no-inline-in-main api/main.go "sql.Open"
no_stray_main step4-no-fabricated-main
builds step4

# ================================================================
step "step 5: gorqlite gate — absent, then installed mid-session"
# ================================================================
# The install is the suite's own offline stub form: go.mod
# gains the require and a replace pointing at a local package
# carrying exactly the API surface the gorqlite rows use.

fifo_open s5
fifo_send "create a folder store"
if fifo_waitfor "Create it?" 60; then fifo_send "y"; else bad step5-folder-gate "no Create it? gate for the store folder"; fi
fifo_send "add a gorqlite connection named Rq to store/db.go"
if fifo_waitfor "go get github.com/rqlite/gorqlite" 90; then
	ok step5-gate-line
else
	bad step5-gate-line "the go get gate line never appeared"
fi
notold step5-no-plan-yet s5 "Apply this edit?"
absent step5-nothing-written store/db.go

mkdir -p "$PROJ/gorqlitestub"
cat > "$PROJ/gorqlitestub/go.mod" <<'EOF'
module github.com/rqlite/gorqlite

go 1.21
EOF
cat > "$PROJ/gorqlitestub/gorqlite.go" <<'EOF'
package gorqlite

// Connection is the stub form of the client's connection.
type Connection struct{}

// Open mirrors the real client's signature.
func Open(connURL string) (*Connection, error) { return &Connection{}, nil }

// QueryResult mirrors the real client's result: its Err
// field is the part the book's shapes name.
type QueryResult struct {
	Err error
}

// QueryOne mirrors the real client's single-statement query.
func (c *Connection) QueryOne(sql string) (QueryResult, error) {
	return QueryResult{}, nil
}

// ParameterizedStatement mirrors the real client's
// statement-plus-arguments form.
type ParameterizedStatement struct {
	Query     string
	Arguments []interface{}
}

// WriteResult mirrors the real client's write result.
type WriteResult struct {
	Err          error
	RowsAffected int64
	LastInsertID int64
}

// WriteOneParameterized mirrors the real client's
// single-statement parameterized write.
func (c *Connection) WriteOneParameterized(ps ParameterizedStatement) (WriteResult, error) {
	return WriteResult{}, nil
}
EOF
cat >> "$PROJ/go.mod" <<'EOF'

require github.com/rqlite/gorqlite v0.0.0

replace github.com/rqlite/gorqlite => ./gorqlitestub
EOF

fifo_send "installed"
if fifo_waitfor "Apply this edit?" 90; then
	ok step5-plan-reoffered
	fifo_send "y"
else
	bad step5-plan-reoffered "the original plan was not re-offered after installed"
	fifo_send "y"
fi
sleep 3
fifo_send "/quit"
fifo_close

has step5-gorqlite-landed store/db.go "func Rq(" "github.com/rqlite/gorqlite"
no_stray_main step5-no-fabricated-main
builds step5

# ================================================================
step "step 6: struct in one file, used from others; cross-file write"
# ================================================================

# Step 6's last-but-one line is the workout's defect 3, kept
# as a regression pin: a function ask carrying BOTH the
# "named <Name>" introducer AND a "that uses <Sym>" clause
# must route to the assembled uses edit. The uses route
# writes into a file that already exists — creating the
# file first is its own prescribed step, so the create is
# part of the direct form here, not a workaround. (In the
# first run this step pre-created NOTHING and every line
# failed: declarations refused files that didn't exist yet
# — the defect-1 fix is what lets the struct and Validate
# lines above run direct.)
cat > "$BASE/.inputs/s6.txt" <<'EOF'
add a struct named Profile to api/profile.go
y
add instantiation of Profile to handler Session
y
add an Email field of type string to Profile
y
add a function named Validate to api/validate.go
y
create api/report.go
y
add a function named Report in api/report.go that uses Validate from api/validate.go
y
/quit
EOF
run_session s6

has step6-struct api/profile.go "type Profile struct"
has step6-field api/profile.go "Email string"
has step6-instantiate api/main.go "Profile{"
has step6-validate api/validate.go "func Validate("
has step6-cross-file api/report.go "func Report(" "Validate("
builds step6

# ================================================================
step "step 7: created-target compound in one line"
# ================================================================

cat > "$BASE/.inputs/s7.txt" <<'EOF'
make a form handler named Welcome in api/welcome.go and add a switch in it
y
y
/quit
EOF
run_session s7

has step7-compound-file api/welcome.go "func Welcome("
has step7-compound-piece api/welcome.go "switch"
builds step7

# ================================================================
step "step 8: serve home.html from api/main.go"
# ================================================================

cat > "$BASE/.inputs/s8.txt" <<'EOF'
serve home.html from api/main.go
y
/quit
EOF
run_session s8

has step8-servefile api/main.go "http.ServeFile"
has step8-page api/home.html "TODO"
builds step8

# ================================================================
step "step 9: run the program, watch it, stop it"
# ================================================================

fifo_open s9
fifo_send "run the program in api/main.go"
if fifo_waitfor "Run it?" 60; then fifo_send "y"; else bad step9-run-gate "no Run it? gate for api/main.go"; fi
served=""
servedpath=""
for _ in $(seq 1 90); do
	for p in / /home.html; do
		if curl --noproxy '*' -sf -o "$TRANSCRIPTS/step9-fetch.txt" --max-time 2 "http://localhost:8080$p"; then
			served=1
			servedpath="$p"
			break 2
		fi
	done
	sleep 1
done
if [ -n "$served" ]; then
	ok "step9-http-served"
	rec "step9 fetched path: $servedpath"
else
	bad step9-http-served "the api server never answered http://localhost:8080/"
fi
fifo_send "show the program log"
sleep 2
fifo_send "top"
sleep 2
fifo_send "stop the program"
sleep 2
fifo_send "top"
sleep 2
fifo_send "/quit"
fifo_close

told step9-background s9 "is running in the background" "http://localhost:8080"
told step9-visit-logged s9 "visit: GET"
told step9-top-running s9 "Programs this chat is running" "api/main.go"
told step9-stopped s9 "Stopped api/main.go"
told step9-top-after-stop s9 "It's no longer running."
notold step9-not-make-start s9 "PACES-MAKE-START-RAN"
if curl --noproxy '*' -sf -o /dev/null --max-time 2 "http://localhost:8080/"; then
	bad step9-port-free "http://localhost:8080/ still answers after stop the program"
else
	ok step9-port-free
fi

# ================================================================
step "step 10: terminal basics inside the project"
# ================================================================

EXPECT_PWD="$(cd "$PROJ" && pwd -P)"
cat > "$BASE/.inputs/s10.txt" <<'EOF'
pwd
y
ls
y
cd api
y
pwd
y
cat main.go
y
cd
y
pwd
y
/quit
EOF
run_session s10

if grep -qFx -- "$EXPECT_PWD" "$TRANSCRIPTS/s10.txt"; then
	ok step10-pwd-root
else
	bad step10-pwd-root "pwd did not print the project root $EXPECT_PWD"
fi
told step10-ls s10 "api/"
told step10-cd-announce s10 "Now working in api/"
if grep -qFx -- "$EXPECT_PWD/api" "$TRANSCRIPTS/s10.txt"; then
	ok step10-pwd-after-cd
else
	bad step10-pwd-after-cd "pwd after cd api did not print $EXPECT_PWD/api"
fi
told step10-cat s10 "ListenAndServe"
told step10-cd-home s10 "back at the folder the chat started in"

# ================================================================
step "step 11: guidance checks — guide, write nothing"
# ================================================================

md5sum "$PROJ/api/main.go" > "$BASE/.inputs/api-main-before.md5"
cat > "$BASE/.inputs/s11.txt" <<'EOF'
add handler Bob
add a database to api
/quit
EOF
run_session s11

told step11-handler-guidance s11 '"add a form handler named Bob"' '"add a session handler named Bob"'
told step11-database-guidance s11 '"add a database connection function named Connect to api"' '"add a database in main to api/main.go"'
notold step11-no-gate s11 "Apply this edit?"
absent step11-no-bob handlers/bob.go
if (cd "$PROJ" && md5sum -c "$BASE/.inputs/api-main-before.md5") > /dev/null 2>&1; then
	ok step11-main-unchanged
else
	bad step11-main-unchanged "api/main.go changed during the guidance session"
fi

# ================================================================
step "step 12: rename a function referenced from another file"
# ================================================================

cat > "$BASE/.inputs/s12.txt" <<'EOF'
rename Validate to CheckInput
y
/quit
EOF
run_session s12

has step12-decl-renamed api/validate.go "func CheckInput("
hasnt step12-old-decl-gone api/validate.go "func Validate("
has step12-caller-updated api/report.go "CheckInput("
hasnt step12-old-call-gone api/report.go "Validate("
builds step12

# ================================================================
step "step 13: rm refusal — file intact"
# ================================================================

cat > "$BASE/.inputs/s13.txt" <<'EOF'
rm api/main.go
/quit
EOF
run_session s13

told step13-rm-refused s13 "doesn't delete files"
has step13-file-intact api/main.go "package main" "func main()"

# ================================================================
step "step 14: one ask, two files — make a function and call it"
# ================================================================

# The one-ask recipe (2026-11-02): a new function declared in
# one file and called from another, in ONE plan behind ONE
# gate. The named-caller form runs cross-package (the callee
# in database/, the caller in api/'s package main, importing
# and qualifying the callee's package); the bare form hosts
# the call in api/main.go's main. One "y" per line — a second
# gate per ask would mean the recipe split into two halves.
cat > "$BASE/.inputs/s14.txt" <<'EOF'
make a function named Ping in database/ping.go and add a function named PingCheck in api/report.go that calls it
y
make a function named Ready in api/ready.go and call it in api/main.go
y
/quit
EOF
run_session s14

has step14-callee database/ping.go "func Ping() {"
has step14-caller api/report.go "func PingCheck()" "database.Ping()"
has step14-bare-callee api/ready.go "func Ready() {"
has step14-bare-call api/main.go "Ready()"
builds step14

# ================================================================
step "step 15: wire the handlers into the webserver"
# ================================================================

# The wiring recipe (2026-11-03): handlers placed earlier in
# the journey — Session in api/main.go itself (step 2), Signup
# over in the handlers package (step 2 again) — get registered
# with api's webserver, one gated line each. Session wires by
# its bare name in its own file; Signup crosses packages, so
# api/main.go gains the paces/handlers import and the
# qualified registration.
cat > "$BASE/.inputs/s15.txt" <<'EOF'
connect the Session handler to the webserver
y
connect the Signup handler to the webserver
y
/quit
EOF
run_session s15

has step15-session-wired api/main.go 'http.HandleFunc("/session", Session)'
has step15-signup-wired api/main.go 'http.HandleFunc("/signup", handlers.Signup)' '"paces/handlers"'
builds step15

# ================================================================
step "step 16: a homepage takes over the webserver's starter reply"
# ================================================================

# The page family (2026-11-04): one ask for a whole page — the
# starter page file, the handler that serves it, and its
# registration — in a single gated plan. A homepage's route
# is "/", which the webserver component's own starter handler
# has answered since step 1 with its friendly line; that
# starter exists to be replaced, and the plan replaces just
# that line (every other registration — /session, /signup —
# is untouched).
cat > "$BASE/.inputs/s16.txt" <<'EOF'
create a homepage in api
y
/quit
EOF
run_session s16

has step16-homepage-page api/index.html "TODO: your homepage heading"
has step16-homepage-wired api/main.go "func Home(" 'http.HandleFunc("/", Home)'
hasnt step16-starter-gone api/main.go "hello — your webserver is running"
builds step16

# ================================================================
step "step 17: a clarifying question, answered in one word"
# ================================================================

# Clarifying questions (2026-11-05): an ask missing one
# piece of information no longer ends in guidance prose —
# the chat asks ONE question and the next line completes
# the original ask. Two fresh handlers land; the bare
# wiring ask can't choose between them (Welcome, from an
# earlier step, is also still waiting), so it asks; the
# one-word answer "Alpha" completes it behind the usual
# gate. Asked again, it asks again — Beta and Welcome are
# both still unregistered — and "Beta" completes that one.
cat > "$BASE/.inputs/s17.txt" <<'EOF'
add a form handler named Alpha to api/main.go
y
add a form handler named Beta to api/main.go
y
connect the handler to the webserver
Alpha
y
connect the handler to the webserver
Beta
y
/quit
EOF
run_session s17

told step17-question s17 "More than one handler isn't registered yet" "1. Alpha" "2. Beta" "Answer with the name or the number"
has step17-alpha-wired api/main.go 'http.HandleFunc("/alpha", Alpha)'
has step17-beta-wired api/main.go 'http.HandleFunc("/beta", Beta)'
builds step17

# ================================================================
step "step 18: a follow-up repeats the last piece"
# ================================================================

# Richer follow-ups (2026-11-05): the last landed piece
# repeats from the session's own shorthand. A struct lands;
# a bare "another one" asks the new struct's name through
# the clarifying machinery and the one-word answer completes
# the repeat through the book's own claim; a named follow-up
# then lands a third directly. Session state only — this
# step touches no store.
cat > "$BASE/.inputs/s18.txt" <<'EOF'
add a struct named PaceProbe to api/main.go
y
another one
PaceProbeTwo
y
another one named PaceProbeThree in api/main.go
y
/quit
EOF
run_session s18

told step18-question s18 "Another struct — what should I call it?" "Answer with one name"
has step18-second api/main.go "type PaceProbeTwo struct"
has step18-third api/main.go "type PaceProbeThree struct"
builds step18

# ================================================================
step "step 19: his own name for a file, taught once"
# ================================================================

# Personal vocabulary (2026-11-05): a name for a file is
# taught once and then works in any sentence that names the
# file, the reading printed. The store is HOME-scoped, and
# paces must never write the developer's real one — so these
# two sessions run with HOME isolated under the scratch,
# handtest-style. Session A teaches the name and uses it;
# session B (fresh chat, same isolated HOME) still knows it,
# uses it again, and forgets it.
run_session_home() {
	local name="$1"
	local out="$TRANSCRIPTS/$name.txt"
	: > "$out"
	(cd "$PROJ" && env -u GOEXPERIMENT HOME="$BASE/.home" timeout "$SESSION_TIMEOUT" "$BIN" -real-chat -domain unified) \
		< "$BASE/.inputs/$name.txt" >> "$out" 2>&1
	local rc=$?
	if [ "$rc" -eq 124 ]; then
		bad "session-$name" "timed out after ${SESSION_TIMEOUT}s (hang)"
	fi
}

cat > "$BASE/.inputs/s19a.txt" <<'EOF'
call api/main.go the api file
add a print line in main to the api file
y
/quit
EOF
run_session_home s19a

cat > "$BASE/.inputs/s19b.txt" <<'EOF'
add a print line in main to the api file
y
forget the api file
/quit
EOF
run_session_home s19b

told step19-reading s19a 'Reading "the api file" as api/main.go — your name for it.'
has step19-landed api/main.go "fmt.Println"
told step19-persists s19b 'Reading "the api file" as api/main.go — your name for it.'
told step19-forgotten s19b 'Forgotten: "the api file" — your name for api/main.go in this project.'
builds step19

# --- scoreboard ---------------------------------------------------

TOTAL=$((PASSN + FAILN))
say ""
say "paces scoreboard: $PASSN passed, $FAILN failed, $TOTAL total"
rec ""
rec "scoreboard: $PASSN passed, $FAILN failed, $TOTAL total"
if [ "$FAILN" -ne 0 ]; then
	exit 1
fi
exit 0
