# Gollemer

**A tiny language model, written in pure Go, that helps you learn Go by talking to it.**

You open one chat in your terminal. It reads the Go project you're in,
explains it in plain English, answers Go questions, and makes small
edits for you — but only after showing you the exact change and getting
your yes. Everything runs on your machine. There is no rented AI model
behind it, no account, and nothing you type leaves your computer.

## Why Gollemer exists

Most AI coding tools are enormous models running on someone else's
computers. You can't read their code, you can't run them offline, and
when they edit your files you mostly have to take their word for it.

Gollemer is the opposite bet:

- **Small enough to read.** The whole thing is one Go program, and the
  "brains" inside it are small trained models you could, in principle,
  inspect yourself. It is a learning project as much as a tool.
- **Built for beginners.** It was made by a beginner, for the first
  months of learning Go: plain words first, jargon explained, and every
  answer tied to the real code in front of you — never a made-up file
  or a function that doesn't exist.
- **It teaches instead of taking over.** It explains your project,
  shows you where a change belongs and why, and lets you say the words
  while it does the typing. The edits it makes are small, known pieces
  — and each one is shown to you first, then checked by actually
  building your project.
- **Honest about its limits.** The small models inside are measured,
  and their scores — failures included — are part of the project. Where
  a model isn't good enough to be trusted, deterministic code (plain,
  ordinary programming, no AI) does the job instead. See
  [How it works](#how-it-works-in-plain-words) and
  [Honest limits](#honest-limits).

## The 60-second start

You need Go installed. Then:

```sh
git clone https://github.com/golangast/gollemer
cd gollemer
make start
```

`make start` opens the chat. (It needs a real terminal — if you pipe
it, it prints one line telling you so.) `make chat` opens the same
chat. `make guide` prints a short start-here guide without opening
anything.

The first screen looks like this:

```
╭─ GOLLEMER ────────────────────────────────────────────────────────╮
│   ( o o )   a small Go companion — reads your code,               │
│    \___/    explains it in plain English                          │
│                                                                   │
│   ▸ start — a six-stop guided tour of this project                │
│   ▸ ask in plain words — "what does this program do"              │
│   ✦ new here? say "teach me" — two minutes, learn by doing        │
│                                                                   │
│   [ your-project ]  [ 17 brains ready ]                           │
╰───────────────────────────────────────────────────────────────────╯
```

The chat works on the folder you started it in. Open it inside any Go
project and it will read *that* project. Brand new? Say `teach me` and
it walks you through the basics on a tiny bundled example.

## Things you can say

You talk in plain sentences. Every reply is tagged with the part of
Gollemer that answered, like `gollemer [go]>`, so you always know who
is talking.

*The exchanges below are real — captured from the chat on this code,
on 2026-11-01. Long diffs are trimmed with `...`.*

### Just talk, and ask about Go

```
you> have you ever played soccer
gollemer [social]> only for fun with friends. i am better at cheering than scoring.

you> what is a goroutine
gollemer [go]> A goroutine is a lightweight thread managed by the Go runtime: go doWork(). Thousands of goroutines cost little. They run concurrently with the caller — use channels or sync primitives to coordinate, never shared memory without synchronization.
```

Questions and conversation that no specialist brain owns land with
the talk seat, tagged `gollemer [talk]>`. It talks about *your*
project from the same index the analyzer builds — what a function
takes, what's in a file, what a piece you just added does ("what
does it do?" works, and the answer names what "it" was read as) —
and it holds a conversation with the local social model. It runs
entirely on your machine: no keys, no network. Before a talk reply
prints, any project name in it is checked against your code, and a
name that isn't there is marked instead of presented as fact; ask
about a function the project doesn't have and it says so plainly.
The talk seat only explains and discusses — when you ask it to
*do* something, doing still belongs to the deterministic side:
say the exact sentence and the change goes through the same plan
and yes-or-no gate as always.

It also knows the state of the moment and remembers what you
tell it to. Ask "what branch am I on?", "what files changed?",
or "what have we made?" and the answer comes from the chat's own
state — the clock, this machine, the branch in `.git/HEAD`, and
the Go files that differ since its last read of the project
(that last one is a read-baseline check, not a full git status).
Say "remember: I prefer …" or "note: …" and it echoes back
exactly what it stored; "what do you remember" shows it, and
"forget this project", "forget my …", and "forget everything"
remove exactly what their replies confirm. Memory lives in
`~/.gollemer_memory.json`, outside every project. When a session
ends it saves a short summary of what was built, and the next
session in the same project opens with one "Last time here:" line.

### Ask what a project does

```
you> analyze this project
▸ IN PLAIN ENGLISH
│ readmecap is a runnable Go program: 1 packages, about 33 lines of Go. It starts in main.go.

▸ STORY: what happens when you run it, in plain words
│ Everything starts at main() in main.go.
│ main() calls Add(): Add increases the count by one
│ main() calls Value(): Value reports the current count

▸ WHERE TO START
1. Start at the entry point:
  • main.go:5 — func main
2. Then read the engine room (most-called functions):
   main.Counter.Add  (1 callers)  counter.go:12
   ...
```

It goes on with a picture of what calls what, the biggest files, and
which packages depend on which — all read from your actual code. You
can also ask about one piece (`what does Add do`, `show me Add`,
`walk me through Add`), paste a GitHub issue link and ask where the
fix belongs, or just paste a Go compiler error and get it translated
into plain words.

### Learn how to build something

```
you> i want to build a webserver
Resources for building webserver — from your Advocates doc:

• Restful resource — https://www.programming-books.io/essential/go/handling-http-method-accessing-query-strings-request-body-2f90f3ba9eab401699dbb119feab6665
• Go examples — https://gowebexamples.com
• ...

+19 more in "Beginner", "General" and 2 other topics.

From Effective Go:
A web server — https://go.dev/doc/effective_go
```

Saying what you want to build gets you a pack of reading — links from
the project maintainer's own collected list (the "Advocates" doc,
below) plus the shelf of canonical Go texts. Say `resources for databases`
for a different topic.

### Create folders and files

```
you> create a folder jane
gollemer [composer]> Here's what I'll create — nothing is written yet:

◆ CREATE
  folder  jane/
Create it? [y/n] gollemer [composer]> Done — created jane/. I checked — it's there.

you> create jane/main.go
gollemer [composer]> Here's what I'll create — nothing is written yet:

◆ CREATE
  file    jane/main.go
          package main

          func main() {
          	// TODO: write the body
          }
Create it? [y/n] gollemer [composer]> Done — created jane/main.go. I compiled it to check — it builds.
```

A file can carry a whole starter program, too:

```
you> create a file webtest/main.go with a webserver
◆ CREATE
  file    webtest/main.go
          package main
          ...
          // A tiny webserver: every page gets one friendly line back.
          // Run it with: go run main.go — then open http://localhost:8080
          func main() {
          ...
          }
Create it? [y/n] gollemer [composer]> Done — created webtest/main.go. I compiled it to check — it builds.

you> serve home.html to webtest/main.go
gollemer [composer]> Serve webtest/home.html from the server already in webtest/main.go: one registration, just before its ListenAndServe, answers http://localhost:8080/home.html with the page. The page webtest/home.html is created fresh with a starter page (its body is a TODO).
...
Apply this edit? [y/n] gollemer [composer]> Done — main.go serves home.html via http.ServeFile, at webtest/main.go:16. I built the project and checked that it's really there.
```

Ask twice and it tells you the truth instead of doing it again:

```
you> serve home.html to webtest/main.go
gollemer [composer]> Not doing that edit: webtest/main.go already serves webtest/home.html at http://localhost:8080/home.html — nothing to add.
```

(A page named `index.html` is wired at its folder's address —
`http://localhost:8080/` for a page beside its program — because
the server redirects every `index.html` request to that address
anyway; the plan says so when it offers one.)

### Add pieces to your code

Gollemer keeps a book of **135 pieces** of Go — small, known shapes
like a print line, a map lookup, a form handler, a JWT check. Ask for
one in plain words. If you don't name a file, web pieces go into the
folder where they belong, as their own little package:

```
you> add a form handler named SignUp
gollemer [composer]> Create the handlers folder and handlers/signup.go in it (package handlers) with the form handler SignUp, from the concept book.
SignUp is placed in the handlers folder as package handlers — placed, not wired: nothing registers the handler with a server yet. To register it, say "connect the SignUp handler to the webserver" and I'll add the one registration line to your server, behind the same gate.
...
Apply this edit? [y/n] gollemer [composer]> Done — signup.go now declares the form handler SignUp in package handlers, at handlers/signup.go:7. I built the project and checked that it's really there.
```

Handlers go to `handlers/`, middleware to `middleware/`, security
pieces to `security/`, database pieces to `database/`. Name a file —
`add a cors middleware named Cors to app/main.go` — and the piece goes
exactly there instead.

A placed handler answers nothing until the server knows about it.
Connecting it is one more sentence — one registration line, added
just before the serve call in whichever file runs the server:

```
you> connect the SignUp handler to the webserver
gollemer [composer]> Wire SignUp into the webserver in webtest/main.go: one line — http.HandleFunc("/signup", handlers.SignUp) — added in main just before it serves, so http://localhost:8080/signup is answered by SignUp (declared at handlers/signup.go:7).
...
Apply this edit? [y/n] gollemer [composer]> Done — main.go registers SignUp at /signup, at webtest/main.go:17. I built the project and checked that it's really there.
```

The handler answers at `/` + its name, or at the path you name
(`connect the Jake handler to the webserver at /contact`). A handler
in the server's own file wires by its bare name; one in another
package is imported and qualified, as above. If exactly one handler
is unconnected, `connect the handler to the webserver` finds it on
its own. Ask twice and it tells you the truth — already registered,
nothing to add — and a path another handler already owns is refused
with that handler named.

The security pieces explain themselves as they land:

```
you> add a jwt token in main to app/main.go
gollemer [composer]> Add a jwt at the end of main in app/main.go, from the concept book: secret := os.Getenv("JWT_SECRET")
Anyone can READ a JWT's claims — the signature proves they weren't changed, it does not hide them — so a token never carries a password or a secret. And the signing secret here comes from the JWT_SECRET environment variable: a secret written in code is a secret published with the code.
...
Apply this edit? [y/n] gollemer [composer]> Done — main.go now has the jwt in main, at app/main.go:18. I built the project and checked that it's really there.
```

### Make a function in one file and call it from another

One sentence can do both halves — a brand-new function in one
file, and the call to it in a second file — as a single plan
behind a single `[y/n]`:

```
you> make a function named Validate in api/validate.go and add a function named Report in api/report.go that calls it
gollemer [composer]> Here's the edit I assembled from this project's own code:

Make Validate in api/validate.go and call it in api/report.go: Validate lands in api/validate.go, and the caller is a new function Report in api/report.go, whose body calls Validate.
This is an empty scaffold — Validate has a name and a body to fill in, but no parameters, no results, and nothing that does work yet.
The api folder's files are package main, so the new file carries that package name, not the folder's.
The call passes Validate no arguments — the declaration this plan writes takes none.
One plan covers both files: if either side fails to build, both files go back exactly as they were.

--- a/api/validate.go
+++ b/api/validate.go
@@ -0,0 +1,5 @@
+package main
+
+func Validate() {
+	// TODO: write the body
+}
--- a/api/report.go
+++ b/api/report.go
@@ -1 +1,7 @@
 package main
+
+// Report calls Validate — assembled by gollemer from api/validate.go:3.
+// It starts the call with placeholder values and handles the results; the real work is yours to write.
+func Report() {
+	Validate()
+}
Apply this edit? [y/n] gollemer [composer]> Done — Validate now exists in api/validate.go and is called from api/report.go, at api/report.go:5. I built the project and checked that it's really there.
```

The caller is assembled from the declaration the same plan is
writing — so the two halves can never disagree. When the files
live in different packages, the calling file imports the new
function's package and qualifies the call, the same way a call
to an existing function is written. The new function's body is
the book's TODO scaffold, and the plan says so — filling it in
is your next step.

Shorter forms work too: `... and call it from main in
api/main.go` puts the call inside an existing function you
name, and `... and call it in api/main.go` puts it at the end
of that file's `main`. If the file has no `main` to hold it,
Gollemer refuses and spells out both working forms instead of
inventing a caller name for you.

### Connect a database

```
you> add a database connection function named Connect
gollemer [composer]> Create the database folder and database/connect.go in it (package database) with the database connect Connect, from the concept book.
Returns a ready connection pool: the handle from sql.Open, proved live by a Ping before the pool settings are applied. It compiles exactly as written — but until you add your driver's blank import (no driver ships with Go), the Ping fails at run time with "unknown driver", because Open only records the driver name and the connection string; nothing has registered that name yet.
...
+func Connect() (*sql.DB, error) {
+	// TODO: no database driver ships with Go — add yours as a blank import at the top of this file, like _ "github.com/go-sql-driver/mysql" for MySQL, and put its registered name in the Open call below
+	db, err := sql.Open("TODO: driver name", "TODO: connection string")
```

The `TODO`s are the honest parts: which database driver to use is
*your* choice, so Gollemer writes the shape and leaves the choice
visible instead of guessing.

Some pieces need a library your project doesn't have yet. Then
Gollemer says so, prints the exact command for *you* to run, and
waits — it never downloads anything itself:

```
you> add a gorqlite connection named Connect
gollemer [composer]> That ask — a gorqlite connection named Connect (it would be created as database/connect.go) — needs the gorqlite client, and this project doesn't have it yet — the code would import github.com/rqlite/gorqlite, and nothing compiles against a dependency the project doesn't list. I don't fetch dependencies for you, so run this in the project first:
gollemer [composer]>   go get github.com/rqlite/gorqlite
gollemer [composer]> Once you've run it, tell me "installed" and I'll offer a gorqlite connection named Connect (it would be created as database/connect.go) again as a plan for your okay.

you> installed
gollemer [composer]> github.com/rqlite/gorqlite is in go.mod now — here's the plan for a gorqlite connection named Connect (it would be created as database/connect.go) again:
gollemer [composer]> Here's the edit I assembled from this project's own code:
...
Apply this edit? [y/n] gollemer [composer]> Done — connect.go now declares the gorqlite connection Connect in package database, at database/connect.go:7. I built the project and checked that it's really there.
```

Say "installed" before it's true and it checks, tells you plainly
that the library isn't in `go.mod` yet, and keeps waiting.

### Put a piece on an exact line

```
you> add a print line in main to app/main.go after line 9
gollemer [composer]> Add a print line in main in app/main.go, after line 9: `fmt.Println("one")`, from the concept book: fmt.Println("")

--- a/app/main.go
+++ b/app/main.go
@@ -7,6 +7,7 @@
 
 func main() {
 	fmt.Println("one")
+	fmt.Println("")
 	fmt.Println("two")
 }

Apply this edit? [y/n] gollemer [composer]> Done — main.go now has the print line in main, at app/main.go:10. I built the project and checked that it's really there.
```

The plan quotes the line it's aiming at, pieces only ever land
*between* whole statements, and a line number that doesn't make sense
gets a refusal that explains why — never a silent guess.

### Run the project's commands

Ask in words; the chat names the exact command and offers to run it:

```
you> run make help
gollemer [makefile]> run make help
[run it here? 'make help'] [y/n]: y
[running: make help — Ctrl-C to interrupt]
Gollemer commands:
  make start            Talk to gollemer — opens the unified chat (terminal only)
  ...
[done]
```

Only commands from the project's own Makefile can run this way, each
behind that `[y/n]`. There is no shell: the chat can't be talked into
running anything else.

### Run your program

`run the program <file>.go` runs *your* program — never a Makefile
target that happens to sound similar. Behind the same run gate, the
chat builds it first (a compile error is reported as one, with the
compiler's own words — nothing is ever said to have run that didn't),
then starts it. A file that shares its folder with other `.go` files
runs as its whole package; a lone file runs as itself.

A program that finishes is reported with its real exit state and its
own output. A program still running after a few seconds — a server —
stays running in the background while the chat gives you your prompt
back:

```
you> run the program jake/main.go
Run it? go run jake/main.go [auto-yes: your 'do all' directive]
gollemer [runner]> [running: jake/main.go — its output is being kept]
gollemer [runner]> jake/main.go is running in the background — serving at http://localhost:8080. Say "show visits" to read its log (handler visits included), or "stop the program" to stop it.
gollemer [runner]> Opening http://localhost:8080 in your browser.
you> show visits
gollemer [runner]> Here's what jake/main.go has printed (2 lines):
listening at http://localhost:8080 — press Ctrl+C to stop
visit: GET /
gollemer [runner]> Visits appear in this log when the program prints them — servers Gollemer creates print one line per visit.
gollemer [runner]> It's still running.
you> top
gollemer [runner]> Programs this chat is running — only the ones it started, not everything on this computer:
gollemer [runner]> jake/main.go — running for 1 second, serving at http://localhost:8080; 2 lines logged, 1 visit. Say "stop the program" to stop it.
you> stop the program
gollemer [runner]> Stopped jake/main.go — it's no longer running. Its log kept 2 lines; say "show the program log" to read them.
```

(From a real run on this tree, 2026-11-01: the folder and the
webserver file were created by the asks in the sections above, the
visit is a real HTTP fetch, and the browser really was opened at the
address the program printed. If the browser can't be opened, the chat
says so and prints the address on its own line instead.)

The rules around it:

- **One at a time.** Running a second program while one runs answers
  with the first one's live status, not a second instance.
- **Stop means the program.** `stop the program`, `stop it`,
  `stop the server` — and bare `stop` while one runs — stop the
  program and keep the chat open. With nothing running, those ask
  plainly ("No program is running right now") and bare `stop` keeps
  its old meaning: it ends the chat. Closing the chat (`/quit`, or
  the input ending) stops any running program — nothing is orphaned.
- **`top` shows what the chat itself is running**: the program, how
  long for, its address if it's a server, and how much it has logged
  (visits counted). A program that died is listed as exited, never
  as live.
- **Visits are the program's own words.** There is no proxy: handler
  visits land in the log because the program prints them — servers
  Gollemer creates print one `visit: GET /…` line per visit, and
  `show visits` / `show the program log` says exactly that for
  programs you wrote by hand.

### Basic terminal commands

The beginner terminal words work too — `pwd`, `ls`, `cd`, `cat`,
`echo`, `clear`, `date`, `whoami` — answered by the chat itself
(tagged `gollemer [terminal]>`), each behind the same run gate, and
each confined to the project folder: a path that climbs out of it is
refused by name. There is still no shell underneath: each command is
done natively, one at a time, so a line carrying a shell character
(`;`, `&`, `|`, `>`, `<`, a backtick, or `$`) is refused, naming the
character. Command words the chat won't run at all are answered with
guidance instead of silence: `mkdir` points at "create a folder
\<name\>", `rm` explains that Gollemer doesn't delete files, and
`mv`, `cp`, `sudo`, `curl` and friends point at the basics above.

`cd` moves the whole session, not just the answer: the chat re-reads
the new folder as its project, so creations, edits, and runs follow
you. Bare `cd` returns to the folder the chat started in, and
`cd ..` stops at the project root with a note instead of climbing
out of it.

```text
you> pwd
Run it? pwd [auto-yes: your 'do all' directive]
…/cap
you> ls
Run it? ls [auto-yes: your 'do all' directive]
data
go.mod
jake/
notes.txt
you> cd jake
Run it? cd jake [auto-yes: your 'do all' directive]
gollemer [terminal]> Now working in jake/ — creations, edits, and runs happen here now. Say "cd" on its own to go back to cap/.
you> rm notes.txt
gollemer [terminal]> I don't run rm — Gollemer doesn't delete files from the chat. Deleting is permanent, and I only do things I can show you first and stand behind; if a file has to go, that's a job for your file manager or your own terminal.
you> ls ../secret
gollemer [terminal]> "../secret" climbs out of the project folder — these commands only work inside cap/.
```

### Test everything at once

```
$ make handtest
...
handtest scoreboard: 102 passed, 0 failed, 102 total
```

`make handtest` drives the real chat through scripted sessions in a
throwaway project — folder creation, bare files, the webserver and
serve-an-html-file recipes, the security pieces, line placement, the
database pieces and the dependency gate, running your own programs
(a quick one, a live server with a real visit in its log, stop, and
top), the basic terminal commands (pwd, ls, cd, cat, echo, date,
whoami — and their refusals: rm, a path escape, a shell character,
cd .. at the top), the everyday statement pieces, and the one-ask
make-and-call — then checks the files it should have written and
builds the whole scratch project. It's the same test a person
would do by hand after every update, automated. (The scoreboard
above is from a real run on 2026-11-02.)

```
$ make paces
...
paces scoreboard: 63 passed, 0 failed, 63 total
```

`make paces` is the longer workout: one project built across 14
stages of real asks in sequence — a webserver, handlers, middleware
and security pieces, a named database connection, the dependency
gate with the install happening mid-conversation, a struct in one
file used from others (including a function written into a second
file that calls a third file's function), an html page served and
then fetched over HTTP from the running program, the terminal
basics, the guidance answers, a cross-file rename, the rm refusal,
and a function made in one file and called from another in a
single ask — with the disk checked after every stage and every
package built. It began as a one-off workout that scored 46 of 58 and found
three real defects; the fixes landed, and the workout stayed as a
test.

## How it works, in plain words

Gollemer is not one big model. It's **17 small specialists** behind
one prompt, plus a router that sends each sentence to the right one
and a scoreboard that remembers who answered and whether it worked
(ask it: `brains`, or `how are your brains doing`).

The specialists divide into two kinds:

- **Small trained brains** — for talking, recognizing what you mean,
  and answering questions from what they've learned: conversation, Go
  concepts, Go commands, make commands. Recognizing and chatting is
  what small models measurably do well.
- **Deterministic code** — ordinary programming, no model involved —
  for everything that must be *exact*: reading your project with Go's
  own parser, and writing every edit. An edit is assembled from the
  book of 135 known pieces and from your project's own code, shown to
  you as a plan with a diff, applied only after your yes, and checked
  by building the project. If the build fails, your file is put back
  exactly as it was.

One narrow exception, kept narrow on purpose: if you describe a new
type or function in words the book's grammar doesn't match ("add a
struct called Point with X as an int and Y as a string"), a small
model may *propose* the declaration. The plan says a model proposed
it, it's behind the same yes/no gate and the same build check — and
it needs to be: on wordings it never saw in training it gets the
whole declaration exactly right about 44% of the time. That's why it
proposes, and never writes on its own.

In short: **learned models recognize and converse; verified code
holds the pen.**

## Commands

Every command is a `make` target in this repo. Inside the chat you can
also just ask — `run make eval`, `what does make smarter do` — and it
will offer to run them for you.

| Command | What it does |
|---|---|
| `make start` | Open the chat — the whole product (needs a real terminal) |
| `make chat` | The same chat, started directly |
| `make guide` | Print the start-here guide — what to do, what it can do, how to teach it more |
| `make explain` | Print an overview like this README, from the command line |
| `make handtest` | Run the scripted end-to-end test suite in a throwaway project; prints a PASS/FAIL scoreboard |
| `make paces` | Run the 14-stage multi-file workout — one project built by real asks in sequence, disk + build checked per stage |
| `make sel` | Pick any make command from a fuzzy-finder list instead of typing it |
| `make debug-chat` | The chat with its thought process and routing shown — for the curious |
| `make eval` | Score every brain on its fixed test sets |
| `make smarter` | The one-command upgrade: more training data, retrained brains, evals, and a report |
| `make train-social` / `make train-go` / `make train-gocode` / `make train-gocli` / `make train-makefile` | Retrain one brain |
| `make import FILE=pairs.jsonl` | Teach it new Q&A pairs — each pair passes a quality gate; bad ones are quarantined, never trained on |
| `make flow PROMPT="create a worker pool"` | One sentence through the beginner pipeline: generated Go, a safety check, and a plain-words walkthrough |
| `make classic ARGS="-help"` | The older flag-driven pipeline tool the chat was built on |
| `make watch DIR=~/myproject` | Watch a folder while you code; notes the changes that matter when you come back |
| `make help` | List all of these |

## Honest limits

Gollemer will refuse some things, out loud and by name. That's a
feature: a small tool that knows its edges beats a big one that
guesses.

- **Methods on your own types.** "Add a `String` method to my
  `Point`" needs your type's whole world in the sentence. The book
  has no honest piece for it, so it refuses instead of inventing one.
  (Reading about methods is on the shelf.)
- **Generics.** Pieces that need type parameters — a generic helper,
  a generic table reader — are excluded for the same reason: the
  sentence can't carry the types.
- **Database drivers.** No database driver ships with Go, and which
  one you use is your choice. Database pieces carry a `TODO` where
  the driver's import goes; they compile as written, and the plan
  tells you the connection only proves itself once your driver is in.
- **Libraries it won't fetch.** If a piece needs a package your
  project doesn't have, Gollemer prints the exact `go get` line and
  waits for you to run it and say "installed". It never downloads or
  installs anything itself.
- **No shell.** The chat runs nothing but the gated make/go commands
  described above — targets from the project's own Makefile,
  validated, each behind a `[y/n]`. It cannot be talked into a shell
  command, and in a piped (non-terminal) session it won't run commands
  at all.
- **Placed, not auto-wired.** A handler added to `handlers/` is a
  complete, compiling piece — registering it with your server is a
  separate step you ask for (`connect the SignUp handler to the
  webserver`), never something it silently does to your `main`.
- **Whole programs from thin air.** It assembles known pieces and
  shapes from your own project. It won't invent a large program from
  a vague sentence — and on the one occasion a model gets to propose
  code, the proposal is labeled and gated, because it's often wrong.
- **One language, one reader.** It speaks Go. It's a companion for
  learning and for small, verified steps — not a general AI assistant.

## A note on the numbers in this README

135 pieces, 17 brains, 139 handtest cases, 66 paces checks — these are
counts of what's actually in this checkout, and the model scores
quoted are the measured ones, kept next to the features they
describe. When a number changes, this file should change with it.
