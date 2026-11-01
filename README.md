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

you> serve an html file to webtest/main.go
gollemer [composer]> Serve webtest/index.html from the server already in webtest/main.go: one registration, just before its ListenAndServe, answers http://localhost:8080/index.html with the page. The page webtest/index.html is created fresh with a starter page (its body is a TODO).
...
Apply this edit? [y/n] gollemer [composer]> Done — main.go serves index.html via http.ServeFile, at webtest/main.go:16. I built the project and checked that it's really there.
```

Ask twice and it tells you the truth instead of doing it again:

```
you> serve an html file to webtest/main.go
gollemer [composer]> Not doing that edit: webtest/main.go already serves webtest/index.html at http://localhost:8080/index.html — nothing to add.
```

### Add pieces to your code

Gollemer keeps a book of **135 pieces** of Go — small, known shapes
like a print line, a map lookup, a form handler, a JWT check. Ask for
one in plain words. If you don't name a file, web pieces go into the
folder where they belong, as their own little package:

```
you> add a form handler named SignUp
gollemer [composer]> Create the handlers folder and handlers/signup.go in it (package handlers) with the form handler SignUp, from the concept book.
SignUp is placed in the handlers folder as package handlers — placed, not wired: nothing registers the handler with a server yet. Wiring it in is a next step.
...
Apply this edit? [y/n] gollemer [composer]> Done — signup.go now declares the form handler SignUp in package handlers, at handlers/signup.go:7. I built the project and checked that it's really there.
```

Handlers go to `handlers/`, middleware to `middleware/`, security
pieces to `security/`, database pieces to `database/`. Name a file —
`add a cors middleware named Cors to app/main.go` — and the piece goes
exactly there instead.

The security pieces explain themselves as they land:

```
you> add a jwt token in main to app/main.go
gollemer [composer]> Add a jwt at the end of main in app/main.go, from the concept book: secret := os.Getenv("JWT_SECRET")
Anyone can READ a JWT's claims — the signature proves they weren't changed, it does not hide them — so a token never carries a password or a secret. And the signing secret here comes from the JWT_SECRET environment variable: a secret written in code is a secret published with the code.
...
Apply this edit? [y/n] gollemer [composer]> Done — main.go now has the jwt in main, at app/main.go:18. I built the project and checked that it's really there.
```

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
gollemer [composer]> That a gorqlite connection named Connect (it would be created as database/connect.go) needs the gorqlite client, and this project doesn't have it yet — the code would import github.com/rqlite/gorqlite, and nothing compiles against a dependency the project doesn't list. I don't fetch dependencies for you, so run this in the project first:
gollemer [composer]>   go get github.com/rqlite/gorqlite
gollemer [composer]> Once you've run it, tell me "installed" and I'll bring that a gorqlite connection named Connect (it would be created as database/connect.go) back as a plan for your okay.

you> installed
gollemer [composer]> github.com/rqlite/gorqlite is in go.mod now — here's that a gorqlite connection named Connect (it would be created as database/connect.go) again:
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

### Test everything at once

```
$ make handtest
...
handtest scoreboard: 45 passed, 0 failed, 45 total
```

`make handtest` drives the real chat through scripted sessions in a
throwaway project — folder creation, bare files, the webserver and
serve-an-html-file recipes, the security pieces, line placement, the
database pieces and the dependency gate, and the everyday statement
pieces — then checks the files it should have written and builds the
whole scratch project. It's the same test a person would do by hand
after every update, automated. (The scoreboard above is from a real
run on 2026-11-01.)

## Where its knowledge comes from

Gollemer doesn't learn by reading the internet while you chat — the
chat makes no external calls at all. Instead, good sources were read
*once*, by people, and turned into data the program ships with: pieces
for its book, and pointers on its reading shelf. Every source below is
really in there:

- **[Go by Example](https://gobyexample.com)** — the reference the
  book's first big expansion was checked against: most of its topics
  (slices, maps, goroutines, JSON, files, …) became pieces Gollemer
  can add, and its hardest-working links power the "learn this first"
  path.
- **[YourBasic](https://yourbasic.org/golang/)** — everyday basics:
  the string builder, environment variables, walking a folder tree
  became pieces; its famous "gotcha" articles became shelf reading.
- **[Callicoder](https://www.callicoder.com/golang-basic-types-operators-type-conversion/)** — a
  beginner tutorial series; its types-and-operators article is on the
  shelf, and most of its topics matched pieces already in the book.
- **[A Tour of Go](https://go.dev/tour/)** — the official tour. Its
  topics were mapped one by one: most were already pieces; the rest
  are named exclusions (methods and generics — see Honest limits).
- **[Go Web Examples](https://gowebexamples.com)** — the web pieces:
  routing, templates, form handlers, sessions-style handlers. (Where
  the site uses third-party libraries, Gollemer wrote the
  standard-library version or left the topic out and said why.)
- **[go-database-sql.org](http://go-database-sql.org/index.html)** —
  the guide to using SQL databases from Go: the database pieces —
  open, query, exec, prepared statements, transactions, and NULL
  values.
- **[SQLInterface](https://go.dev/wiki/SQLInterface)** (the Go Wiki) —
  shelf reading on how `database/sql` really works: that `sql.Open`
  doesn't connect, that a driver registers its name through a blank
  import. (The Wiki's Switch page is behind the type-switch piece.)
- **[Effective Go](https://go.dev/doc/effective_go)** — the style
  authority on the shelf, section by section: how idiomatic Go is
  written, linked from answers about formatting, naming, errors, and
  more.
- **[Devhints' Go cheatsheet](https://devhints.io/go)** — a one-page
  syntax reference, on the shelf; the two pieces the book was missing
  (the while-style loop, type conversions) became pieces.
- **[GoInterview cheatsheet](https://app.gointerview.dev/cheatsheet)**
  — an interview reference, on the shelf; its code-shaped topics
  mostly matched pieces already in the book.
- **[Programming Idioms](https://programming-idioms.org/cheatsheet/Go)**
  — a few hundred tiny tasks, each answered in Go. The
  statement-sized ones the book lacked became 19 pieces: the map and
  slice verbs, reading input, swapping two variables, summing a
  slice, the file verbs, and more.
- **[rqlite](https://rqlite.io)'s documentation** — the rqlite pieces
  (which talk to the database over plain HTTP, no library needed),
  the gorqlite client pieces behind the dependency gate above, and
  the printed guidance for running rqlite yourself.
- **[OWASP Cheat Sheets](https://cheatsheetseries.owasp.org/) and
  [MDN](https://developer.mozilla.org/)** — the reading behind the
  security pieces (JWT, CORS, IDOR, password storage). The pieces
  themselves are standard-library only.
- **Practical Go Lessons, DigitalOcean's *How To Code in Go*, and
  Astaxie's *Build Web Application with Golang*** — shelf reading,
  plus the form-values piece and struct-tag support.
- **The maintainer's own "Advocates" doc** — John's hand-collected
  list of Go links. When you say "i want to build X", the resource
  pack is drawn from this list.

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
- **Placed, not wired.** A handler added to `handlers/` is a complete,
  compiling piece — but registering it with your server is a next
  step the plan names, not something it silently does to your `main`.
- **Whole programs from thin air.** It assembles known pieces and
  shapes from your own project. It won't invent a large program from
  a vague sentence — and on the one occasion a model gets to propose
  code, the proposal is labeled and gated, because it's often wrong.
- **One language, one reader.** It speaks Go. It's a companion for
  learning and for small, verified steps — not a general AI assistant.

## A note on the numbers in this README

135 pieces, 17 brains, 45 handtest cases — these are counts of what's
actually in this checkout, and the model scores quoted are the
measured ones, kept next to the features they describe. When a number
changes, this file should change with it.
