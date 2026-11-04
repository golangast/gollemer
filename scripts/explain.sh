#!/bin/bash
# make explain — what Gollemer is and what to say to it.
# This text tells the same story as README.md, condensed for the
# terminal. Keep the two in step: same counts (135 pieces,
# 17 brains), same examples, same limits.
cat << 'EOF'
Gollemer — a tiny language model in pure Go that helps you learn Go
by talking to it. One chat: it reads the Go project you're in,
explains it in plain English, answers Go questions, and makes small
edits — only after showing you the exact change and getting your
yes. No rented AI model, no account, no external calls: everything
runs on your machine.

WHY IT EXISTS
-------------
Most AI coding tools are enormous models on someone else's
computers. Gollemer is the opposite: small enough to read, built
for beginners, honest about its limits. It teaches instead of
taking over — it explains your project, shows where a change
belongs and why, and lets you say the words while it does the
typing. Where a small model isn't good enough to trust,
deterministic code (ordinary programming, no AI) does the job.

START
-----
  make start     open the chat (needs a real terminal)
  make chat      the same chat, started directly
  make guide     the start-here guide, printed, no chat

The chat works on the folder you started it in. New here? Say
"teach me" — a two-minute guided tour on a tiny bundled example.

WHAT TO SAY TO THE CHAT
-----------------------
Just talk in plain sentences. Every reply is tagged with the part
that answered, like gollemer [go]>, so you know who is talking.

Talk and ask about Go:

  you> have you ever played soccer
  gollemer [social]> only for fun with friends. i am better at
      cheering than scoring.
  you> what is a goroutine
  gollemer [go]> A goroutine is a lightweight thread managed by
      the Go runtime: go doWork(). ...

  Questions and conversation no specialist brain owns land
  with the talk seat, tagged gollemer [talk]>: it talks about
  your project from the analyzer's index (what a function
  takes, what's in a file, what the piece you just added
  does — "what does it do?" names what "it" was read as) and
  holds a conversation with the local social model. No keys,
  no network; project names in a talk reply are checked
  against your code before they print, and a name that isn't
  there is marked, never presented as fact. It explains and
  discusses only — ask it to DO something and it points you
  to the exact sentence; doing still goes through the gate.
  It answers state questions from the chat itself (the time,
  this machine, the branch in .git/HEAD, Go files changed
  since its last read — a read-baseline check, not a full
  git status) and remembers only what you explicitly tell
  it: "remember: ..." and "note: ..." are echoed as stored,
  "what do you remember" shows them, and the forget shapes
  remove exactly what their replies confirm. Memory lives
  in ~/.gollemer_memory.json, outside every project, and a
  returning session opens with one "Last time here:" line.

Ask what a project does — it answers from your real code:

  you> analyze this project
      IN PLAIN ENGLISH: what the program is, in plain words
      STORY: what happens when you run it
      WHERE TO START: the entry point, the engine room, the types
      ... then visuals: what calls what, biggest files, coupling
  you> what does Add do / show me Add / walk me through Add
  you> where for <github issue link>   where the fix belongs
  (or just paste a Go compiler error — it translates it)

Learn how to build something:

  you> i want to build a webserver
      a pack of reading from the maintainer's own Advocates link
      collection, plus the shelf of canonical Go texts
  you> resources for databases

Create folders and files — shown first, gated, then verified:

  you> create a folder jane
  gollemer [composer]> Here's what I'll create — nothing is
      written yet: ◆ CREATE folder jane/
      Create it? [y/n] ... Done — created jane/. I checked.
  you> create jane/main.go        a starter program file
  you> create a file webtest/main.go with a webserver
  you> serve an html file to webtest/main.go
      creates the page and wires the server to serve it; ask
      again and it says "already serves ... — nothing to add"
  you> connect the SignUp handler to the webserver
      registers a handler the book placed: one gated line —
      http.HandleFunc("/signup", handlers.Signup) — just before
      the serve call; ask again and it says "already
      registered ... — nothing to add"
  you> create a homepage
      a whole page in one ask: the starter page (index.html),
      its handler, and its registration at / — the webserver's
      starter reply gives way to it. The same shape for an
      about page (/about), a contact page (/contact, with the
      form built in), an item list page (/items), a details
      page (/details), a form page (/form) — and "create a
      handler with a form" asks for the form kind from the
      handler side. A form in a page asks for exactly the
      fields its handler reads, so the two can't drift apart.
      With no webserver yet, the page and handler still land,
      and the plan names the connect sentence that finishes
      the wiring. A page that already exists is never
      overwritten.

Add pieces from the book of 135 known Go shapes:

  you> add a form handler named SignUp
      no file named -> handlers/signup.go, its own package:
      "placed, not wired" — the connect sentence above is how
      you register it with your server
  you> add a cors middleware named Cors to app/main.go
      a named file always wins over the folder default
  you> add a jwt token in main to app/main.go
      the security pieces (JWT, CORS, IDOR check, CSRF, rate
      limit, ...) explain themselves as they land
  you> add a database connection function named Connect
      database/connect.go: sql.Open + a Ping + pool settings.
      The driver is a TODO on purpose — no driver ships with Go,
      and which one is yours to choose; until you add its blank
      import the Ping fails with "unknown driver", and the plan
      says so up front.

Make a function in one file and call it from another, in one ask:

  you> make a function named Validate in api/validate.go and add a function named Report in api/report.go that calls it
      ONE plan, ONE [y/n]: the function lands in the first file
      (created if its folder exists), the caller is assembled
      from the declaration that same plan writes, and a build
      failure restores BOTH files. "... and call it in
      api/main.go" puts the call at the end of main instead;
      no main to hold it -> a refusal spelling both working
      forms, never an invented caller. The new function's body
      stays the book's TODO scaffold — the plan says so.

Pieces that need a library your project doesn't have are gated:

  you> add a gorqlite connection named Connect
  gollemer [composer]> ... this project doesn't have it yet.
      I don't fetch dependencies for you, so run this first:
        go get github.com/rqlite/gorqlite
      Once you've run it, tell me "installed" ...
  you> installed
      it re-checks go.mod, then re-offers the original plan
      behind the same [y/n] gate. Say "installed" early and it
      tells you plainly it isn't there yet, and keeps waiting.

Put a piece on an exact line:

  you> add a print line in main to app/main.go after line 9
      the plan quotes line 9's actual text; pieces land only
      between whole statements; a bad line number gets a refusal
      that explains why — never a silent guess

Run the project's commands:

  you> run make help
  gollemer [makefile]> run make help
  [run it here? 'make help'] [y/n]: y
  [running: make help — Ctrl-C to interrupt] ... [done]

  Only targets from the project's own Makefile can run, each
  behind [y/n]. There is no shell. In a piped session the chat
  won't run commands at all.

Run your program:

  you> run the program jake/main.go
  Run it? go run jake/main.go [auto-yes: your 'do all' directive]
  gollemer [runner]> jake/main.go is running in the background — serving at http://localhost:8080. Say "show visits" to read its log (handler visits included), or "stop the program" to stop it.
  gollemer [runner]> Opening http://localhost:8080 in your browser.
  you> stop the program
  gollemer [runner]> Stopped jake/main.go — it's no longer running.

  "run the program <file>.go" runs YOUR program — never the
  project's make start. It builds first (compile errors are
  reported as compile errors, in the compiler's own words),
  runs behind the same [y/n] gate, and a file with sibling
  .go files runs as its whole package. A quick program is
  reported with its real exit state and its own output; a
  server stays running in the background while you keep
  chatting, its output (visits included — servers Gollemer
  creates print one line per visit) kept for "show visits" /
  "show the program log". "top" lists what the chat itself is
  running, honestly scoped. "stop the program" / "stop it" /
  bare "stop" while one runs stop the program and keep the
  chat open; closing the chat stops it too — no orphans. If
  the browser can't open, the address is printed plainly.

Basic terminal commands:

  you> pwd
  Run it? pwd [auto-yes: your 'do all' directive]
  …/cap
  you> cd jake
  gollemer [terminal]> Now working in jake/ — creations, edits,
      and runs happen here now.
  you> rm notes.txt
  gollemer [terminal]> I don't run rm — Gollemer doesn't delete
      files from the chat. …

  pwd, ls, cd, cat, echo, clear, date, whoami — each done natively
  by the chat (no shell, ever), each behind the same Run it? gate,
  and each confined to the project folder: a path that climbs out
  is refused by name, and a line carrying a shell character
  (; & | > < ` $) is refused, naming the character. Words that
  aren't run at all — rm, mv, cp, touch, rmdir, chmod, sudo,
  curl, wget — get guidance instead: mkdir points at
  "create a folder <name>", rm explains that Gollemer doesn't
  delete files. cd re-roots the session — the analyzer follows,
  so creations and edits move with you; bare cd returns to the
  folder the chat started in, and cd .. stops at the project
  root with a note.

Test everything at once:

  make handtest
      drives the real chat through scripted sessions in a
      throwaway project — folders, bare files, webserver +
      serve-html, the page kinds, security, line placement,
      database + the
      gorqlite gate, program running, the basic terminal
      commands with their refusals, the statement pieces,
      the one-ask make-and-call —
      checks the files, builds the scratch project, prints a
      PASS/FAIL scoreboard (102 passed, 0 failed on 2026-11-02).

  make paces
      the longer workout: one project built across 14 stages
      of real asks in sequence — webserver, handlers,
      middleware + security, a named database connection, the
      gorqlite gate with the install mid-conversation, a
      struct used across files, a cross-file write, a page
      served and fetched over HTTP, the terminal basics,
      guidance, a cross-file rename, the rm refusal, and the
      one-ask make-and-call — disk checked after every stage,
      every package built
      (63 passed, 0 failed on 2026-11-02). It began as a
      one-off workout that found three real defects; the
      fixes landed and the workout stayed as a test.

HONEST LIMITS
-------------
  Methods on your own types — refused by name: a method needs
  your type's whole world in the sentence. (Reading is shelved.)
  Generics — pieces needing type parameters are excluded: the
  sentence can't carry the types.
  Database drivers — never written; the TODO marks the spot and
  you pick the driver.
  Dependencies — never fetched; it prints the go get line and
  waits for your "installed".
  No shell — only gated make/go commands, from the project's
  own Makefile allowlist.
  Placed, not auto-wired — a handler in handlers/ compiles,
  and registering it is its own gated ask ("connect the
  <Name> handler to the webserver"), never a silent extra.
  Whole programs from thin air — it assembles known pieces; it
  won't invent a large program from a vague sentence.

COMMANDS
--------
  make start         open the chat — the whole product
  make chat          the same chat, started directly
  make guide         the start-here guide
  make explain       this overview
  make handtest      scripted end-to-end tests, PASS/FAIL out
  make paces         the 14-stage multi-file workout, PASS/FAIL out
  make sel           pick a command from a fuzzy finder
  make debug-chat    the chat with its thought process shown
  make eval          score every brain on its fixed eval suite
  make smarter       the one-command upgrade: more data,
                     retrained brains, evals, and a report
  make train-social  retrain one brain (also: train-go,
                     train-gocode, train-gocli, train-makefile)
  make import        teach it Q&A pairs through the quality
                     gate (FILE=pairs.jsonl) — bad pairs are
                     quarantined, never trained on
  make flow          one prompt through the beginner pipeline:
                     PROMPT="create a worker pool"
  make classic       the older flag-driven pipeline CLI
                     (ARGS="-help")
  make watch         watch a folder while you code (DIR=...)
  make help          list the commands

Ask inside the chat and it will offer to run these for you:
  you> list the commands
  gollemer [makefile]> run make help
  [run it here? 'make help'] [y/n]: y

THE ENGINE ROOM (for the curious)
---------------------------------
  pkg/ holds the libraries behind the chat (ast, analysis,
  memory, runner, synthesis, engine, xray, beginner):
    go test ./pkg/...
  internal/ai/analyze/ reads Go code with go/ast — summaries,
  the story, where-to-change answers.
  internal/ai/training/chat/ is the chat itself: the router,
  the brains, the composer, the book (data/concepts.json).

How it learns more: new Q&A pairs pass a coherence quality
gate, merge into data/training/chat_pairs.jsonl, and the brains
retrain on old + new (make smarter). Ask it "what is gollemer"
or "how do i teach you new things" in the chat.
EOF
