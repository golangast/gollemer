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

There is no splash screen — the chat opens with one short line:

```
17 brains ready. Say "start" for the tour, or "what can you do" for everything I can do.
```

(If you ever want the old full-screen welcome back, launch with
`GOLLEMER_BANNER=full` in the environment.)

The chat works on the folder you started it in. Open it inside any Go
project and it will read *that* project. Brand new? Say `teach me` and
it walks you through the basics on a tiny bundled example.

## Things you can say

You talk in plain sentences. Every reply is tagged with the part of
Gollemer that answered, like `gollemer [go]>`, so you always know who
is talking.

Not sure what's on the menu? Ask `what can you do` in chat and it
lists what it can actually do — build, change, ask, remember, run —
each line with a real sentence that does it. The list is rendered
from the chat's own machinery and tested against it, so it never
promises a piece that isn't there.

*The exchanges below are real — captured from the chat on this code,
on 2026-11-01. Long diffs are trimmed with `...`.*

### Just talk, and ask about Go

```
you> have you ever played soccer
gollemer [social]> only for fun with friends. i am better at cheering than scoring.

you> what is a goroutine
gollemer [go]> A goroutine is a lightweight thread managed by the Go runtime: go doWork(). Thousands of goroutines cost little. They run concurrently with the caller — use channels or sync primitives to coordinate, never shared memory without synchronization.
```

Its built-in book answers over 250 Go topics — concepts, recipes,
and the pitfalls that bite — and how-to questions ("how do I serve
static files") get the same plain-words treatment.

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
remove exactly what their replies confirm. Facts about you are
the one thing it also picks up in passing: mention "my dog is
Bella", "I live in Omaha", or "my favorite color is forest
green" in ordinary conversation and it stores the fact — saying
so in one short line under its reply, never silently — and
later answers "what's my dog's name?" or "where do I live?"
from the store, exactly, or says plainly that you haven't told
it yet rather than guessing. Memory lives in
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
Create it? [y/n] gollemer [composer]> Done — created jane/.

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
Create it? [y/n] gollemer [composer]> Done — created webtest/main.go. It's the webserver starter and it builds — "run the program webtest/main.go" will start it.

you> serve home.html to webtest/main.go
gollemer [composer]> Serve webtest/home.html from the server already in webtest/main.go: one registration, just before its ListenAndServe, answers http://localhost:8080/home.html with the page. The page webtest/home.html is created fresh with a starter page (its body is a TODO).
...
Apply this edit? [y/n] gollemer [composer]> Done — webtest/main.go now serves webtest/home.html. I built the project and checked that it's really there.
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
Apply this edit? [y/n] gollemer [composer]> Done — The form handler SignUp is in handlers/signup.go. I built the project and checked that it's really there. It isn't wired to the server yet — "connect the SignUp handler to the webserver" will do it.
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
Apply this edit? [y/n] gollemer [composer]> Done — SignUp is in handlers/signup.go, registered at /signup. I built the project and checked that it's really there.
```

The handler answers at `/` + its name, or at the path you name
(`connect the Jake handler to the webserver at /contact`). A handler
in the server's own file wires by its bare name; one in another
package is imported and qualified, as above. If exactly one handler
is unconnected, `connect the handler to the webserver` finds it on
its own. Ask twice and it tells you the truth — already registered,
nothing to add — and a path another handler already owns is refused
with that handler named.

A goroutine can't hand a value back by returning — a `go` statement
throws results away — so `have the goroutine return a string` wires
Go's own answer instead: a channel declared just before the
goroutine, a send at the end of its body (the type's zero value for
now, under a TODO marking where the real result goes), and the
receive right after it, where the function waits for the value.

### When something's missing, it asks

If an ask is one piece of information short — which handler, which
file, what name — Gollemer doesn't stop at a paragraph of guidance
for you to decode and retype. It asks one short question, and your
next line completes the original ask, through the same plan and
the same `[y/n]` gate as if you'd typed it in full:

```
you> connect the handler to the webserver
gollemer [composer]> More than one handler isn't registered yet — which one should I wire in? Nothing's been written.
  1. Jake (webtest/main.go:5)
  2. Pete (webtest/main.go:6)
Answer with the name or the number — or say anything else and we'll move on.
you> Jake
gollemer [composer]> Wire Jake into the webserver in webtest/main.go: one line — http.HandleFunc("/jake", Jake) — added in main just before it serves, so http://localhost:8080/jake is answered by Jake (declared at webtest/main.go:5).
...
Apply this edit? [y/n] gollemer [composer]> Done — Jake is in webtest/main.go, registered at /jake. I built the project and checked that it's really there.
```

The same shape everywhere a choice or a missing piece is all
that's wrong: `add a database` asks whether you mean the
connection function or the inline piece, and the one word
`connection` (or `inline`) is a complete answer; a piece with no
file named asks which file — and offers the file you're working
on as a numbered suggestion; a declaration with no name asks what
to call it. Answer with a name, a number, a path, or the
distinguishing word — or just say something else entirely: the
question is forgotten and your line is handled as itself, exactly
as if no question had been asked. Facts are never questions,
though: a path another handler owns, a page that already exists, a
handler name already taken — those still refuse outright, question
or no question.

### Your names for things, and follow-ups

Paths are yours to name. Teach the chat a name once — "call
jim/main.go the api file", or "call this the api file" for the
file you just landed — and from then on the name works anywhere
a sentence names that file or folder. The reading is printed,
never silent: the chat says which path it understood before it
shows a plan. Your names are kept in `~/.gollemer_vocab.json`,
keyed by project, so a name taught in one project means nothing
in another — and they're a different file from your taught
pieces and your remembered facts on purpose, so forgetting one
kind of thing can never erase another. A real file with the
same name always wins over a taught name. "What do you call
things" lists them; "forget the api file" removes one.

And once a piece has landed, the session remembers it well
enough to repeat it: "another one" lands the same piece
again — asking for its name first, the same way a bare ask
does — "another one named X in <file>" and "the same but for
<file>" say the variations out loud, and "the other file"
means the other file this session has been working in. Every
repeat is rebuilt as the full sentence and goes through the
same plan, gate, and build check; with nothing landed yet, a
follow-up line is just a line, exactly as before.

```
you> call this the server file
gollemer [go]> Learned your name for it:
  "the server file" — vocabjim/main.go in this project.
you> add a form handler named Vex to the server file
gollemer [go]> Reading "the server file" as vocabjim/main.go — your name for it.
... the same plan, gate, and build check, then:
gollemer [go]> Done — The form handler Vex is in vocabjim/main.go. I built the project and checked that it's really there. It isn't wired to the server yet — "connect the Vex handler to the webserver" will do it.

you> another one
gollemer [go]> Another struct — what should I call it? Answer with one name — or say anything else and we'll move on.
you> SequelProbe
gollemer [go]> I read that as: "add a struct named SequelProbe to sequel/main.go".
... the same plan, gate, and build check, and the second struct lands.
```

### When something lands, it says what landed

`Done` still means what it always meant — the plan was applied,
the project builds, and the change was verified in the code. What
changed is how the landing is reported: not as a build log
("main.go now declares the form handler…, at main.go:19. I built
the project and checked…"), but the way a partner would confirm
it — what landed, and where, in plain words. The line is written
from the transaction's own record — the plan that actually ran
and the project as re-read after it — never from the sentence
you typed:

```
you> create a file partner/main.go with a webserver
...
Create it? [y/n] gollemer [composer]> Done — created partner/ and partner/main.go. It's the webserver starter and it builds — "run the program partner/main.go" will start it.

you> add a form handler named Petra to partner/main.go
...
Apply this edit? [y/n] gollemer [composer]> Done — The form handler Petra is in partner/main.go. I built the project and checked that it's really there. It isn't wired to the server yet — "connect the Petra handler to the webserver" will do it.

you> connect the Petra handler to the webserver
...
Apply this edit? [y/n] gollemer [composer]> Done — Petra is in partner/main.go, registered at /petra. I built the project and checked that it's really there.
```

When a landing has a natural next step, the confirmation names
the exact line that does it — a handler's wiring, a starter's
run line, the webserver a handler is still missing. It only
names a step the machinery has itself verified will land: the
wiring plan is checked against your project first, so a handler
whose server is a custom one gets no promised line at all,
rather than a line that would refuse. And when there is no
honest next step, the one sentence stands alone — nothing is
invented to fill the space. Refusals, `n` answers at the gate,
and edits that fail their build check answer exactly as they
always did; a confirmation is earned by a landing, never
printed instead of one.

### Pages in one ask

Some pages come in kinds, and a kind can be had whole — the page
file, the handler that serves it, and its registration, in one
plan. There are six: a **homepage**, an **about page**, a **contact
page**, an **item list page**, a **details page**, and a **form
page**. (Ask to `create a handler with a form` and you get the form
kind from the other side: the handler, plus the page with its
form.)

```
you> create a homepage in webtest
gollemer [composer]> Add the homepage: webtest/index.html (a starter page), its handler Home in webtest/main.go, and the starter registration at / gives way to http.HandleFunc("/", Home), so Home answers http://localhost:8080/.
The webserver's own starter handler at / — the one that answers "hello — your webserver is running" — is replaced by the registration above; nothing else in the server changes.
...
Apply this edit? [y/n] gollemer [composer]> Done — The homepage is in webtest/index.html — Home answers at /. I built the project and checked that it's really there.

you> create a contact page in webtest
gollemer [composer]> Add the contact page: webtest/contact.html (a starter page), its handler Contact in webtest/main.go, and one registration — http.HandleFunc("/contact", Contact) — added in webtest/main.go just before it serves, so Contact answers http://localhost:8080/contact.
The form in webtest/contact.html asks for exactly the fields Contact reads — email, subject and message — so the page and the handler can't drift apart.
...
Apply this edit? [y/n] gollemer [composer]> Done — The contact page is in webtest/contact.html — Contact answers at /contact. I built the project and checked that it's really there.
```

Each kind answers at its own address — the homepage at `/`, about
at `/about`, contact at `/contact`, the item list at `/items`,
details at `/details`, the form at `/form` — and the handler takes
the kind's name (Home, About, Contact, Items, Details, Form) unless
you give one: `create a form page named Signup`. The page lands
beside the handler's file, and a page file that already exists is
never overwritten — the ask refuses and names it.

If the program has no webserver yet, the page and handler still
land, and the plan names the finishing sentence — `connect the Home
handler to the webserver` — for when one exists. The starter pages
carry TODOs where your own words and data go: the item list opens
with placeholder items, the details page with placeholder fields.
Gollemer doesn't invent your content.

### A plain webpage, and serving its files

Not every page is one of the six kinds. `add a webpage to
jim/temp/index.html` puts a plain webpage into a named file, from
a template you pick — **basic** (a title, a heading, a paragraph),
**responsive** (the same, plus a viewport line and a little CSS
so the page fits a phone's screen too), **with assets** (wired
to load a style sheet, a script, and a picture from `/static/`),
or a layout: **two columns** (a sidebar beside the main area),
**three columns** (a sidebar on each side), **grid** (cards in a
grid that narrows with the screen — or counted exactly:
`add template 2 columns 3 rows to jim/temp/index.html`, or
`add the grid 2 columns 3 rows to jim/temp/index.html`, makes
a grid of precisely 2 columns by 3 rows of boxes, 1 to 6 of
each), or **rows** (full-width bands
in different shades) — or **material** (Google's Material Design
look, hand-written plain CSS in the page itself, no framework).
You can ask in the template's own words too —
`add a two column template to jim/temp/index.html`, or
`add the template to jim/temp/index.html` when you'd rather
answer the question it asks in return. It asks which template
whenever you don't name one. The file is
created when new, filled when it sits there empty — and a page
that already has words in it is never overwritten.

Once a page exists, you can add one Material component at a
time — the same parts the material template shows off: a
**button** (the filled one), a **tonal button**, an
**outlined button**, a **text button**, a **card**, an **app
bar** (or top bar), a **navigation bar** (or nav bar, navbar),
a **floating button** (or fab), a
**chip**, a **text field**, a **switch**, and a **footer**. Say it like
`add a button to jim/temp/index.html`, `add a text field to
jim/temp/index.html`, or `add a card to the page` ("the page"
is the webpage you last made or filled in this chat). The
component's styles come with it the first time and never
twice. Ask `what components can I add` for the whole set.

A page can also be built the way real sites are: a header and
a footer written once, and each page's own body between them.
`add a templated page called home in jim` creates
`jim/templates/header.html` (the head, the Material styles, and
the responsive navigation bar), `jim/templates/footer.html`
(the footer bar), and `jim/templates/home.html` (the body,
which calls the header at its top and the footer at its foot),
joined when the page is served by Go's html/template — plus
the handler that renders it (`HomePage` in `jim/homepage.go`).
No route is wired: the handler waits for the connect sentence
like any other. Run the server from the project root — that's
where the handler's template paths point. Then grow the body:
`add 2 columns and 3 rows to the body in jim` — or, in the
grid's own words, `add the grid 2 columns 3 rows to the body
in jim` — puts a counted
section of boxes into the body file alone (its styles join the
header's, once), and asking again adds another section below
the first. With no counts it makes 2 columns and 2 rows. The
counted grid aimed at the body file itself —
`add the grid 2 columns 3 rows to jim/templates/home.html` —
lands the same way: the sentence is one, and the file it
names decides which thing it is. A component grows the body
too: `add a card to jim/templates/home.html` puts the card
into the body above its footer call, its styles joining the
header's the same way — one of each component per body, and
asking for the same one again answers that it's already
there. The navigation bar and the footer need no adding:
every templated page already has them, from the header and
footer files. Ask for a whole template
on one of the three files instead —
`add the material template to jim/templates/home.html` —
and the chat explains the page rather than refusing
flatly: the body is one part of a page, not a page on its
own, the design lives in the header and already covers
the whole page, and sections and components are what the
body takes.

A page can also be a program: `add a wasm app in jim`
creates a browser app in that folder — Go compiled to
WebAssembly, building its page in the browser through
syscall/js. The app is its own module: `jim/go.mod` pins
**Go 1.27** (the project around it keeps its own Go
version, and the app's files stay out of its build),
because the app's vector unit uses Go 1.27's portable
**simd** package — the one that works on wasm targets.
The folder gains `jim/main.go` (the app: it sets the
garbage collector's GOGC knob low in code —
`runtime/debug.SetGCPercent(50)` — since a browser page
has no environment to pass flags through, then builds a
welcome heading and the vector demo's result into the
page), `jim/vector.go` (the simd unit), `jim/index.html`
(the host page, carrying the Material styles, with the
build in its opening comment:
`GOEXPERIMENT=simd GOOS=js GOARCH=wasm go build -o main.wasm .`,
run in the folder — a first build fetches the Go 1.27
toolchain by itself), and `jim/wasm_exec.js`, copied from
this machine's Go 1.27 install when one is found; when
none is, the plan says so in the open and names the one
file to copy by hand. Once `main.wasm` is built, any
static file server can serve the folder. Components join
the app as Go code, not HTML: `add a card to jim` (or
`add a card to jim/main.go`) puts the card's builder in
`jim/components.go` and mounts it on the page, its styles
joining the host page exactly once — the same Material
set, one of each kind per app. In an app, a component can
also carry a name — `add a button called save to jim` —
a second, third button beside the unnamed one, each with
its own builder; every component's root element carries
an id: its name when it has one, else its place among its
kind (`button-1`, `card-1`).

And the app can answer: an event is one sentence —
`when the save button is clicked, set the card's words to
'Saved'` (or `... in jim`), `when the switch is flipped,
toggle the card`. It lands as a Go handler in
`jim/events.go`: a `js.FuncOf` callback registered on the
component once the page is mounted, kept in the file's
list for the page's whole life — a callback Go lets go of
is garbage-collected and silently stops answering, so
none of them are let go. Buttons and cards answer to
clicks; a switch or a text field answers to being
changed. What can happen is show, hide, or toggle another
component, or set a button's or a card's words (theirs
are the components whose words are the whole of what they
say). With several of a kind mounted and no name in the
sentence, the chat asks which one — it never guesses.
Naming a folder works like the other app sentences: no
folder named means the last app this conversation worked
on. Plain pages stay script-free: events belong to wasm
apps alone.

And building the app is a sentence: `build the wasm app
in jim` runs the host page's build in the app's folder —
the exact command from its comment, behind the usual
yes — and answers with the compiler's own words when it
fails, or the new `main.wasm` and its size when it lands.
No folder named builds the last app the conversation
worked on, and building again after an edit is ordinary:
the sentence never refuses because `main.wasm` is
already there. One word does two duties — with no app in
the folder yet, "build a wasm app in jim" is the creation
ask above; the chat tells the two apart by what the
folder already holds, and only ever compiles an app that
is really there.

A page's style sheets, scripts, and pictures are files too, and
the server can serve the whole folder they live in: `serve static
files` registers that folder at `/static/` on your existing
server, so the folder's `style.css` answers at
`http://localhost:8080/static/style.css`. Name the folder in
the sentence when you have one in mind — `serve static files
from jim/temp`, or `serve the jim/temp folder`; with no folder
named, it finds the folder itself — the webpage's folder, or
the one folder in the project holding web files — and asks
when there's more than one it could mean. You can say it as an
update too: `update the server to serve images, js, css`. One
mechanism covers images, js, and css alike — the file server
picks each file's content type from its extension.

The security pieces explain themselves as they land:

```
you> add a jwt token in main to app/main.go
gollemer [composer]> Add a jwt at the end of main in app/main.go, from the concept book: secret := os.Getenv("JWT_SECRET")
Anyone can READ a JWT's claims — the signature proves they weren't changed, it does not hide them — so a token never carries a password or a secret. And the signing secret here comes from the JWT_SECRET environment variable: a secret written in code is a secret published with the code.
...
Apply this edit? [y/n] gollemer [composer]> Done — main.go now has the jwt in main. I built the project and checked that it's really there.
```

### Teach it a piece of your own

The book's pieces are the ones that ship with Gollemer. Yours can
join them: show the chat one declaration in your own project, give
its kind a name, and the chat keeps the shape — as your data, in
your home folder (`~/.gollemer_taught.json`, never inside a
project), working in every project you open:

```
you> learn this as a signup form from Jake in teachjim/main.go
gollemer [composer]> Here's what I learned from Jake in teachjim/main.go:
  The kind: "signup form" — from here on, "add a signup form named <Name> to <file>" assembles one.
  The example: the function Jake, at teachjim/main.go:11.
  What varies: the name — Jake becomes the new name everywhere the piece names itself, in its declaration and in its calls to itself — and where it lands.
  What stays fixed: everything else — the code stays exactly as in Jake, and the imports it leans on (net/http) travel with it.

you> add a signup form named Signup to teachsite/site.go
gollemer [composer]> Add the signup form Signup to teachsite/site.go — the piece you taught me, renamed from its example.
From your example Jake in teachjim/main.go, taught Nov 4, 2026.
...
Apply this edit? [y/n] gollemer [composer]> Done — The signup form Signup is in teachsite/site.go. I built the project and checked that it's really there.
```

What varies is the name — the declaration's own name and every
place the piece names itself (a constructor that returns it, a
recursive call, a field of its own type) — and where it lands.
Everything else stays exactly as the example has it: same fields,
same words, same code. The imports the example leans on travel
with it and are added to the target file if it lacks them. The
landing goes through the same plan, `[y/n]`, transaction and
build check as a book piece — if the new home can't build it, the
edit is restored whole and the compiler's words are the answer.

Ask `what have I taught you` for the list, `forget the signup
form` to remove one, `forget what I taught you` to remove them
all. Nothing is learned from ordinary conversation — only the
teach sentences teach. A taught kind can never take a book
kind's name: your pieces sit beside the book, never over it.
Version one teaches from Go functions and types only: a method
can't be taught (it belongs to its receiver's type), and there
is no pasting a snippet in — the example has to be a declaration
in the project, so the chat reads its real shape instead of
guessing one.

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
Apply this edit? [y/n] gollemer [composer]> Done — Validate now exists in api/validate.go and is called from api/report.go. I built the project and checked that it's really there.
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
Apply this edit? [y/n] gollemer [composer]> Done — The gorqlite connection Connect is in database/connect.go. I built the project and checked that it's really there.
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

Apply this edit? [y/n] gollemer [composer]> Done — main.go now has the print line in main. I built the project and checked that it's really there.
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

Or watch it while you work: `watch jim/main.go` starts the program
just like `run jim/main.go`, then keeps an eye on its folder — when a
file changes and the changes settle, the chat rebuilds and restarts
it, and says so in a bracketed note:
`[jim/main.go restarted — main.go changed]`. The same ask in other
words is `restart jim/main.go when files change`. Any file counts —
Go, html, js, css. A change that doesn't compile restarts nothing:
the note says the build failed, and the version that's running
stays up.

The watch covers the program's own folder and everything under it.
A folder elsewhere in the project joins by name: `watch folder
<path>` adds it as another watched folder, and a change there
restarts the program the same way, named by its full path
(`[jim/main.go restarted — assets/app.js changed]`). So
`watch folder jim/temp`, said while `jim/main.go` is watched,
answers that it's already watching — `jim/temp` sits inside `jim/`,
which the watch already covers — and naming the same folder twice is
the same answer. Folder watching belongs to the chat's program, but
naming a folder before any program runs is fine: the chat notes it,
`top` shows it among the folders waiting for a program, and it
attaches when watching starts. `stop watching folder jim/temp` removes just that folder;
`stop watching jim/main.go` (or bare `stop watching`) ends the
watching and leaves the program running, and `top` lists the extra
folders next to the program.

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
paces scoreboard: 83 passed, 0 failed, 83 total
```

`make paces` is the longer workout: one project built across 19
stages of real asks in sequence — a webserver, handlers, middleware
and security pieces, a named database connection, the dependency
gate with the install happening mid-conversation, a struct in one
file used from others (including a function written into a second
file that calls a third file's function), an html page served and
then fetched over HTTP from the running program, the terminal
basics, the guidance answers, a cross-file rename, the rm refusal,
a function made in one file and called from another in a
single ask, a clarifying question answered in one word, a
follow-up that repeats the last piece, and a taught name for a
file used across two chats — with the disk checked after every stage and every
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

135 pieces, 17 brains, 195 handtest cases, 83 paces checks — these are
counts of what's actually in this checkout, and the model scores
quoted are the measured ones, kept next to the features they
describe. When a number changes, this file should change with it.
