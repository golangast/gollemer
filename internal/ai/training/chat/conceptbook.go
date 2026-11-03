package chat

// The concept book + composer: the write family's answer to
// grammar-per-kind sprawl. Concepts — the basic pieces of Go a
// beginner asks to add, by NAME — live as trained DATA
// (data/concepts.json in the repo, loaded like the Advocates
// snapshot; the planner and row format are
// internal/ai/analyze/editconcept.go), and ONE composed grammar
// here matches a sentence against the loaded book, binds its
// slots, and hands the result to the PlanEdit spine. Adding a
// concept is adding a row: the day a "for loop" row lands in the
// book, "add a for loop to main.go" works with no new Go code
// (proven by the fixture-book test). The bespoke shapes keep
// their own sentences — this claim sits after them and before
// the write-sentence boundary, so known concepts are served and
// unknown ones still land on the boundary's honest answer.
// Neural generation stays out of it: the book is recall data,
// assembled deterministically, like the makefile brain's pairs.
//
// Grammar (slots may flex in the family's order):
//
//	add|put [a|an|the] <concept name from the book, longest
//	  match> [named <Name> | name is <Name>] [saying "<text>"]
//	  [like <Type>] [in <Func>] to|in|into [the] [file|folder]
//	  <target>
//
// A line phrase — "on line N" | "at line N" | "after line N"
// | "before line N", N a positive integer — may ride the slot
// zone (before the target clause, in any position among the
// other slots) or the tail behind the target. The concept
// name is consumed before either is read, so the "print line"
// in "add a print line on line 12 in main to jim/main.go" is
// the concept and the phrase is the placement, never garbage.
// A started phrase with a broken number (missing, zero,
// spelled out) still claims: runConcept refuses it in
// placement terms.
//
// "saying" may also be spelled "that says". The "in <Func>"
// qualifier is read only when a separate target clause follows
// it — a lone "in <word>" is the target, exactly as the
// add-server grammar reads it. A concept matched with no target
// (or a declaration with no name) is still claimed: the composer
// answers with the guidance naming the working sentence, in the
// nameless-designator tradition. Unknown concept names are NOT
// claimed here at all — they fall to the boundary.
//
// A name belongs to a declaration: a STATEMENT matched with a
// name slot never drops the name. When the book carries the
// statement's declaration sibling — a declaration named with
// one of the statement's names plus " function", the way
// "database connection function" stands beside the statement's
// "database connection" — the ask resolves to that declaration
// whole. With no sibling, the ask refuses with the guidance
// spelling the same ask unnamed, inside a function.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// defaultConceptBook mirrors data/concepts.json: the snapshot in
// the repo is the editable source, and this compiled copy is
// what a chat started outside the repo (where no data/ file
// exists) falls back to. A test pins the two to the same rows so
// they cannot drift.
const defaultConceptBook = `{
  "concepts": [
    {
      "id": "print-line",
      "imports": [
        "fmt"
      ],
      "kind": "statement",
      "names": [
        "print line",
        "print statement",
        "println",
        "print"
      ],
      "template": "fmt.Println(\"{text}\")"
    },
    {
      "id": "variable",
      "kind": "statement",
      "names": [
        "variable",
        "new variable",
        "var"
      ],
      "template": "value := 0\n// TODO: rename value and pick the value — the line below just keeps the build green until you use it\n_ = value"
    },
    {
      "id": "constant",
      "kind": "statement",
      "names": [
        "constant",
        "const"
      ],
      "template": "const limit = 10\n// TODO: rename limit and pick its value — a constant never changes while the program runs"
    },
    {
      "id": "for-loop",
      "kind": "statement",
      "names": [
        "for loop",
        "loop",
        "for"
      ],
      "template": "for i := 0; i < 3; i++ {\n\t// TODO: what the loop does is yours to write\n}"
    },
    {
      "id": "range-loop",
      "kind": "statement",
      "names": [
        "range loop",
        "for range",
        "range"
      ],
      "template": "items := []string{}\nfor index, value := range items {\n\t// TODO: each pass hands you the next index and value — use them here\n\t_ = index\n\t_ = value\n}"
    },
    {
      "id": "if-else",
      "kind": "statement",
      "names": [
        "if statement",
        "if else",
        "if"
      ],
      "template": "if true {\n\t// TODO: replace true with your condition, and write what happens when it holds\n} else {\n\t// TODO: what happens when the condition doesn't hold\n}"
    },
    {
      "id": "switch",
      "kind": "statement",
      "names": [
        "switch statement",
        "switch"
      ],
      "template": "switch {\ncase true:\n\t// TODO: your case — replace true with the condition that picks this branch\ndefault:\n\t// TODO: what happens when no case matches\n}"
    },
    {
      "id": "array",
      "kind": "statement",
      "names": [
        "array"
      ],
      "template": "numbers := [3]int{1, 2, 3}\n// TODO: your own size and values — an array's size is fixed and part of its type\n_ = numbers"
    },
    {
      "id": "slice",
      "kind": "statement",
      "names": [
        "slice"
      ],
      "template": "names := []string{\"ann\", \"bo\"}\n// TODO: your own values — a slice grows and shrinks, unlike an array\n_ = names"
    },
    {
      "id": "map",
      "kind": "statement",
      "names": [
        "map",
        "dictionary"
      ],
      "template": "ages := map[string]int{\"ann\": 3}\n// TODO: your own keys and values — a map looks a value up by its key\n_ = ages"
    },
    {
      "id": "closure",
      "kind": "statement",
      "names": [
        "anonymous function",
        "closure",
        "function value"
      ],
      "template": "greet := func() {\n\t// TODO: what the closure does — it can use the variables around it\n}\n_ = greet"
    },
    {
      "id": "pointers",
      "kind": "statement",
      "names": [
        "pointer",
        "pointers"
      ],
      "template": "value := 42\npointer := &value\n// TODO: pointer holds value's address — *pointer reads and writes value itself\n_ = pointer"
    },
    {
      "id": "strings-runes",
      "kind": "statement",
      "names": [
        "string",
        "strings and runes",
        "rune"
      ],
      "template": "text := \"hello\"\n// TODO: a string is text; ranging over it walks its characters — its runes — one by one\n_ = text"
    },
    {
      "id": "errors",
      "imports": [
        "errors"
      ],
      "kind": "statement",
      "names": [
        "error",
        "new error",
        "errors"
      ],
      "template": "err := errors.New(\"something went wrong\")\n// TODO: your own message — an error is a value you return or handle, not a crash\n_ = err"
    },
    {
      "id": "goroutine",
      "kind": "statement",
      "names": [
        "goroutine",
        "go routine"
      ],
      "template": "go func() {\n\t// TODO: the work to run in the background — it starts now and runs alongside the rest\n}()"
    },
    {
      "id": "channel",
      "kind": "statement",
      "names": [
        "channel"
      ],
      "template": "messages := make(chan string)\n// TODO: one goroutine sends with messages <- \"hi\" and another receives with text := <-messages\n_ = messages"
    },
    {
      "id": "buffered-channel",
      "kind": "statement",
      "names": [
        "buffered channel"
      ],
      "template": "messages := make(chan string, 2)\n// TODO: the 2 is how many values can wait in the channel before a sender has to block\n_ = messages"
    },
    {
      "id": "channel-sync",
      "kind": "statement",
      "names": [
        "channel synchronization",
        "channel sync",
        "wait for a goroutine"
      ],
      "template": "done := make(chan bool, 1)\ngo func() {\n\t// TODO: the work to wait for goes here\n\tdone <- true\n}()\n<-done"
    },
    {
      "id": "select",
      "kind": "statement",
      "names": [
        "select statement",
        "select",
        "non-blocking channel"
      ],
      "template": "messages := make(chan string, 1)\nmessages <- \"ping\"\nselect {\ncase received := <-messages:\n\t_ = received\n\t// TODO: act on the message that arrived\ndefault:\n\t// TODO: what to do when no channel is ready — this branch never waits\n}"
    },
    {
      "id": "timeout",
      "imports": [
        "time"
      ],
      "kind": "statement",
      "names": [
        "timeout"
      ],
      "template": "result := make(chan string, 1)\ngo func() {\n\t// TODO: the slow work goes here — send its answer on result when it's done\n\tresult <- \"done\"\n}()\nselect {\ncase answer := <-result:\n\t_ = answer\ncase <-time.After(2 * time.Second):\n\t// TODO: two seconds passed with no answer — decide what the wait means\n}"
    },
    {
      "id": "close-channel",
      "kind": "statement",
      "names": [
        "close channel",
        "closing a channel",
        "close a channel"
      ],
      "template": "jobs := make(chan int, 2)\njobs <- 1\nclose(jobs)\n// TODO: closing tells receivers no more values are coming — they can still take what's already in"
    },
    {
      "id": "channel-range",
      "kind": "statement",
      "names": [
        "channel range",
        "range over a channel",
        "range over channel",
        "loop over a channel"
      ],
      "template": "queue := make(chan string, 2)\nqueue <- \"one\"\nqueue <- \"two\"\nclose(queue)\nfor item := range queue {\n\t_ = item\n\t// TODO: what to do with each value — the loop ends when the closed channel is drained\n}"
    },
    {
      "id": "timer",
      "imports": [
        "time"
      ],
      "kind": "statement",
      "names": [
        "timer"
      ],
      "template": "timer := time.NewTimer(2 * time.Second)\n<-timer.C\n// TODO: the timer has fired — what happens now is yours to write"
    },
    {
      "id": "ticker",
      "imports": [
        "time"
      ],
      "kind": "statement",
      "names": [
        "ticker"
      ],
      "template": "ticker := time.NewTicker(500 * time.Millisecond)\n// TODO: range over ticker.C to repeat something on every tick — and call ticker.Stop() when you're done\n_ = ticker"
    },
    {
      "id": "waitgroup",
      "imports": [
        "sync"
      ],
      "kind": "statement",
      "names": [
        "wait group",
        "waitgroup"
      ],
      "template": "var wg sync.WaitGroup\nwg.Add(1)\ngo func() {\n\tdefer wg.Done()\n\t// TODO: one piece of work the group waits for\n}()\nwg.Wait()"
    },
    {
      "id": "atomic-counter",
      "imports": [
        "sync/atomic"
      ],
      "kind": "statement",
      "names": [
        "atomic counter",
        "atomic"
      ],
      "template": "var count atomic.Uint64\ncount.Add(1)\n// TODO: Add and Load are safe to call from many goroutines at once — no lock needed\n_ = count.Load()"
    },
    {
      "id": "mutex",
      "imports": [
        "sync"
      ],
      "kind": "statement",
      "names": [
        "mutex"
      ],
      "template": "var mu sync.Mutex\nmu.Lock()\n// TODO: the shared thing only one goroutine at a time may touch goes here\nmu.Unlock()"
    },
    {
      "id": "sorting",
      "imports": [
        "sort"
      ],
      "kind": "statement",
      "names": [
        "sorting",
        "sort"
      ],
      "template": "numbers := []int{3, 1, 2}\nsort.Ints(numbers)\n// TODO: your own slice — sort.Strings and sort.Float64s sort the same way"
    },
    {
      "id": "sorting-by",
      "imports": [
        "sort"
      ],
      "kind": "statement",
      "names": [
        "custom sort",
        "sort by function",
        "sort by"
      ],
      "template": "names := []string{\"cy\", \"ann\", \"bo\"}\nsort.Slice(names, func(i, j int) bool {\n\treturn len(names[i]) < len(names[j])\n})\n// TODO: the function settles the order — this one sorts shortest first"
    },
    {
      "id": "defer",
      "imports": [
        "fmt"
      ],
      "kind": "statement",
      "names": [
        "defer",
        "deferred call"
      ],
      "template": "defer fmt.Println(\"TODO: a deferred call runs when the function around it returns — this one prints last\")"
    },
    {
      "id": "recover",
      "kind": "statement",
      "names": [
        "recover",
        "panic recovery"
      ],
      "template": "defer func() {\n\tif recovered := recover(); recovered != nil {\n\t\t_ = recovered\n\t\t// TODO: a panic was stopped here — decide what to do about it\n\t}\n}()"
    },
    {
      "id": "string-functions",
      "imports": [
        "strings"
      ],
      "kind": "statement",
      "names": [
        "string functions"
      ],
      "template": "text := \"hello\"\nshouted := strings.ToUpper(text)\n// TODO: the strings package holds Contains, Split, Join, Replace and friends\n_ = shouted"
    },
    {
      "id": "string-format",
      "imports": [
        "fmt"
      ],
      "kind": "statement",
      "names": [
        "string formatting",
        "format a string",
        "sprintf"
      ],
      "template": "text := fmt.Sprintf(\"TODO: %s\", \"your words\")\n// TODO: %s stands for a string, %d for a number, %v for almost anything — the layout is yours\n_ = text"
    },
    {
      "id": "text-template",
      "imports": [
        "text/template"
      ],
      "kind": "statement",
      "names": [
        "text template",
        "text templates"
      ],
      "template": "tmpl := template.Must(template.New(\"greeting\").Parse(\"Hello, {{.}}!\"))\n// TODO: Execute writes the template out with your data standing where {{.}} is\n_ = tmpl"
    },
    {
      "id": "regexp",
      "imports": [
        "regexp"
      ],
      "kind": "statement",
      "names": [
        "regular expression",
        "regex",
        "regexp"
      ],
      "template": "pattern := regexp.MustCompile(\"TODO: your pattern\")\n// TODO: pattern.MatchString(text) answers true or false; FindString picks the match out\n_ = pattern"
    },
    {
      "id": "json",
      "imports": [
        "encoding/json"
      ],
      "kind": "statement",
      "names": [
        "json",
        "marshal to json",
        "json encoding"
      ],
      "template": "blob, err := json.Marshal(\"TODO: the value to turn into JSON\")\nif err == nil {\n\t_ = blob\n}"
    },
    {
      "id": "xml",
      "imports": [
        "encoding/xml"
      ],
      "kind": "statement",
      "names": [
        "xml",
        "marshal to xml"
      ],
      "template": "blob, err := xml.Marshal(\"TODO: the value to turn into XML\")\nif err == nil {\n\t_ = blob\n}"
    },
    {
      "id": "time-now",
      "imports": [
        "time"
      ],
      "kind": "statement",
      "names": [
        "current time",
        "time",
        "now"
      ],
      "template": "now := time.Now()\n// TODO: now holds the current date and time — Format it, or measure how far apart two times are\n_ = now"
    },
    {
      "id": "epoch",
      "imports": [
        "time"
      ],
      "kind": "statement",
      "names": [
        "epoch",
        "unix time"
      ],
      "template": "seconds := time.Now().Unix()\n// TODO: seconds counts the time since January 1, 1970 — the epoch\n_ = seconds"
    },
    {
      "id": "time-format",
      "imports": [
        "time"
      ],
      "kind": "statement",
      "names": [
        "time formatting",
        "format a time"
      ],
      "template": "stamp := time.Now().Format(\"2006-01-02 15:04:05\")\n// TODO: the layout is written in Go's reference time — swap the pieces around, but keep the values; time.Parse reads it back\n_ = stamp"
    },
    {
      "id": "random",
      "imports": [
        "math/rand"
      ],
      "kind": "statement",
      "names": [
        "random number",
        "random numbers",
        "random"
      ],
      "template": "roll := rand.Intn(100)\n// TODO: a random whole number from 0 up to (not including) 100 — pick your own range\n_ = roll"
    },
    {
      "id": "number-parsing",
      "imports": [
        "strconv"
      ],
      "kind": "statement",
      "names": [
        "number parsing",
        "parse numbers"
      ],
      "template": "number, err := strconv.ParseFloat(\"3.14\", 64)\nif err == nil {\n\t_ = number\n}\n// TODO: ParseInt, ParseFloat and Atoi turn text into numbers — err is set when the text isn't one"
    },
    {
      "id": "url-parsing",
      "imports": [
        "net/url"
      ],
      "kind": "statement",
      "names": [
        "url parsing",
        "parse a url"
      ],
      "template": "address, err := url.Parse(\"https://example.com/TODO\")\nif err == nil {\n\t_ = address\n}"
    },
    {
      "id": "sha256",
      "imports": [
        "crypto/sha256"
      ],
      "kind": "statement",
      "names": [
        "sha256 hash",
        "sha256",
        "hash"
      ],
      "template": "sum := sha256.Sum256([]byte(\"TODO: the text to hash\"))\n// TODO: sum is the text's fingerprint — the same text always gives the same 32 bytes\n_ = sum"
    },
    {
      "id": "base64",
      "imports": [
        "encoding/base64"
      ],
      "kind": "statement",
      "names": [
        "base64 encoding",
        "base64"
      ],
      "template": "encoded := base64.StdEncoding.EncodeToString([]byte(\"TODO: your text\"))\n// TODO: base64 turns bytes into text that is safe to send almost anywhere\n_ = encoded"
    },
    {
      "id": "read-file",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "file reading",
        "read file",
        "reading a file"
      ],
      "template": "contents, err := os.ReadFile(\"TODO: the file's path\")\nif err == nil {\n\t_ = contents\n}"
    },
    {
      "id": "write-file",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "file writing",
        "write file",
        "writing a file"
      ],
      "template": "data := []byte(\"TODO: what the file should say\")\nif err := os.WriteFile(\"TODO: the file's path\", data, 0644); err != nil {\n\t// TODO: the write failed — decide what to do about it\n}"
    },
    {
      "id": "file-paths",
      "imports": [
        "path/filepath"
      ],
      "kind": "statement",
      "names": [
        "file path",
        "file paths",
        "join paths"
      ],
      "template": "path := filepath.Join(\"TODO\", \"file.txt\")\n// TODO: Join builds a path with the right slashes for the computer it runs on\n_ = path"
    },
    {
      "id": "directories",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "list a directory",
        "read a directory",
        "list files"
      ],
      "template": "entries, err := os.ReadDir(\"TODO: the folder's path\")\nif err == nil {\n\t_ = entries\n}"
    },
    {
      "id": "temp-file",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "temporary file",
        "temp file"
      ],
      "template": "file, err := os.CreateTemp(\"\", \"TODO-*\")\nif err == nil {\n\t_ = file\n}"
    },
    {
      "id": "log-line",
      "imports": [
        "log"
      ],
      "kind": "statement",
      "names": [
        "log line",
        "log message",
        "logging"
      ],
      "template": "log.Println(\"TODO: the message to log\")"
    },
    {
      "id": "http-client",
      "imports": [
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "http client",
        "http get"
      ],
      "template": "response, err := http.Get(\"https://example.com/TODO\")\nif err == nil {\n\t_ = response\n\t// TODO: read response.Body — and close it when you're done\n}"
    },
    {
      "id": "context",
      "imports": [
        "context"
      ],
      "kind": "statement",
      "names": [
        "context"
      ],
      "template": "ctx := context.Background()\n// TODO: a context carries a deadline, or a cancellation, into the calls that accept one\n_ = ctx"
    },
    {
      "id": "spawn-process",
      "imports": [
        "os/exec"
      ],
      "kind": "statement",
      "names": [
        "spawn a process",
        "run a program",
        "external command"
      ],
      "template": "output, err := exec.Command(\"TODO: the program to run\").Output()\nif err == nil {\n\t_ = output\n}"
    },
    {
      "id": "interface",
      "kind": "declaration",
      "names": [
        "interface"
      ],
      "template": "type {name} interface {\n{methods}\n}"
    },
    {
      "id": "struct",
      "kind": "declaration",
      "names": [
        "struct",
        "struct type"
      ],
      "template": "type {name} struct {\n\t// TODO: add fields — or say \"add a <Name> field of type <Type> to {name}\"\n}"
    },
    {
      "id": "function",
      "kind": "declaration",
      "names": [
        "function",
        "func"
      ],
      "template": "func {name}() {\n\t// TODO: write the body\n}"
    },
    {
      "id": "enum",
      "kind": "declaration",
      "names": [
        "enum",
        "enumeration"
      ],
      "template": "type {name} int\n\nconst (\n\t{name}First {name} = iota\n\t// TODO: the rest of {name}'s values, one per line — they count up on their own\n)"
    },
    {
      "id": "embedding",
      "kind": "declaration",
      "names": [
        "embedded struct",
        "struct embedding",
        "embedding"
      ],
      "template": "type {name} struct {\n\t// TODO: embed a type by writing its name alone on a line of its own here — its fields and methods become part of {name}\n}"
    },
    {
      "id": "recursion",
      "kind": "declaration",
      "names": [
        "recursive function",
        "recursion"
      ],
      "template": "func {name}() {\n\t// TODO: the stopping case comes first — check it and return, or the calls never end\n\t{name}()\n\t// TODO: what each call does with the smaller problem\n}"
    },
    {
      "id": "type-switch",
      "kind": "statement",
      "names": [
        "type switch",
        "switch on type",
        "type assertion switch"
      ],
      "template": "var value any = \"TODO: the value whose type you want to check\"\nswitch value.(type) {\ncase string:\n\t// TODO: value holds a string — work with it here\ncase int:\n\t// TODO: value holds an int — work with it here\ndefault:\n\t// TODO: value holds something else\n}\n// TODO: write switch v := value.(type) instead, and each case sees v already holding that case's type"
    },
    {
      "id": "routing",
      "imports": [
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "routing",
        "router",
        "routes",
        "http routing",
        "serve mux"
      ],
      "template": "mux := http.NewServeMux()\nmux.HandleFunc(\"/hello/{name}\", func(w http.ResponseWriter, r *http.Request) {\n\t// TODO: write the response here — r.PathValue(\"name\") reads the {name} part of the URL\n})\n// TODO: hand mux to your server with http.ListenAndServe(\":8080\", mux) — patterns like /hello/{name} and \"GET /path\" are the go 1.22 mux's; an older go.mod matches paths literally"
    },
    {
      "id": "html-template",
      "imports": [
        "html/template"
      ],
      "kind": "statement",
      "names": [
        "html template",
        "html templates",
        "web template",
        "template"
      ],
      "template": "page := template.Must(template.ParseFiles(\"TODO: your template file.html\"))\n// TODO: inside a handler, write the page out with page.Execute(w, data) — data is what stands in for {{.}} in the template\n_ = page"
    },
    {
      "id": "static-files",
      "imports": [
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "static files",
        "static file serving",
        "serve static files",
        "assets and files"
      ],
      "template": "fs := http.FileServer(http.Dir(\"static\"))\nhttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", fs))\n// TODO: this serves the files in the static folder at /static/ once the server is running — rename the folder and the path to yours"
    },
    {
      "folder": "handlers",
      "id": "form-handler",
      "imports": [
        "net/http"
      ],
      "kind": "declaration",
      "names": [
        "form handler",
        "html form",
        "contact form",
        "form"
      ],
      "template": "func {name}(w http.ResponseWriter, r *http.Request) {\n\tif r.Method != http.MethodPost {\n\t\t// TODO: show the form here — a browser asks for it with GET before it can send anything\n\t\treturn\n\t}\n\tdetails := struct {\n\t\tEmail string\n\t\tSubject string\n\t\tMessage string\n\t}{\n\t\tEmail: r.FormValue(\"email\"),\n\t\tSubject: r.FormValue(\"subject\"),\n\t\tMessage: r.FormValue(\"message\"),\n\t}\n\t// TODO: details holds the form's fields — rename them to your form's, and use them here\n\t_ = details\n}"
    },
    {
      "folder": "middleware",
      "id": "middleware",
      "imports": [
        "log",
        "net/http"
      ],
      "kind": "declaration",
      "names": [
        "middleware",
        "logging middleware",
        "basic middleware"
      ],
      "template": "func {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn func(w http.ResponseWriter, r *http.Request) {\n\t\tlog.Println(r.URL.Path)\n\t\tnext(w, r)\n\t}\n}"
    },
    {
      "folder": "middleware",
      "id": "middleware-timing",
      "imports": [
        "log",
        "net/http",
        "time"
      ],
      "kind": "declaration",
      "names": [
        "timing middleware",
        "middleware timing",
        "advanced middleware"
      ],
      "template": "func {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn func(w http.ResponseWriter, r *http.Request) {\n\t\tstart := time.Now()\n\t\tdefer func() { log.Println(r.URL.Path, time.Since(start)) }()\n\t\tnext(w, r)\n\t}\n}"
    },
    {
      "folder": "middleware",
      "id": "middleware-chain",
      "imports": [
        "net/http"
      ],
      "kind": "declaration",
      "names": [
        "middleware chain",
        "chain middleware",
        "middleware chaining"
      ],
      "template": "func {name}(f http.HandlerFunc, middlewares ...func(http.HandlerFunc) http.HandlerFunc) http.HandlerFunc {\n\tfor _, m := range middlewares {\n\t\tf = m(f)\n\t}\n\treturn f\n}"
    },
    {
      "folder": "handlers",
      "id": "session-handler",
      "imports": [
        "net/http"
      ],
      "kind": "declaration",
      "names": [
        "session handler",
        "session cookie",
        "session",
        "login session"
      ],
      "template": "func {name}(w http.ResponseWriter, r *http.Request) {\n\t// A session in plain net/http is a cookie: the browser stores it and sends it back on every request.\n\thttp.SetCookie(w, &http.Cookie{Name: \"session\", Value: \"TODO: your session id\", Path: \"/\"})\n\t// TODO: write the response here — on later requests, read the cookie back with r.Cookie(\"session\")\n}"
    },
    {
      "id": "database",
      "imports": [
        "database/sql"
      ],
      "kind": "statement",
      "names": [
        "database",
        "database connection",
        "sql database",
        "open database"
      ],
      "template": "db, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n// TODO: database/sql ships no driver — add yours as a blank import at the top of the file, like _ \"github.com/go-sql-driver/mysql\" for MySQL\nif err == nil {\n\terr = db.Ping()\n}\nif err == nil {\n\t// TODO: the number caps how many connections the pool may open at once — tune it for your database\n\tdb.SetMaxOpenConns(10)\n}\n// TODO: Open alone never connects — the Ping above is what proves the database answers. Keep db and pass it around: it is a connection pool, not a single connection"
    },
    {
      "id": "database-query",
      "imports": [
        "database/sql"
      ],
      "kind": "statement",
      "names": [
        "database query",
        "query rows",
        "database rows",
        "query the database"
      ],
      "template": "db, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n// TODO: database/sql ships no driver — add yours as a blank import at the top of the file, like _ \"github.com/go-sql-driver/mysql\" for MySQL\nrows, err := db.Query(\"TODO: your query — one ? for each value after it\")\nif err == nil {\n\tdefer rows.Close()\n\tfor rows.Next() {\n\t\tvar name string\n\t\t// TODO: Scan takes one variable per column your query returns — this row reads just the first, as text\n\t\tif err := rows.Scan(&name); err == nil {\n\t\t\t_ = name\n\t\t}\n\t}\n\t_ = rows.Err()\n}"
    },
    {
      "id": "database-query-row",
      "imports": [
        "database/sql"
      ],
      "kind": "statement",
      "names": [
        "database query row",
        "query one row",
        "query a single row",
        "single row query"
      ],
      "template": "db, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n// TODO: database/sql ships no driver — add yours as a blank import at the top of the file, like _ \"github.com/go-sql-driver/mysql\" for MySQL\nvar name string\nerr = db.QueryRow(\"TODO: your query — one ? for each value after it\", 1).Scan(&name)\nif err == nil {\n\t_ = name\n}\n// TODO: when no row matches, Scan sets err to sql.ErrNoRows — check for that before treating err as a failure"
    },
    {
      "id": "database-exec",
      "imports": [
        "database/sql"
      ],
      "kind": "statement",
      "names": [
        "database exec",
        "database insert",
        "insert into database",
        "execute statement"
      ],
      "template": "db, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n// TODO: database/sql ships no driver — add yours as a blank import at the top of the file, like _ \"github.com/go-sql-driver/mysql\" for MySQL\nresult, err := db.Exec(\"TODO: your insert, update or delete — one ? for each value after it\", \"TODO: a value\")\nif err == nil {\n\t// TODO: result.LastInsertId() gives the new row's id when the table auto-numbers its rows\n\t_ = result\n}\n// TODO: statements that change data go through Exec — never Query, which would hold a connection open waiting for rows that never come"
    },
    {
      "id": "database-prepared",
      "imports": [
        "database/sql"
      ],
      "kind": "statement",
      "names": [
        "prepared statement",
        "database prepared statement",
        "prepare a statement"
      ],
      "template": "db, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n// TODO: database/sql ships no driver — add yours as a blank import at the top of the file, like _ \"github.com/go-sql-driver/mysql\" for MySQL\nstmt, err := db.Prepare(\"TODO: your statement — a ? where each value goes\")\nif err == nil {\n\tdefer stmt.Close()\n\t// TODO: run Exec once per set of values — preparing once is the point. Placeholders differ by database: ? here, $1, $2 in Postgres\n\tif _, err := stmt.Exec(\"TODO: one set of values\"); err == nil {\n\t}\n}"
    },
    {
      "id": "database-transaction",
      "imports": [
        "database/sql"
      ],
      "kind": "statement",
      "names": [
        "database transaction",
        "transaction",
        "sql transaction"
      ],
      "template": "db, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n// TODO: database/sql ships no driver — add yours as a blank import at the top of the file, like _ \"github.com/go-sql-driver/mysql\" for MySQL\ntx, err := db.Begin()\nif err == nil {\n\t// TODO: run every statement on tx (tx.Exec, tx.Query) — not on db — so they all share the transaction's one connection, and Commit keeps them or Rollback undoes them\n\tif err := tx.Commit(); err != nil {\n\t\t_ = tx.Rollback()\n\t}\n}"
    },
    {
      "id": "database-null",
      "imports": [
        "database/sql"
      ],
      "kind": "statement",
      "names": [
        "database null",
        "nullable column",
        "null value",
        "sql null"
      ],
      "template": "db, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n// TODO: database/sql ships no driver — add yours as a blank import at the top of the file, like _ \"github.com/go-sql-driver/mysql\" for MySQL\nrows, err := db.Query(\"TODO: your query\")\nif err == nil {\n\tdefer rows.Close()\n\tfor rows.Next() {\n\t\tvar name sql.NullString\n\t\tif err := rows.Scan(&name); err == nil {\n\t\t\t// TODO: name.Valid tells you whether the column was NULL; name.String holds the text when it wasn't\n\t\t\t_ = name\n\t\t}\n\t}\n}"
    },
    {
      "folder": "database",
      "id": "database-connect",
      "imports": [
        "context",
        "database/sql"
      ],
      "kind": "declaration",
      "names": [
        "database connect",
        "database connection function",
        "db connect function"
      ],
      "note": "Returns a ready connection pool: the handle from sql.Open, proved live by a Ping before the pool settings are applied. It compiles exactly as written — but until you add your driver's blank import (no driver ships with Go), the Ping fails at run time with \"unknown driver\", because Open only records the driver name and the connection string; nothing has registered that name yet. Whoever calls this owns the handle: close it when the program is done with it.",
      "template": "func {name}() (*sql.DB, error) {\n\t// TODO: no database driver ships with Go — add yours as a blank import at the top of this file, like _ \"github.com/go-sql-driver/mysql\" for MySQL, and put its registered name in the Open call below\n\tdb, err := sql.Open(\"TODO: driver name\", \"TODO: connection string\")\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\t// TODO: Open alone never connects — the Ping is what proves the database answers, and it is the call that fails with \"unknown driver\" until the blank import above is real\n\tif err := db.PingContext(context.Background()); err != nil {\n\t\treturn nil, err\n\t}\n\t// TODO: the number caps how many connections the pool may open at once — tune it for your database\n\tdb.SetMaxOpenConns(10)\n\treturn db, nil\n}"
    },
    {
      "id": "rqlite-connect",
      "imports": [
        "net/http",
        "time"
      ],
      "kind": "statement",
      "names": [
        "rqlite connection",
        "rqlite connect",
        "connect to rqlite",
        "rqlite client"
      ],
      "note": "This code needs a rqlite node (rqlited) already running at that address — it compiles and checks without one, but nothing answers at runtime until a node is up. The node owns the database file (it lives in the data folder rqlited was started with); your program never creates that file, only tables. Ask \"how do I run rqlite\" and I'll show the ways to start one.",
      "template": "rqliteAddr := \"http://localhost:4001\"\n// TODO: this is the address of your rqlite node — change it if your node listens somewhere else. A node (the rqlited program) must already be running at that address before this code can do anything: your program only ever talks to the node over HTTP, and the node — never your program — owns the database file\nrqliteClient := &http.Client{Timeout: 10 * time.Second}\nresp, err := rqliteClient.Get(rqliteAddr + \"/status\")\nif err == nil {\n\tdefer resp.Body.Close()\n\t// TODO: a 200 means the node answered — the JSON it sends back describes the node and its cluster. Any other answer, or none at all, means the node is not up yet\n\t_ = resp.StatusCode\n}"
    },
    {
      "id": "rqlite-create-table",
      "imports": [
        "bytes",
        "encoding/json",
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "rqlite create table",
        "rqlite table",
        "create table in rqlite",
        "rqlite create"
      ],
      "note": "This runs against a rqlite node (rqlited) that must already be running at that address — it compiles and checks without one, but at runtime nothing answers until a node is up. The table is created inside the node's own database file; your program never touches that file directly.",
      "template": "rqliteAddr := \"http://localhost:4001\"\npayload, err := json.Marshal([]string{\"CREATE TABLE TODO: your table (id INTEGER NOT NULL PRIMARY KEY, name TEXT)\"})\nvar resp *http.Response\nif err == nil {\n\tresp, err = http.Post(rqliteAddr+\"/db/execute\", \"application/json\", bytes.NewReader(payload))\n}\nif err == nil {\n\tdefer resp.Body.Close()\n\t// TODO: an HTTP 200 does not by itself prove the table was made — the JSON answer can carry an \"error\" entry for the statement even on a 200, so read it before trusting the table exists\n\t_ = resp.StatusCode\n}"
    },
    {
      "id": "rqlite-insert",
      "imports": [
        "bytes",
        "encoding/json",
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "rqlite insert",
        "insert into rqlite",
        "rqlite insert row",
        "insert row into rqlite"
      ],
      "note": "This runs against a rqlite node (rqlited) that must already be running at that address — it compiles and checks without one, but at runtime nothing answers until a node is up.",
      "template": "rqliteAddr := \"http://localhost:4001\"\n// TODO: one ? in the statement for each value listed after it — rqlite receives the values apart from the SQL, so a value stays data and can never change what the statement does\npayload, err := json.Marshal([]any{[]any{\"INSERT INTO TODO: your table(name) VALUES(?)\", \"TODO: a value\"}})\nvar resp *http.Response\nif err == nil {\n\tresp, err = http.Post(rqliteAddr+\"/db/execute\", \"application/json\", bytes.NewReader(payload))\n}\nif err == nil {\n\tdefer resp.Body.Close()\n\t// TODO: the JSON answer carries last_insert_id and rows_affected for the insert — or an \"error\" entry if the statement failed, which can arrive even on an HTTP 200\n\t_ = resp.StatusCode\n}"
    },
    {
      "id": "rqlite-query",
      "imports": [
        "bytes",
        "encoding/json",
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "rqlite query",
        "query rqlite",
        "rqlite select",
        "select from rqlite"
      ],
      "note": "This runs against a rqlite node (rqlited) that must already be running at that address — it compiles and checks without one, but at runtime nothing answers until a node is up.",
      "template": "rqliteAddr := \"http://localhost:4001\"\npayload, err := json.Marshal([]string{\"SELECT TODO: your query — for example SELECT * FROM your table\"})\nvar resp *http.Response\nif err == nil {\n\tresp, err = http.Post(rqliteAddr+\"/db/query\", \"application/json\", bytes.NewReader(payload))\n}\nif err == nil {\n\tdefer resp.Body.Close()\n\tvar answer struct {\n\t\tResults []struct {\n\t\t\tColumns []string\n\t\t\tValues  [][]any\n\t\t\tError   string\n\t\t}\n\t}\n\tif err := json.NewDecoder(resp.Body).Decode(&answer); err == nil {\n\t\t// TODO: answer.Results[0] is the result — Columns names its columns, Values carries one list per row (numbers arrive as float64). Check Results[0].Error first: rqlite can answer 200 and still report the query failed there. For a read that must never be stale, add ?level=strong to the /db/query address above\n\t\t_ = answer\n\t}\n}"
    },
    {
      "folder": "database",
      "id": "gorqlite-connect",
      "imports": [
        "github.com/rqlite/gorqlite"
      ],
      "kind": "declaration",
      "names": [
        "gorqlite connection",
        "gorqlite connect",
        "connect with gorqlite",
        "gorqlite client"
      ],
      "note": "Opens a connection to an rqlite node through the gorqlite client and returns it — whoever calls this owns it and closes it when done. This family only plans when the project already requires github.com/rqlite/gorqlite (run go get github.com/rqlite/gorqlite first); the client speaks rqlite's HTTP API itself, so no database/sql driver is involved.",
      "template": "func {name}() (*gorqlite.Connection, error) {\n\t// TODO: the node's address — put user:password@ before the host when the node asks for a login (like \"http://user:password@localhost:4001\"); add \"?level=strong\" at the end when reads must wait for the whole cluster to agree\n\tconn, err := gorqlite.Open(\"http://localhost:4001\")\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\t// TODO: unlike sql.Open, this Open talks to the node right away (it asks the cluster who leads), so an error returned here means the node did not answer\n\treturn conn, nil\n}"
    },
    {
      "id": "gorqlite-query",
      "imports": [
        "github.com/rqlite/gorqlite"
      ],
      "kind": "statement",
      "names": [
        "gorqlite query",
        "query with gorqlite",
        "query gorqlite"
      ],
      "note": "Runs one SELECT against an rqlite node through the gorqlite client and holds the rows for the walk the TODO describes. This family only plans when the project already requires github.com/rqlite/gorqlite — run go get github.com/rqlite/gorqlite first.",
      "template": "conn, err := gorqlite.Open(\"http://localhost:4001\")\nif err == nil {\n\t// TODO: your SELECT here — QueryOne runs it against the node and hands back the rows\n\trows, qerr := conn.QueryOne(\"SELECT TODO FROM TODO\")\n\tif qerr == nil {\n\t\t// TODO: walk the answer with rows.Next(), read each row with rows.Scan(...), check rows.Err when the walk ends — and close the connection when you are done with it\n\t\t_ = rows\n\t}\n}"
    },
    {
      "id": "gorqlite-write",
      "imports": [
        "github.com/rqlite/gorqlite"
      ],
      "kind": "statement",
      "names": [
        "gorqlite write",
        "write with gorqlite",
        "gorqlite insert",
        "insert with gorqlite"
      ],
      "note": "Writes one row into an rqlite node through the gorqlite client, the statement and its values kept apart so the values bind as data. This family only plans when the project already requires github.com/rqlite/gorqlite — run go get github.com/rqlite/gorqlite first.",
      "template": "conn, err := gorqlite.Open(\"http://localhost:4001\")\nif err == nil {\n\t// TODO: your INSERT here — the values ride as arguments beside the ? marks, so the node binds them as data, never as SQL text\n\tresult, werr := conn.WriteOneParameterized(gorqlite.ParameterizedStatement{\n\t\tQuery:     \"INSERT INTO TODO (TODO) VALUES (?)\",\n\t\tArguments: []any{\"TODO: a value\"},\n\t})\n\tif werr == nil {\n\t\t// TODO: check result.Err for the statement's own error; result.LastInsertID and result.RowsAffected say what the write did. Close the connection when you are done with it\n\t\t_ = result\n\t}\n}"
    },
    {
      "id": "form-values",
      "kind": "statement",
      "names": [
        "form values",
        "form data",
        "submitted form values",
        "parse form"
      ],
      "note": "This piece reads the values a form sent, so it belongs inside a handler — it parses and reads the request that handler receives.",
      "template": "if err := {request}.ParseForm(); err == nil {\n\t// TODO: {request}.Form now holds everything the form sent, by field name — read one field with {request}.FormValue(\"email\"), or walk them all like this\n\tfor key, values := range {request}.Form {\n\t\t_ = key\n\t\t_ = values\n\t}\n}"
    },
    {
      "id": "append-slice",
      "kind": "statement",
      "names": [
        "append to a slice",
        "append to slice",
        "slice append"
      ],
      "template": "names := []string{\"ann\", \"bo\"}\nnames = append(names, \"cy\")\n// TODO: your own slice and values — append returns the grown slice, so the result has to be assigned back"
    },
    {
      "id": "map-lookup",
      "kind": "statement",
      "names": [
        "map lookup",
        "look up a key",
        "find a key in a map",
        "check a map key"
      ],
      "template": "ages := map[string]int{\"ann\": 3}\nage, ok := ages[\"ann\"]\n// TODO: your own map and key — ok is false when the key isn't there (and age is the zero value), so a missing key can't fool you\n_ = age\n_ = ok"
    },
    {
      "id": "env-var",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "environment variable",
        "env var",
        "read an environment variable",
        "get an environment variable"
      ],
      "template": "value := os.Getenv(\"TODO: the variable's name\")\n// TODO: an unset variable reads as \"\" — use os.LookupEnv when you need to tell \"unset\" from \"set but empty\"\n_ = value"
    },
    {
      "id": "working-directory",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "current working directory",
        "working directory",
        "current directory",
        "get the working directory"
      ],
      "template": "dir, err := os.Getwd()\nif err == nil {\n\t// TODO: dir is the folder the program was started from — relative paths are read from here\n\t_ = dir\n}"
    },
    {
      "id": "read-lines",
      "imports": [
        "bufio",
        "os"
      ],
      "kind": "statement",
      "names": [
        "read a file line by line",
        "read lines from a file",
        "read file line by line",
        "line by line reading"
      ],
      "template": "file, err := os.Open(\"TODO: the file's path\")\nif err == nil {\n\tdefer file.Close()\n\tscanner := bufio.NewScanner(file)\n\tfor scanner.Scan() {\n\t\tline := scanner.Text()\n\t\t// TODO: one line of the file — do what the line is for here\n\t\t_ = line\n\t}\n}"
    },
    {
      "id": "append-file",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "append to a file",
        "append text to a file",
        "append to file",
        "append a file"
      ],
      "template": "file, err := os.OpenFile(\"TODO: the file's path\", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)\nif err == nil {\n\tdefer file.Close()\n\tif _, err := file.WriteString(\"TODO: the text to add\\n\"); err != nil {\n\t\t// TODO: the append failed — decide what to do about it\n\t}\n}"
    },
    {
      "id": "walk-directory",
      "imports": [
        "os",
        "path/filepath"
      ],
      "kind": "statement",
      "names": [
        "walk a directory",
        "walk directory",
        "list files recursively",
        "recursive file listing"
      ],
      "template": "err := filepath.Walk(\"TODO: the folder's path\", func(path string, info os.FileInfo, err error) error {\n\t// TODO: path is each file and folder under the start, one call each — do what each path is for here\n\t_ = path\n\t_ = info\n\treturn err\n})\n// TODO: err is the first problem the walk met, if any\n_ = err"
    },
    {
      "id": "log-to-file",
      "imports": [
        "log",
        "os"
      ],
      "kind": "statement",
      "names": [
        "log to a file",
        "log to file",
        "write log to a file",
        "logging to a file"
      ],
      "template": "file, err := os.Create(\"TODO: the log file's path\")\nif err == nil {\n\tdefer file.Close()\n\tlog.SetOutput(file)\n\t// TODO: from here, log lines go to the file instead of the screen\n}"
    },
    {
      "id": "string-builder",
      "imports": [
        "strings"
      ],
      "kind": "statement",
      "names": [
        "string builder",
        "build a string",
        "concatenate strings",
        "string concatenation"
      ],
      "template": "var builder strings.Builder\nbuilder.WriteString(\"hello\")\nbuilder.WriteString(\", \")\nbuilder.WriteString(\"world\")\ntext := builder.String()\n// TODO: a builder grows a string piece by piece without copying it on every step — text is the finished string\n_ = text"
    },
    {
      "id": "split-string",
      "imports": [
        "strings"
      ],
      "kind": "statement",
      "names": [
        "split a string",
        "split string",
        "string split",
        "split text"
      ],
      "template": "parts := strings.Split(\"ann,bo,cy\", \",\")\n// TODO: your own text and separator — parts holds the pieces, in order\n_ = parts"
    },
    {
      "id": "trim-string",
      "imports": [
        "strings"
      ],
      "kind": "statement",
      "names": [
        "trim whitespace",
        "trim a string",
        "trim string",
        "trim spaces"
      ],
      "template": "trimmed := strings.TrimSpace(\"  hello  \")\n// TODO: your own text — TrimSpace cuts spaces (and tabs and newlines) off both ends\n_ = trimmed"
    },
    {
      "id": "reverse-string",
      "kind": "statement",
      "names": [
        "reverse a string",
        "string reverse",
        "reverse string",
        "reversed string"
      ],
      "template": "text := \"hello\"\nrunes := []rune(text)\nfor i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {\n\trunes[i], runes[j] = runes[j], runes[i]\n}\nreversed := string(runes)\n// TODO: your own text — going through []rune keeps letters that take more than one byte in one piece\n_ = reversed"
    },
    {
      "id": "delete-from-slice",
      "kind": "statement",
      "names": [
        "delete from a slice",
        "remove from a slice",
        "delete an element from a slice",
        "remove an element from a slice"
      ],
      "template": "names := []string{\"ann\", \"bo\", \"cy\"}\n// TODO: the element at index 1 goes — pick your own slice and index\nnames = append(names[:1], names[2:]...)"
    },
    {
      "id": "clear-slice",
      "kind": "statement",
      "names": [
        "clear a slice",
        "empty a slice",
        "clear slice"
      ],
      "template": "names := []string{\"ann\", \"bo\", \"cy\"}\nnames = names[:0]\n// TODO: your own slice — its length is now 0, but the backing array (and its capacity) is kept, so it can grow again without new allocation; assigning nil instead gives the memory back"
    },
    {
      "id": "slice-search",
      "kind": "statement",
      "names": [
        "slice search",
        "find in a slice",
        "search a slice",
        "find an element in a slice"
      ],
      "template": "names := []string{\"ann\", \"bo\", \"cy\"}\nfound := -1\nfor i, name := range names {\n\tif name == \"bo\" {\n\t\tfound = i\n\t\tbreak\n\t}\n}\n// TODO: your own slice and the value you're hunting — found is its index, or -1 when it isn't there\n_ = found"
    },
    {
      "id": "copy-slice",
      "kind": "statement",
      "names": [
        "slice copy",
        "copy a slice",
        "copy slice",
        "copying a slice"
      ],
      "template": "names := []string{\"ann\", \"bo\", \"cy\"}\ncopied := make([]string, len(names))\ncopy(copied, names)\n// TODO: copied is independent — changing it leaves names alone (a plain second := names would share the same backing array)\n_ = copied"
    },
    {
      "id": "shuffle",
      "imports": [
        "math/rand"
      ],
      "kind": "statement",
      "names": [
        "shuffle a slice",
        "shuffle slice",
        "shuffle",
        "shuffled slice"
      ],
      "template": "names := []string{\"ann\", \"bo\", \"cy\"}\nrand.Shuffle(len(names), func(i, j int) {\n\tnames[i], names[j] = names[j], names[i]\n})\n// TODO: your own slice — it is shuffled in place, into a random order"
    },
    {
      "id": "execution-time",
      "imports": [
        "time"
      ],
      "kind": "statement",
      "names": [
        "execution time",
        "measure execution time",
        "measure time"
      ],
      "template": "start := time.Now()\n// TODO: the code whose time you want to measure goes here\nelapsed := time.Since(start)\n// TODO: elapsed is how long it took — print it, or decide something with it\n_ = elapsed"
    },
    {
      "folder": "security",
      "id": "cors",
      "imports": [
        "net/http"
      ],
      "kind": "declaration",
      "names": [
        "cors middleware",
        "cors",
        "cross origin middleware",
        "cross-origin resource sharing"
      ],
      "note": "CORS is not authentication: it loosens the browser's same-origin rule for the origins named — anyone can still call the API from a non-browser client. The origin in the template is a TODO only you can choose: a wildcard (*) together with cookies or tokens is the footgun; named origins are the safe shape.",
      "template": "func {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn func(w http.ResponseWriter, r *http.Request) {\n\t\t// CORS lets a browser page served from ONE origin read answers from another. The answer below names the page's origin — TODO: replace it with the one origin your own page is served from\n\t\tw.Header().Set(\"Access-Control-Allow-Origin\", \"TODO: https://your-page.example\")\n\t\tw.Header().Set(\"Access-Control-Allow-Methods\", \"GET, POST, PUT, DELETE, OPTIONS\")\n\t\tw.Header().Set(\"Access-Control-Allow-Headers\", \"Content-Type, Authorization\")\n\t\t// Never answer \"*\" for the origin when requests carry cookies or tokens — a wildcard plus credentials is the combination browsers reject and attackers love; an origin must be named\n\t\tif r.Method == http.MethodOptions {\n\t\t\t// The browser's preflight question, asked before the real request: the headers above are the whole answer\n\t\t\tw.WriteHeader(http.StatusNoContent)\n\t\t\treturn\n\t\t}\n\t\tnext(w, r)\n\t}\n}"
    },
    {
      "folder": "security",
      "id": "jwt-verify",
      "imports": [
        "crypto/hmac",
        "crypto/sha256",
        "encoding/base64",
        "encoding/json",
        "net/http",
        "os",
        "strings",
        "time"
      ],
      "kind": "declaration",
      "names": [
        "jwt middleware",
        "jwt auth",
        "jwt verification",
        "jwt auth middleware"
      ],
      "note": "The token's claims are trusted only AFTER the signature and the expiry check out — before that they are text a stranger typed. Production systems usually take a maintained JWT library (an external dependency you add yourself); this hand-rolled HS256 check is the honest small-server shape, and it reads the secret from the environment, never from code.",
      "template": "func {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn func(w http.ResponseWriter, r *http.Request) {\n\t\tsecret := os.Getenv(\"JWT_SECRET\")\n\t\t// TODO: JWT_SECRET is the environment variable your server reads its signing secret from — the same one the token was created with. This code never obeys the token's own \"alg\" header: it always checks with HMAC-SHA256, so a token asking for a weaker (or \"none\") check gets nothing\n\t\ttoken, found := strings.CutPrefix(r.Header.Get(\"Authorization\"), \"Bearer \")\n\t\tparts := strings.Split(token, \".\")\n\t\tverified := false\n\t\tvar claims map[string]any\n\t\tif found && len(parts) == 3 {\n\t\t\tunsigned := parts[0] + \".\" + parts[1]\n\t\t\tmac := hmac.New(sha256.New, []byte(secret))\n\t\t\tmac.Write([]byte(unsigned))\n\t\t\tsignature, sigErr := base64.RawURLEncoding.DecodeString(parts[2])\n\t\t\tif sigErr == nil && hmac.Equal(signature, mac.Sum(nil)) {\n\t\t\t\tpayload, payloadErr := base64.RawURLEncoding.DecodeString(parts[1])\n\t\t\t\tif payloadErr == nil && json.Unmarshal(payload, &claims) == nil {\n\t\t\t\t\tif exp, isNumber := claims[\"exp\"].(float64); isNumber && time.Now().Unix() < int64(exp) {\n\t\t\t\t\t\tverified = true\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}\n\t\t}\n\t\tif !verified {\n\t\t\thttp.Error(w, \"unauthorized\", http.StatusUnauthorized)\n\t\t\treturn\n\t\t}\n\t\t// TODO: claims now holds claims you can trust — claims[\"sub\"] says who the token was issued to. Pass that on (a request context is the usual way) instead of believing anything the request itself claims about who sent it\n\t\t_ = claims\n\t\tnext(w, r)\n\t}\n}"
    },
    {
      "folder": "security",
      "id": "csp-nonce",
      "imports": [
        "context",
        "crypto/rand",
        "encoding/base64",
        "net/http"
      ],
      "kind": "declaration",
      "names": [
        "csp nonce middleware",
        "nonce middleware",
        "csp nonce",
        "content security policy nonce"
      ],
      "note": "The nonce is minted fresh from crypto/rand for every request and dies with it: a script copied from yesterday's page — or injected by an attacker — carries the wrong value and the browser refuses it. The policy also pins everything else to your own origin ('self').",
      "template": "type {name}Key struct{}\n\n// {name} mints a fresh nonce — a one-time random value — for every request, sets the Content-Security-Policy header that demands it, and passes the value along for the page's own templates.\nfunc {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn func(w http.ResponseWriter, r *http.Request) {\n\t\traw := make([]byte, 16)\n\t\tif _, err := rand.Read(raw); err != nil {\n\t\t\t// No honest randomness, no honest nonce — refuse rather than send a page whose policy a stranger could guess\n\t\t\thttp.Error(w, \"internal error\", http.StatusInternalServerError)\n\t\t\treturn\n\t\t}\n\t\tnonce := base64.StdEncoding.EncodeToString(raw)\n\t\tw.Header().Set(\"Content-Security-Policy\", \"default-src 'self'; script-src 'self' 'nonce-\"+nonce+\"'; style-src 'self' 'nonce-\"+nonce+\"'\")\n\t\t// TODO: your templates read this request's nonce back with r.Context().Value({name}Key{}) and write it onto their own tags — <script nonce=\"...\"> — because a script WITHOUT this request's nonce is exactly what the browser now refuses to run\n\t\tnext(w, r.WithContext(context.WithValue(r.Context(), {name}Key{}, nonce)))\n\t}\n}"
    },
    {
      "folder": "security",
      "id": "csrf",
      "imports": [
        "crypto/rand",
        "crypto/subtle",
        "encoding/base64",
        "net/http"
      ],
      "kind": "declaration",
      "names": [
        "csrf protection",
        "csrf middleware",
        "csrf",
        "cross-site request forgery protection"
      ],
      "note": "Go 1.25's net/http has this built in — http.NewCrossOriginProtection checks where a request came from instead of comparing tokens — but it only compiles into projects whose go.mod says go 1.25 or newer. This row is the double-submit token shape every project can take; on go 1.25+, prefer the built-in.",
      "template": "func {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn func(w http.ResponseWriter, r *http.Request) {\n\t\t// CSRF: a malicious page can make a signed-in browser SEND your server a request, but it can never READ your pages — so a state-changing request must hand back a random token only a page you served could know (the double-submit pattern)\n\t\tcookie, cookieErr := r.Cookie(\"csrf_token\")\n\t\ttoken := \"\"\n\t\tif cookieErr == nil {\n\t\t\ttoken = cookie.Value\n\t\t}\n\t\tif token == \"\" {\n\t\t\traw := make([]byte, 32)\n\t\t\tif _, err := rand.Read(raw); err != nil {\n\t\t\t\thttp.Error(w, \"internal error\", http.StatusInternalServerError)\n\t\t\t\treturn\n\t\t\t}\n\t\t\ttoken = base64.StdEncoding.EncodeToString(raw)\n\t\t\thttp.SetCookie(w, &http.Cookie{Name: \"csrf_token\", Value: token, Path: \"/\", HttpOnly: true, SameSite: http.SameSiteLaxMode})\n\t\t\t// A fresh token was just minted for the NEXT request — this one still has nothing proven, and the check below refuses it if it tries to change anything\n\t\t}\n\t\tif r.Method != http.MethodGet && r.Method != http.MethodHead && r.Method != http.MethodOptions {\n\t\t\tsent := r.Header.Get(\"X-CSRF-Token\")\n\t\t\tif sent == \"\" {\n\t\t\t\tsent = r.FormValue(\"csrf_token\")\n\t\t\t}\n\t\t\tif subtle.ConstantTimeCompare([]byte(sent), []byte(token)) != 1 {\n\t\t\t\thttp.Error(w, \"forbidden\", http.StatusForbidden)\n\t\t\t\treturn\n\t\t\t}\n\t\t\t// TODO: your forms send the token back — a hidden field named csrf_token holding the csrf_token cookie's value (read it in the handler that shows the form), or an X-CSRF-Token header from your scripts. The comparison is constant-time, so a wrong guess reveals nothing about how close it was\n\t\t}\n\t\tnext(w, r)\n\t}\n}"
    },
    {
      "folder": "security",
      "id": "security-headers",
      "imports": [
        "net/http"
      ],
      "kind": "declaration",
      "names": [
        "security headers",
        "security headers middleware",
        "http security headers",
        "secure headers"
      ],
      "note": "Headers only speak to browsers, and only when set before the answer is written — wrap the OUTERMOST handler with this so every route gets them. HSTS stays a TODO in the template on purpose: it is HTTPS-only, and a browser that hears it remembers — sending it while you still test over plain http locks you out of your own site.",
      "template": "func {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn func(w http.ResponseWriter, r *http.Request) {\n\t\t// These headers ask the browser to treat your pages carefully — they cost nothing and close whole families of tricks\n\t\tw.Header().Set(\"X-Content-Type-Options\", \"nosniff\")\n\t\tw.Header().Set(\"X-Frame-Options\", \"DENY\")\n\t\tw.Header().Set(\"Referrer-Policy\", \"strict-origin-when-cross-origin\")\n\t\tw.Header().Set(\"Content-Security-Policy\", \"default-src 'self'; frame-ancestors 'none'\")\n\t\t// TODO: over HTTPS, also set Strict-Transport-Security to max-age=63072000 so browsers refuse plain http for your site — never send it while plain http is still on purpose (local testing): a browser that hears it remembers\n\t\tnext(w, r)\n\t}\n}"
    },
    {
      "folder": "security",
      "id": "rate-limit",
      "imports": [
        "net",
        "net/http",
        "sync",
        "time"
      ],
      "kind": "declaration",
      "names": [
        "rate limit middleware",
        "rate limiter",
        "rate limiting",
        "throttle middleware"
      ],
      "note": "Visitors are counted by the address the request arrives from: behind a proxy or a shared office NAT, many people share one address and so share one count. A fixed window also means a burst straddling the boundary can briefly double the limit. A busy production API usually reaches for golang.org/x/time/rate (an external package); this is the honest small-server shape.",
      "template": "// {name} counts requests per visitor inside a fixed window; past the limit the answer is 429 until the window turns over.\nvar {name}State = struct {\n\tsync.Mutex\n\twindowStart time.Time\n\tcounts      map[string]int\n}{windowStart: time.Now(), counts: map[string]int{}}\n\nfunc {name}(next http.HandlerFunc) http.HandlerFunc {\n\treturn func(w http.ResponseWriter, r *http.Request) {\n\t\tvisitor, _, splitErr := net.SplitHostPort(r.RemoteAddr)\n\t\tif splitErr != nil {\n\t\t\tvisitor = r.RemoteAddr\n\t\t}\n\t\t{name}State.Lock()\n\t\tif time.Since({name}State.windowStart) >= time.Minute {\n\t\t\t// TODO: your window — one minute here. A window turning over wipes every count at once\n\t\t\t{name}State.windowStart = time.Now()\n\t\t\t{name}State.counts = map[string]int{}\n\t\t}\n\t\t{name}State.counts[visitor]++\n\t\tcount := {name}State.counts[visitor]\n\t\t{name}State.Unlock()\n\t\tif count > 100 {\n\t\t\t// TODO: your limit — requests per window, per visitor\n\t\t\thttp.Error(w, \"too many requests\", http.StatusTooManyRequests)\n\t\t\treturn\n\t\t}\n\t\tnext(w, r)\n\t}\n}"
    },
    {
      "id": "jwt-create",
      "imports": [
        "crypto/hmac",
        "crypto/sha256",
        "encoding/base64",
        "encoding/json",
        "os",
        "time"
      ],
      "kind": "statement",
      "names": [
        "jwt",
        "json web token",
        "jwt token",
        "make a jwt"
      ],
      "note": "Anyone can READ a JWT's claims — the signature proves they weren't changed, it does not hide them — so a token never carries a password or a secret. And the signing secret here comes from the JWT_SECRET environment variable: a secret written in code is a secret published with the code.",
      "template": "secret := os.Getenv(\"JWT_SECRET\")\n// TODO: JWT_SECRET is the environment variable YOUR server reads its signing secret from — the secret itself never goes in code. A JWT is three base64url parts joined by dots (header, claims, signature); the signature below is HMAC-SHA256 over the first two\nclaims := map[string]any{\n\t\"sub\": \"TODO: who the token is for — usually the user's id\",\n\t\"exp\": time.Now().Add(time.Hour).Unix(),\n}\nclaimsJSON, claimsErr := json.Marshal(claims)\ntoken := \"\"\nif claimsErr == nil {\n\tunsigned := base64.RawURLEncoding.EncodeToString([]byte(\"{\\\"alg\\\":\\\"HS256\\\",\\\"typ\\\":\\\"JWT\\\"}\")) + \".\" + base64.RawURLEncoding.EncodeToString(claimsJSON)\n\tmac := hmac.New(sha256.New, []byte(secret))\n\tmac.Write([]byte(unsigned))\n\ttoken = unsigned + \".\" + base64.RawURLEncoding.EncodeToString(mac.Sum(nil))\n}\n// TODO: token is the signed JWT — hand it to the client (in the login answer, say); it stays valid until its exp, an hour here, so keep lifetimes short\n_ = token"
    },
    {
      "id": "idor-check",
      "imports": [
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "idor check",
        "ownership check",
        "idor",
        "check ownership"
      ],
      "note": "Run this check on EVERY way in — read, change, delete, download — not just the page that shows the record: an attacker aims at whichever door was left unchecked. Making ids unguessable (random instead of counting up) slows the guessing down, but this check is the actual lock.",
      "template": "callerID := \"TODO: the signed-in user's id — from the session or the verified token, never from the URL, a form field, or anything else the caller typed\"\nrecordOwnerID := \"TODO: the id of the user who owns the record this request reached — read it off the record itself, after you load it\"\n// An id in the URL is not permission: whoever the record belongs to, prove the caller IS that person before the code below does anything with it\nif callerID == \"\" || callerID != recordOwnerID {\n\thttp.Error({response}, \"forbidden\", http.StatusForbidden)\n\treturn\n}"
    },
    {
      "id": "secure-cookie",
      "imports": [
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "secure cookie",
        "cookie flags",
        "hardened cookie",
        "safe cookie"
      ],
      "note": "The session handler sets this same cookie with none of the flags set. These flags are the difference between a cookie that other sites and stray scripts can abuse and one they can't — and Secure is the one that bites in testing: over plain http on a non-localhost address the browser silently refuses the cookie, which looks exactly like the cookie code not working.",
      "template": "cookie := &http.Cookie{\n\tName:  \"session\",\n\tValue: \"TODO: your session id\",\n\tPath:  \"/\",\n\t// Secure: the browser sends this cookie only over HTTPS. Browsers treat localhost as trustworthy even over http, but on any other plain-http address the cookie is silently refused\n\tSecure: true,\n\t// HttpOnly: page scripts cannot read this cookie — a buggy or injected script cannot steal what it cannot read\n\tHttpOnly: true,\n\t// SameSite Lax: the cookie stays home on cross-site requests, except top-level page visits — Strict would withhold it even then, which breaks following a link into your site while signed in\n\tSameSite: http.SameSiteLaxMode,\n}\nhttp.SetCookie({response}, cookie)"
    },
    {
      "id": "server-timeouts",
      "imports": [
        "net/http",
        "time"
      ],
      "kind": "statement",
      "names": [
        "server timeouts",
        "http server timeouts",
        "server with timeouts",
        "hardened server"
      ],
      "note": "This builds the server value but starts nothing: the fields are the whole piece. The timeouts are also a promise about your own handlers — a handler slower than WriteTimeout is cut off mid-answer, so the numbers belong above your slowest honest page, not below it.",
      "template": "server := &http.Server{\n\tAddr: \":8080\",\n\t// TODO: your address — and your Handler, if your routes live on a mux of your own; left unset, this serves everything registered on the default mux (http.Handle / http.HandleFunc)\n\tReadHeaderTimeout: 5 * time.Second,\n\tReadTimeout:       10 * time.Second,\n\tWriteTimeout:      10 * time.Second,\n\tIdleTimeout:       60 * time.Second,\n}\n// TODO: start it with server.ListenAndServe() where a plain http.ListenAndServe would go. A server with NO timeouts lets one slow client hold a connection open forever — the trick behind slow-loris stalls — and ReadHeaderTimeout is the field that closes it\n_ = server"
    },
    {
      "id": "body-limit",
      "imports": [
        "net/http"
      ],
      "kind": "statement",
      "names": [
        "body limit",
        "request body limit",
        "limit body size",
        "body size limit"
      ],
      "note": "A request that crosses the limit fails its READS — the handler hears it as an error from reading or ParseForm, and hears nothing at all if it never reads the body. The limit only guards a handler that actually consumes what was sent.",
      "template": "limit := int64(1 << 20)\n{request}.Body = http.MaxBytesReader({response}, {request}.Body, limit)\n// TODO: one megabyte here — pick the biggest body this handler honestly needs. From here on, reading the body past the limit fails with an error instead of swallowing memory, so a giant upload cannot fill the server. This piece belongs BEFORE anything else reads the body — move it up if the reads come first"
    },
    {
      "id": "while-loop",
      "kind": "statement",
      "names": [
        "while loop",
        "while"
      ],
      "template": "count := 0\nfor count < 3 {\n\t// TODO: the work of each pass goes here — the loop runs while the condition holds, so something has to move it toward false, or it never ends\n\tcount++\n}"
    },
    {
      "id": "infinite-loop",
      "kind": "statement",
      "names": [
        "infinite loop",
        "forever loop",
        "endless loop"
      ],
      "template": "for {\n\t// TODO: a loop with no condition and no counter runs until something inside it stops it — a break, a return, or the program ending\n}"
    },
    {
      "id": "type-conversion",
      "kind": "statement",
      "names": [
        "type conversion",
        "type cast",
        "convert a type"
      ],
      "template": "whole := 2\ndecimal := float64(whole)\n// TODO: a conversion rewrites a value as another type — float64(whole) here; int(decimal) converts back, cutting the fraction away\n_ = decimal"
    },
    {
      "id": "sleep",
      "imports": [
        "time"
      ],
      "kind": "statement",
      "names": [
        "sleep",
        "pause execution",
        "delay"
      ],
      "template": "time.Sleep(2 * time.Second)\n// TODO: the program waits right here for as long as you set — this goroutine does nothing else until the time has passed"
    },
    {
      "id": "delete-from-map",
      "kind": "statement",
      "names": [
        "delete from a map",
        "remove from a map",
        "delete a map entry",
        "delete a key from a map"
      ],
      "template": "ages := map[string]int{\"ann\": 3, \"bo\": 4}\ndelete(ages, \"ann\")\n// TODO: your own map and key — deleting a key that isn't there is not an error, and the rest of the map is untouched"
    },
    {
      "id": "clear-map",
      "kind": "statement",
      "names": [
        "clear a map",
        "empty a map",
        "clear map"
      ],
      "template": "ages := map[string]int{\"ann\": 3}\nclear(ages)\n// TODO: your own map — clear takes every entry out, and the map itself stays, ready to be filled again"
    },
    {
      "id": "map-keys",
      "kind": "statement",
      "names": [
        "map keys",
        "list of map keys",
        "list the keys of a map",
        "keys of a map"
      ],
      "template": "ages := map[string]int{\"ann\": 3, \"bo\": 4}\nkeys := make([]string, 0, len(ages))\nfor key := range ages {\n\tkeys = append(keys, key)\n}\n// TODO: your own map — keys now holds every key, in no promised order; range for the value instead of the key and you collect the values the same way\n_ = keys"
    },
    {
      "id": "read-input",
      "imports": [
        "bufio",
        "os"
      ],
      "kind": "statement",
      "names": [
        "read input",
        "read from stdin",
        "read a line from stdin",
        "read from standard input",
        "standard input"
      ],
      "template": "scanner := bufio.NewScanner(os.Stdin)\nif scanner.Scan() {\n\tline := scanner.Text()\n\t// TODO: line is what was typed — keep calling scanner.Scan() in a loop to read line after line, or read one value straight into a variable with fmt.Scan(&value)\n\t_ = line\n}"
    },
    {
      "id": "swap-values",
      "kind": "statement",
      "names": [
        "swap values",
        "swap two variables",
        "swap variables",
        "swap"
      ],
      "template": "first := \"ann\"\nsecond := \"bo\"\nfirst, second = second, first\n// TODO: your own two variables — Go trades their values in one line, no temporary variable needed"
    },
    {
      "id": "sum",
      "kind": "statement",
      "names": [
        "sum of a slice",
        "sum a slice",
        "add up a slice",
        "sum"
      ],
      "template": "numbers := []int{1, 2, 3}\ntotal := 0\nfor _, number := range numbers {\n\ttotal += number\n}\n// TODO: your own slice — total holds the sum when the loop ends\n_ = total"
    },
    {
      "id": "join-strings",
      "imports": [
        "strings"
      ],
      "kind": "statement",
      "names": [
        "join strings",
        "join a list of strings",
        "join a slice of strings",
        "join"
      ],
      "template": "words := []string{\"ann\", \"bo\", \"cy\"}\njoined := strings.Join(words, \", \")\n// TODO: your own slice and separator — Join glues the pieces into one string with the separator between them\n_ = joined"
    },
    {
      "id": "repeat-string",
      "imports": [
        "strings"
      ],
      "kind": "statement",
      "names": [
        "repeat a string",
        "repeat string",
        "repeated string",
        "string repeat"
      ],
      "template": "repeated := strings.Repeat(\"ha\", 3)\n// TODO: your own text and count — the result is the text written out that many times, end to end\n_ = repeated"
    },
    {
      "id": "string-contains",
      "imports": [
        "strings"
      ],
      "kind": "statement",
      "names": [
        "string contains",
        "check if a string contains",
        "check a string contains",
        "search a string"
      ],
      "template": "text := \"hello world\"\nfound := strings.Contains(text, \"world\")\n// TODO: your own text and the piece you hunt — strings.Index says where a piece sits, LastIndex finds the last one, HasPrefix and HasSuffix test the two ends, and Count says how many times a piece appears\n_ = found"
    },
    {
      "id": "file-exists",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "file exists",
        "file exists check",
        "check if a file exists"
      ],
      "template": "info, err := os.Stat(\"TODO: the path\")\nexists := err == nil\n// TODO: exists is true when the path is there — info then describes it (info.Size() is its size, info.IsDir() says folder), and when err is os.IsNotExist the path simply isn't there\n_ = exists\n_ = info"
    },
    {
      "id": "delete-file",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "delete a file",
        "remove a file",
        "delete file"
      ],
      "template": "if err := os.Remove(\"TODO: the file's path\"); err != nil {\n\t// TODO: the remove failed — the file may not be there, or the program may not be allowed; decide what to do about it\n}"
    },
    {
      "id": "rename-file",
      "imports": [
        "os"
      ],
      "kind": "statement",
      "names": [
        "rename a file",
        "move a file",
        "rename file"
      ],
      "template": "if err := os.Rename(\"TODO: the file's path now\", \"TODO: the path it should get\"); err != nil {\n\t// TODO: the rename failed — decide what to do about it\n}"
    },
    {
      "id": "hex-encoding",
      "imports": [
        "encoding/hex"
      ],
      "kind": "statement",
      "names": [
        "hex encoding",
        "encode to hex",
        "hex encode",
        "hex"
      ],
      "template": "encoded := hex.EncodeToString([]byte(\"TODO: your text\"))\n// TODO: hex writes each byte as two characters, 0-9 and a-f — hex.DecodeString reads the text back into bytes\n_ = encoded"
    },
    {
      "id": "insert-into-slice",
      "kind": "statement",
      "names": [
        "insert into a slice",
        "insert an element into a slice",
        "insert in a slice",
        "insert into slice"
      ],
      "template": "names := []string{\"ann\", \"cy\"}\nnames = append(names, \"\")\ncopy(names[2:], names[1:])\nnames[1] = \"bo\"\n// TODO: your own slice, position and value — the append makes room, the copy shifts everything after the position one place over, and the value drops into the gap"
    },
    {
      "id": "remove-value-from-slice",
      "kind": "statement",
      "names": [
        "remove a value from a slice",
        "remove all occurrences from a slice",
        "remove occurrences from a slice",
        "remove values from a slice"
      ],
      "template": "names := []string{\"ann\", \"bo\", \"cy\", \"bo\"}\nkept := names[:0]\nfor _, name := range names {\n\tif name != \"bo\" {\n\t\tkept = append(kept, name)\n\t}\n}\nnames = kept\n// TODO: your own slice and the value that goes — every copy of it is dropped and the rest keep their order; kept shares the slice's backing array, so the tail beyond the new end still holds the old values"
    },
    {
      "id": "reverse-slice",
      "kind": "statement",
      "names": [
        "reverse a slice",
        "reverse a list",
        "reverse slice",
        "slice reverse"
      ],
      "template": "names := []string{\"ann\", \"bo\", \"cy\"}\nfor i, j := 0, len(names)-1; i < j; i, j = i+1, j-1 {\n\tnames[i], names[j] = names[j], names[i]\n}\n// TODO: your own slice — the two ends walk toward the middle, trading places, until the whole slice reads backwards"
    },
    {
      "id": "print-stderr",
      "imports": [
        "fmt",
        "os"
      ],
      "kind": "statement",
      "names": [
        "print to stderr",
        "write to stderr",
        "print to standard error",
        "standard error print"
      ],
      "template": "fmt.Fprintln(os.Stderr, \"TODO: the message\")\n// TODO: standard error is the stream set aside for problems — shells and tools can send it somewhere different from the program's normal output"
    }
  ],
  "source": "Gollemer's concept book — names of code mapped to their basic code. Rows are data, not grammar: add a row (id, names, kind, template) and the composer can assemble that concept in every phrasing its slots allow, with no new Go code. A malformed row is skipped; the rest of the book still loads."
}`

var (
	conceptOnce  = &sync.Once{} // a pointer so tests can swap in a fresh one
	conceptCache []analyze.ConceptEntry
)

// conceptEntries lazily loads the book: data/concepts.json from
// the chat's project root (or the working directory) first —
// the advocates precedent — else the compiled default. A book
// whose rows are all malformed degrades to the default too:
// ParseConceptBook skips bad rows, never the whole book.
func conceptEntries() []analyze.ConceptEntry {
	conceptOnce.Do(func() {
		for _, base := range []string{unifiedProjectRoot, startFolder()} {
			if base == "" {
				continue
			}
			blob, err := os.ReadFile(filepath.Join(base, "data", "concepts.json"))
			if err != nil {
				continue
			}
			if entries, _ := analyze.ParseConceptBook(blob); len(entries) > 0 {
				conceptCache = entries
				return
			}
		}
		conceptCache, _ = analyze.ParseConceptBook([]byte(defaultConceptBook))
	})
	return conceptCache
}

// conceptName is one phrasing of one entry, word-split for the
// longest-match walk.
type conceptName struct {
	words []string
	entry int // index into the loaded entries
}

// conceptDeclarationSibling finds the declaration a named
// statement ask belongs to: the unique declaration entry
// carrying, among its own names, one of the statement's names
// with " function" appended — the book's own naming carries
// the pairing ("database connection" the statement, "database
// connection function" the declaration), so a future pair
// named the same way resolves with no new code. Zero or
// several candidates means no sibling: the ask must not
// guess between declarations.
func conceptDeclarationSibling(stmt analyze.ConceptEntry, entries []analyze.ConceptEntry) (analyze.ConceptEntry, bool) {
	var found []analyze.ConceptEntry
	seen := map[string]bool{}
	for _, e := range entries {
		if e.Kind != analyze.ConceptDeclaration || seen[e.ID] {
			continue
		}
		for _, n := range e.Names {
			for _, sn := range stmt.Names {
				if n == sn+" function" {
					found = append(found, e)
					seen[e.ID] = true
				}
			}
		}
	}
	if len(found) == 1 {
		return found[0], true
	}
	return analyze.ConceptEntry{}, false
}

// conceptNames flattens the book's phrasings, longest first (so
// "print line" beats its own prefix "print"), ties broken by
// phrase for determinism.
func conceptNames(entries []analyze.ConceptEntry) []conceptName {
	var out []conceptName
	for i, e := range entries {
		for _, n := range e.Names {
			out = append(out, conceptName{words: strings.Fields(n), entry: i})
		}
	}
	sort.Slice(out, func(a, b int) bool {
		if len(out[a].words) != len(out[b].words) {
			return len(out[a].words) > len(out[b].words)
		}
		return strings.Join(out[a].words, " ") < strings.Join(out[b].words, " ")
	})
	return out
}

// conceptReq is one parsed composer sentence: the matched book
// entry and every bound slot.
type conceptReq struct {
	entry   analyze.ConceptEntry
	name    string // {name}: "named <Name>" or "name is <Name>"
	text    string // {text}: the saying slot's string value
	textSet bool
	inFunc  string // statement target function: "in <Func>"
	like    string // {methods} source type: "like <Type>"
	target  string // the file/folder as typed; "" = unnamed

	// The line phrase: "on line 12" / "before line 8",
	// riding the slot zone or the tail behind the target.
	// placeBad marks a STARTED phrase whose number is
	// missing or unusable (placeBadTok is the token as
	// typed, "" when the number never came): the sentence
	// still claims, and runConcept refuses in placement
	// terms rather than letting the line fall through.
	placeLine   int
	placeBefore bool
	placeBad    bool
	placeBadTok string
}

// placeBound reports whether a line phrase already rode this
// sentence — a second one is never this grammar's.
func (r conceptReq) placeBound() bool { return r.placeLine > 0 || r.placeBad }

// isPlacePrep reports whether w opens a line phrase.
func isPlacePrep(w string) bool {
	switch w {
	case "on", "at", "after", "before":
		return true
	}
	return false
}

// lineNumberToken reads a line phrase's number: digits only,
// counting from 1 — zero, a negative, or a word is not a
// number a line can be aimed at.
func lineNumberToken(tok string) (int, bool) {
	n, err := strconv.Atoi(tok)
	if err != nil || n < 1 {
		return 0, false
	}
	return n, true
}

var reConceptSaying = regexp.MustCompile(`(?i)\b(?:saying|that says|says)\s+"((?:[^"\\]|\\.)*)"`)

// parseConcept reads one line as a concept-book ask against the
// loaded book. The second return is false for anything outside
// the composed grammar — including every bespoke shape's
// sentence (no book name matches their object words) and any
// line whose slot zone carries a word the grammar doesn't know
// (unknown qualifiers are never silently dropped).
func parseConcept(line string) (conceptReq, bool) {
	return parseConceptFrom(line, conceptEntries())
}

// parseConceptFrom is parseConcept against a given book — the
// seam the data-driven tests drive a fixture book through.
func parseConceptFrom(line string, entries []analyze.ConceptEntry) (conceptReq, bool) {
	s := strings.TrimSpace(line)

	// The saying slot comes out first, quoted text intact: its
	// value is a Go string's content, unescaped here and
	// re-escaped by the planner's renderer.
	var req conceptReq
	if m := reConceptSaying.FindStringSubmatchIndex(s); m != nil {
		inner := s[m[2]:m[3]]
		req.text = inner
		if uq, err := strconv.Unquote(`"` + inner + `"`); err == nil {
			req.text = uq
		}
		req.textSet = true
		s = strings.TrimSpace(s[:m[0]] + " " + s[m[1]:])
	}
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return conceptReq{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	if low[0] != "add" && low[0] != "put" {
		return conceptReq{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	if i >= len(fields) {
		return conceptReq{}, false
	}

	// The concept name: the book's longest phrasing that matches
	// at this position. No match, no claim — an unknown concept
	// belongs to the boundary.
	matched := false
	for _, cn := range conceptNames(entries) {
		if i+len(cn.words) > len(fields) {
			continue
		}
		ok := true
		for j, w := range cn.words {
			if low[i+j] != w {
				ok = false
				break
			}
		}
		if ok {
			req.entry = entries[cn.entry]
			i += len(cn.words)
			matched = true
			break
		}
	}
	if !matched {
		return conceptReq{}, false
	}

	// Split what remains into the slot zone and the target
	// clause. The target starts at its preposition; an "in"
	// counts only when what follows it reads as a target (a
	// .go file, a path, or — when no later target clause
	// exists — the line's tail), so "in main to jim/main.go"
	// leaves "in main" in the slot zone.
	tIdx := -1
	for k := i; k < len(fields); k++ {
		switch low[k] {
		case "to", "into":
			tIdx = k
		case "in":
			j := k + 1
			if j < len(fields) && low[j] == "the" {
				j++
			}
			if j < len(fields) && conceptDesignator(low[j]) {
				j++
			}
			switch {
			case j >= len(fields):
				tIdx = k // "… in folder" — claimed, nameless
			case strings.HasSuffix(low[j], ".go") || strings.Contains(fields[j], "/"):
				tIdx = k
			case !conceptHasLaterTarget(low, k+1):
				tIdx = k // lone "in <word>": the target (add-server rule)
			}
		}
		if tIdx >= 0 {
			break
		}
	}
	slotEnd := len(fields)
	if tIdx >= 0 {
		slotEnd = tIdx
	}

	// The slot zone: named <Name> | name is <Name> | like
	// <Type> | in <Func>, in any order the family allows.
	// Anything else refuses the claim.
	for j := i; j < slotEnd; {
		switch low[j] {
		case "named":
			if j+1 >= slotEnd {
				return conceptReq{}, false
			}
			req.name = fields[j+1]
			j += 2
		case "name":
			// "name is <Name>": the name slot spelled out,
			// the same binding as "named", one word longer.
			if j+2 >= slotEnd || low[j+1] != "is" {
				return conceptReq{}, false
			}
			req.name = fields[j+2]
			j += 3
		case "like":
			if j+1 >= slotEnd {
				return conceptReq{}, false
			}
			req.like = fields[j+1]
			j += 2
		case "in":
			if j+1 >= slotEnd {
				return conceptReq{}, false
			}
			req.inFunc = fields[j+1]
			j += 2
		case "on", "at", "after", "before":
			// A line phrase riding the slot zone: "on
			// line 8 in main to …", "… in main before
			// line 8 to …". The concept's own name is
			// already consumed by here, so the "print
			// line" in "add a print line on line 12"
			// is never misread — the phrase starts at
			// its preposition. A preposition NOT
			// followed by "line" is no phrase at all
			// and declines, exactly as before.
			if req.placeBound() {
				return conceptReq{}, false
			}
			if j+1 >= slotEnd || low[j+1] != "line" {
				return conceptReq{}, false
			}
			req.placeBefore = low[j] == "before"
			if j+2 < slotEnd {
				if n, ok := lineNumberToken(low[j+2]); ok {
					req.placeLine = n
					j += 3
					continue
				}
				req.placeBad, req.placeBadTok = true, fields[j+2]
			} else {
				req.placeBad, req.placeBadTok = true, ""
			}
			// A broken phrase ends the slot parse: the
			// sentence still claims (runConcept refuses
			// it in placement terms), and whatever words
			// the bad number swallowed — the next slot's
			// opener, say — are part of the refusal,
			// never re-read as slots.
			j = slotEnd
		default:
			return conceptReq{}, false
		}
	}

	// The target clause: article and designator are slots, never
	// part of the path (the add-server rule). The target itself
	// is ONE path token: a longer tail is not a path with spaces
	// in it — it is another ask glued behind this one ("add a
	// for loop in main to jim/main.go and add a print line in
	// main to jim/main.go"), and joining the tail whole once let
	// a compound be claimed here and refused by the resolver as
	// a single ask naming a file nobody typed. Multi-piece
	// lines belong to the phrasing layer; this grammar declines
	// them, exactly as it declines a "that" clause in the tail.
	if tIdx >= 0 {
		j := tIdx + 1
		if j < len(fields) && low[j] == "the" {
			j++
		}
		if j < len(fields) && conceptDesignator(low[j]) {
			j++
		}
		if j < len(fields) {
			rest := fields[j:]
			req.target = rest[0]
			switch {
			case len(rest) == 1:
				// The plain target clause.
			case len(rest) >= 3 && isPlacePrep(low[j+1]) && low[j+2] == "line":
				// The line phrase rides behind the
				// target: "to jim/main.go on line 12".
				// A phrase that already rode the slot
				// zone makes this a second one — never
				// this grammar's.
				if req.placeBound() {
					return conceptReq{}, false
				}
				req.placeBefore = low[j+1] == "before"
				switch {
				case len(rest) == 3:
					req.placeBad, req.placeBadTok = true, ""
				case len(rest) == 4:
					if n, ok := lineNumberToken(low[j+3]); ok {
						req.placeLine = n
					} else {
						req.placeBad, req.placeBadTok = true, rest[3]
					}
				default:
					return conceptReq{}, false
				}
			default:
				return conceptReq{}, false
			}
		}
	}

	// Two bound slots can still make the line another shape's:
	// a "like" the entry's template can't spend (it has no
	// {methods} slot) would be silently dropped — and "add a
	// function like Greet named GreetLoudly" is the CLONE's
	// sentence, not an empty scaffold's. Likewise a "that" in
	// the target tail is a behaviour clause, not a path: "add
	// a function in jim/main.go that calls Load" is the
	// DELEGATE's sentence, and its tail is not a file. Both
	// refuse the claim; qualifiers are never silently dropped
	// and behaviour clauses are never read as paths.
	if req.like != "" && !strings.Contains(req.entry.Template, "{methods}") {
		return conceptReq{}, false
	}
	for _, w := range strings.Fields(strings.ToLower(req.target)) {
		if w == "that" {
			return conceptReq{}, false
		}
	}
	return req, true
}

// conceptHasLaterTarget reports whether a target clause starts
// somewhere at or after index from: a to|into, or an "in" whose
// following word is a .go file or a path.
func conceptHasLaterTarget(low []string, from int) bool {
	for k := from; k < len(low); k++ {
		if low[k] == "to" || low[k] == "into" {
			return true
		}
		if low[k] == "in" && k+1 < len(low) &&
			(strings.HasSuffix(low[k+1], ".go") || strings.Contains(low[k+1], "/")) {
			return true
		}
	}
	return false
}

// conceptArticle picks a/an for a concept's canonical name.
func conceptArticle(name string) string {
	if name != "" && strings.ContainsRune("aeiou", rune(name[0])) {
		return "an"
	}
	return "a"
}

// conceptDesignator reports whether w is a target kind word
// (file|folder|directory, a trailing slash tolerated).
func conceptDesignator(w string) bool {
	switch strings.TrimSuffix(w, "/") {
	case "file", "folder", "directory":
		return true
	}
	return false
}

// tryConceptBook is the chat-loop entry: it claims concept-book
// sentences and runs the flow on the chat's line reader.
func tryConceptBook(line string, r *LineReader) bool {
	return runConcept(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runConcept carries one concept sentence from the book match
// to the shared edit spine (or to composer's guidance when the
// sentence is missing a piece no plan can supply).
func runConcept(line string, ask editAsk) bool {
	req, ok := parseConcept(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	canon := req.entry.Canonical()

	// A started line phrase with an unusable number refuses
	// here, in placement terms, before any other guidance:
	// the sentence WAS this claim's, and a refusal about the
	// missing target or name would answer a question the
	// user didn't ask.
	if req.placeBad {
		var why string
		switch {
		case req.placeBadTok == "":
			why = "the sentence names a line but never gives its number — write it as digits, like \"on line 12\""
		default:
			if _, err := strconv.Atoi(req.placeBadTok); err == nil {
				why = fmt.Sprintf("\"line %s\" doesn't pick a line — a file's lines count from 1, so the first line is line 1. Say \"on line 1\" or a higher number", req.placeBadTok)
			} else {
				why = fmt.Sprintf("\"line %s\" isn't a number I can aim at — write the line number as digits, like \"on line 12\"", req.placeBadTok)
			}
		}
		fmt.Printf("%s Not adding that: %s.\n", tag, why)
		recordBrain("composer", false, "concept refused: line phrase")
		return true
	}

	// A name belongs to a declaration (named-connection
	// build, 2026-11-01). A statement matched with a
	// "named <Name>" slot has nowhere to put the name: the
	// statement lands inside a function, unnamed by
	// construction, and the name used to drop in silence —
	// "add a database connection named Connet to jake/db.go"
	// assembled the inline sql.Open snippet into a file with
	// no main and fabricated one to hold it, when the file
	// was exactly where a connection FUNCTION belonged.
	// Two honest outcomes replace the drop:
	//   - the book's declaration sibling takes the ask
	//     whole (conceptDeclarationSibling): the entry is
	//     rebound and the ask flows on through the
	//     declaration's own plan, gate, transaction and
	//     build check, exactly as its own sentences do;
	//   - otherwise the ask refuses here, before any
	//     project work: the piece is a statement inside a
	//     function and takes no name, and the guidance
	//     spells the same ask unnamed, placed in a
	//     function — the user's function and target when
	//     they gave them, the statement's defaults (main,
	//     main.go) when they didn't.
	if req.entry.Kind == analyze.ConceptStatement && req.name != "" {
		if sibling, ok := conceptDeclarationSibling(req.entry, conceptEntries()); ok {
			req.entry = sibling
			canon = req.entry.Canonical()
		} else {
			fn := req.inFunc
			if fn == "" {
				fn = "main"
			}
			target := req.target
			if target == "" {
				target = "main.go"
			}
			fmt.Printf("%s Not adding that: %s %s is a statement — a piece of code that lives inside a function — so it doesn't take a name of its own, and %q would be lost. Say \"add %s %s in %s to %s\" and it lands inside %s, unnamed.\n",
				tag, conceptArticle(canon), canon, req.name,
				conceptArticle(canon), canon, fn, target, fn)
			recordBrain("composer", false, "concept refused: statement named")
			return true
		}
	}

	// Guidance before any project work: a declaration with no
	// name, and any concept with no target, are answered in
	// composer terms — the sentence was this claim's.
	if req.entry.Kind == analyze.ConceptDeclaration && req.name == "" {
		if req.entry.ID == "function" {
			// A nameless function ask is a near-miss of THREE
			// shapes, not one: this claim's scaffold, the
			// delegate, and the clone. The guidance names all
			// three, with the user's own target filled in.
			file := req.target
			if file == "" {
				file = "main.go"
			}
			fmt.Printf("%s Not adding that: a function needs a name — say \"add a function named <Name> to %s\" for an empty scaffold you fill in. If you want it doing real work from the start, say \"add a function in %s that calls <something that exists>\" or \"copy <existing function> as <new name>\".\n", tag, file, file)
			recordBrain("composer", false, "concept refused: function unnamed")
			return true
		}
		if req.entry.Folder != "" {
			// A folder-mapped row's unnamed ask gets the
			// folder form of the guidance: the name is the
			// only missing piece.
			fmt.Printf("%s Not adding that: %s %s needs a name — say \"add a %s named <Name>\" and it lands in the %s folder as its own file.\n", tag, conceptArticle(canon), canon, canon, req.entry.Folder)
			recordBrain("composer", false, "concept refused: declaration unnamed")
			return true
		}
		fmt.Printf("%s Not adding that: %s %s needs a name — say \"add a %s named <Name> to <file>\", like \"add a %s named Reader to main.go\".\n", tag, conceptArticle(canon), canon, canon, canon)
		recordBrain("composer", false, "concept refused: declaration unnamed")
		return true
	}
	// The folder default: a declaration whose book row carries
	// a family folder, asked with NO file or folder named,
	// lands in that folder as its own package — the planner
	// creates <folder>/<name>.go. A named target always wins:
	// this branch is reachable only when the sentence named
	// nothing. Statement rows never take it (they land inside
	// functions), and neither do declarations whose rows carry
	// no folder.
	folderDefault := req.target == "" && req.entry.Kind == analyze.ConceptDeclaration && req.entry.Folder != ""
	if req.target == "" && !folderDefault {
		fmt.Printf("%s Not adding that: the sentence doesn't name the file or folder — say it like \"add a %s to jim/main.go\" (a file) or \"add a %s to jim\" (its folder).\n", tag, canon, canon)
		recordBrain("composer", false, "concept refused: no target named")
		return true
	}

	// Target resolution and project acquisition: the
	// add-server flow's own rules, verbatim in spirit — the
	// creations fence's root, its folder-to-file resolver, its
	// quiet analyze / freshness guard.
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "concept refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "concept refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "concept refused: root missing")
		return true
	}
	// The gorqlite dependency gate (database build,
	// 2026-11-01): a book row importing the gorqlite client
	// can only plan in a project that already requires the
	// module — code naming the import cannot compile without
	// it, and the chat never fetches dependencies. Absent,
	// no plan is offered: the claim answers with the exact
	// go get line and holds the ask in the session until the
	// user says it is installed. See gorqlitegate.go.
	if gorqliteGated(req.entry) && !goModRequires(rootAbs, gorqliteModule) {
		gorqliteDepMissing(req, rootAbs, line)
		return true
	}
	rel := ""
	if !folderDefault {
		// A folder-default ask has no target to resolve: the
		// planner derives the file from the row's folder and
		// the declaration's name. A declaration resolves its
		// target through the declaration resolver: a named
		// file that doesn't exist yet, in a folder that does,
		// is created by the plan (resolveConceptDeclTarget).
		// Statements keep resolveServerFile exactly — a
		// statement lives inside a function in a file that
		// exists.
		var refusal string
		if req.entry.Kind == analyze.ConceptDeclaration {
			rel, refusal = resolveConceptDeclTarget(rootAbs, req.target)
		} else {
			rel, refusal = resolveServerFile(rootAbs, req.target)
		}
		if refusal != "" {
			fmt.Printf("%s Not adding that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "concept refused: target")
			return true
		}
	}
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		if !analyzeQuiet(rootAbs) {
			recordBrain("composer", false, "concept refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "concept refused: stale project unreadable")
		return true
	}
	entry := req.entry
	folder := ""
	if folderDefault {
		folder = entry.Folder
	}
	// A statement that stays a statement carries its book
	// sibling into the planner: if the target file declares
	// no functions, the planner's missing-host refusal spells
	// the sibling's declaration sentence — the one sentence
	// that can land in such a file — instead of naming a
	// function the file doesn't have. (When the named-ask
	// branch above rebound the entry to the declaration
	// itself, the entry is no longer a statement and this
	// stays nil: the planner reads the field for statements
	// only.)
	var sibling *analyze.ConceptEntry
	if entry.Kind == analyze.ConceptStatement {
		if sib, ok := conceptDeclarationSibling(entry, conceptEntries()); ok {
			sibling = &sib
		}
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:           analyze.EditConcept,
		File:           rel,
		Concept:        &entry,
		ConceptName:    req.name,
		ConceptText:    req.text,
		ConceptFunc:    req.inFunc,
		ConceptLike:    req.like,
		ConceptFolder:  folder,
		ConceptSibling: sibling,
		PlaceLine:      req.placeLine,
		PlaceBefore:    req.placeBefore,
	}, ask)
}

// resolveConceptDeclTarget settles a concept DECLARATION's
// named target. Declarations are the one claim whose target
// file may be new: folder-default declarations create theirs
// (handlers/signup.go, database/connect.go), and the
// serve-html recipe creates its named program file — but a
// declaration that NAMED its file was refused by
// resolveServerFile with the webserver claim's remedy ("say
// \"create a file api/profile.go with a webserver\""), which
// is the wrong piece for a struct or a connect function and,
// inside a package-main folder, would write a second func
// main. The rules here:
//
//   - an existing file, a folder target, and every non-.go
//     shape resolve exactly as resolveServerFile resolves
//     them today (this function delegates);
//   - a .go file that doesn't exist yet, in a folder that
//     does, resolves to itself — the planner creates it as
//     part of the plan (planConceptNamedFileCreate), under
//     the bare-file create family's package rules;
//   - a .go file whose FOLDER doesn't exist refuses with
//     the create family's real first step — "create a folder
//     <folder> first" — never the webserver remedy;
//   - a file standing where the folder should be refuses in
//     the create family's words too.
func resolveConceptDeclTarget(rootAbs, target string) (rel, refusal string) {
	disp := strings.TrimSuffix(target, "/")
	if !strings.HasSuffix(strings.ToLower(disp), ".go") {
		return resolveServerFile(rootAbs, target)
	}
	abs, rel2, refusal2 := serverTargetResolve(rootAbs, target)
	if refusal2 != "" {
		return "", refusal2
	}
	if st, err := os.Stat(abs); err == nil {
		if st.IsDir() {
			// A folder that happens to end in .go: the
			// folder rules decide, as today.
			return resolveServerFile(rootAbs, target)
		}
		return rel2, ""
	}
	// The file is new; its folder decides.
	dirAbs := filepath.Dir(abs)
	dst, derr := os.Stat(dirAbs)
	if derr == nil && !dst.IsDir() {
		locRel := filepath.ToSlash(mustRel(rootAbs, dirAbs))
		return "", fmt.Sprintf("%s is a file, not a folder — the new file can't go inside it", locRel)
	}
	if derr != nil {
		folder := filepath.ToSlash(mustRel(rootAbs, dirAbs))
		return "", fmt.Sprintf("there's no folder %s in this project — say \"create a folder %s\" first, then ask again", folder, folder)
	}
	return rel2, ""
}
