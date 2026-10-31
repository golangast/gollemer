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
// "saying" may also be spelled "that says". The "in <Func>"
// qualifier is read only when a separate target clause follows
// it — a lone "in <word>" is the target, exactly as the
// add-server grammar reads it. A concept matched with no target
// (or a declaration with no name) is still claimed: the composer
// answers with the guidance naming the working sentence, in the
// nameless-designator tradition. Unknown concept names are NOT
// claimed here at all — they fall to the boundary.

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
		default:
			return conceptReq{}, false
		}
	}

	// The target clause: article and designator are slots, never
	// part of the path (the add-server rule).
	if tIdx >= 0 {
		j := tIdx + 1
		if j < len(fields) && low[j] == "the" {
			j++
		}
		if j < len(fields) && conceptDesignator(low[j]) {
			j++
		}
		if j < len(fields) {
			req.target = strings.Join(fields[j:], " ")
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
		fmt.Printf("%s Not adding that: %s %s needs a name — say \"add a %s named <Name> to <file>\", like \"add a %s named Reader to main.go\".\n", tag, conceptArticle(canon), canon, canon, canon)
		recordBrain("composer", false, "concept refused: declaration unnamed")
		return true
	}
	if req.target == "" {
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
	rel, refusal := resolveServerFile(rootAbs, req.target)
	if refusal != "" {
		fmt.Printf("%s Not adding that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "concept refused: target")
		return true
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
	return executeEditPlan(analyze.EditRequest{
		Kind:        analyze.EditConcept,
		File:        rel,
		Concept:     &entry,
		ConceptName: req.name,
		ConceptText: req.text,
		ConceptFunc: req.inFunc,
		ConceptLike: req.like,
	}, ask)
}
