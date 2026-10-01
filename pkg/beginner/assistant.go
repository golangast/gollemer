// Package beginner translates simple natural-language requests into
// formatted Go code with plain-English explanations, explains core Go
// concepts with analogies and runnable snippets, and maps code
// snippets to visual execution sequences.
//
// Everything is deterministic and standard library only: requests are
// matched against hand-written idiomatic templates (each verified to
// compile), formatted with go/format, and paired with explanations of
// why each construct was chosen.
package beginner

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// recipe is one generated program plus its explanation.
type recipe struct {
	code        string // raw Go source; formatted with go/format before return
	explanation string // 2-3 sentences on why these constructs were used
}

// intent matches a normalized prompt against keyword groups.
type intent struct {
	name        string
	description string // shown in the "I can help with" list
	groups      [][]string
	build       func(prompt string) recipe
}

var (
	portRe    = regexp.MustCompile(`(?i)\bport\s+(\d{1,5})\b`)
	workersRe = regexp.MustCompile(`\b(\d{1,3})\s+workers?\b`)
	quotedRe  = regexp.MustCompile(`["']([^"']+)["']`)
	safeName  = regexp.MustCompile(`^[A-Za-z0-9_./-]+$`)
)

var intents = []intent{
	{
		name:        "worker pool",
		description: "worker pools with channels and WaitGroup",
		groups:      [][]string{{"worker", "workers"}, {"pool"}},
		build:       buildWorkerPool,
	},
	{
		name:        "mutex counter",
		description: "safe concurrent counters with a mutex",
		groups:      [][]string{{"mutex"}, {"counter", "count"}},
		build:       buildMutexCounter,
	},
	{
		name:        "read file lines",
		description: "reading a file line by line safely",
		groups:      [][]string{{"read"}, {"file"}, {"line", "lines"}},
		build:       buildReadFileLines,
	},
	{
		name:        "write file",
		description: "writing data to a file",
		groups:      [][]string{{"write"}, {"file"}},
		build:       buildWriteFile,
	},
	{
		name:        "http server",
		description: "HTTP servers with handlers",
		groups:      [][]string{{"http", "web"}, {"server"}},
		build:       buildHTTPServer,
	},
	{
		name:        "json",
		description: "encoding and decoding JSON",
		groups:      [][]string{{"json"}},
		build:       buildJSON,
	},
	{
		name:        "context timeout",
		description: "contexts with timeouts",
		groups:      [][]string{{"context"}, {"timeout"}},
		build:       buildContextTimeout,
	},
	{
		name:        "string builder",
		description: "building strings efficiently",
		groups:      [][]string{{"string"}, {"build", "concat", "join", "efficient"}},
		build:       buildStringBuilder,
	},
	{
		name:        "sort slice",
		description: "sorting slices",
		groups:      [][]string{{"sort"}, {"slice", "list"}},
		build:       buildSortSlice,
	},
	{
		name:        "ticker",
		description: "repeating work on a ticker",
		groups:      [][]string{{"ticker"}},
		build:       buildTicker,
	},
	{
		name:        "ticker",
		description: "repeating work on a ticker",
		groups:      [][]string{{"every"}, {"second", "seconds", "minute", "interval", "tick"}},
		build:       buildTicker,
	},
	{
		name:        "cli args",
		description: "reading command-line arguments",
		groups:      [][]string{{"command", "cli"}, {"arg", "args", "argument", "arguments"}},
		build:       buildCLIArgs,
	},
	{
		name:        "env var",
		description: "reading environment variables",
		groups:      [][]string{{"environment", "env"}, {"variable", "var"}},
		build:       buildEnvVar,
	},
}

// GenerateGoFromCommand translates a simple natural-language request
// into formatted, compilable Go code plus a plain-English explanation
// of why each construct was chosen. It returns an error (listing what
// it can build) when the request matches no known intent.
func GenerateGoFromCommand(prompt string) (formattedCode string, explanation string, err error) {
	normalized := strings.ToLower(prompt)
	for _, in := range intents {
		if matchGroups(normalized, in.groups) {
			r := in.build(prompt)
			formatted, ferr := format.Source([]byte(r.code))
			if ferr != nil {
				return "", "", fmt.Errorf("beginner: internal template error for %q: %w", in.name, ferr)
			}
			return string(formatted), r.explanation, nil
		}
	}
	var can []string
	seen := map[string]bool{}
	for _, in := range intents {
		if !seen[in.description] {
			seen[in.description] = true
			can = append(can, in.description)
		}
	}
	return "", "", fmt.Errorf("beginner: I don't know how to build that yet; I can help with: %s", strings.Join(can, "; "))
}

func matchGroups(normalized string, groups [][]string) bool {
	for _, group := range groups {
		hit := false
		for _, word := range group {
			if strings.Contains(normalized, word) {
				hit = true
				break
			}
		}
		if !hit {
			return false
		}
	}
	return true
}

// extractPort finds "port NNNN" in the prompt; default 8080.
func extractPort(prompt string) int {
	if m := portRe.FindStringSubmatch(prompt); m != nil {
		if p, err := strconv.Atoi(m[1]); err == nil && p >= 1 && p <= 65535 {
			return p
		}
	}
	return 8080
}

// extractWorkers finds "N workers" in the prompt; default 3.
func extractWorkers(prompt string) int {
	if m := workersRe.FindStringSubmatch(prompt); m != nil {
		if w, err := strconv.Atoi(m[1]); err == nil && w >= 1 && w <= 64 {
			return w
		}
	}
	return 3
}

// extractFilename finds a quoted filename in the prompt; default def.
// Only safe characters are accepted, so the value can be embedded in
// a Go string literal.
func extractFilename(prompt, def string) string {
	if m := quotedRe.FindStringSubmatch(prompt); m != nil && safeName.MatchString(m[1]) {
		return m[1]
	}
	return def
}

func buildHTTPServer(prompt string) recipe {
	port := extractPort(prompt)
	lower := strings.ToLower(prompt)
	health := strings.Contains(lower, "health")
	healthRoute := ""
	if health {
		if strings.Contains(lower, "json") {
			healthRoute = `
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, ` + "`{\"status\":\"ok\"}`" + `)
	})`
		} else {
			healthRoute = `
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "ok")
	})`
		}
	}
	code := fmt.Sprintf(`package main

import (
	"fmt"
	"net/http"
)

func main() {
	mux := http.NewServeMux()
%s
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "hello from gollemer")
	})
	addr := ":%d"
	fmt.Println("listening on", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Println("server error:", err)
	}
}
`, healthRoute, port)
	expl := "We use net/http's ServeMux because it routes each URL path to its own handler function, keeping endpoints isolated and testable. " +
		"ListenAndServe blocks forever accepting connections, so we check its returned error — a server that can't bind its port should fail loudly instead of dying silently."
	if health {
		expl += " The /healthz endpoint exists so load balancers and monitoring can ask 'are you alive?' without touching real traffic."
	}
	return recipe{code: code, explanation: expl}
}

func buildReadFileLines(prompt string) recipe {
	filename := extractFilename(prompt, "input.txt")
	code := fmt.Sprintf(`package main

import (
	"bufio"
	"fmt"
	"os"
)

func readLines(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var lines []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return lines, nil
}

func main() {
	lines, err := readLines(%q)
	if err != nil {
		fmt.Println("read error:", err)
		return
	}
	for _, line := range lines {
		fmt.Println(line)
	}
}
`, filename)
	expl := "defer f.Close() guarantees the file descriptor is released when readLines returns, on every path including errors — leaked descriptors eventually crash long-running programs. " +
		"We check errors in two places because opening can fail (missing file) and reading can fail halfway through (disk error), and bufio.Scanner only reveals read errors when you ask via scanner.Err()."
	return recipe{code: code, explanation: expl}
}

func buildWorkerPool(prompt string) recipe {
	workers := extractWorkers(prompt)
	code := fmt.Sprintf(`package main

import (
	"fmt"
	"sync"
)

func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {
	defer wg.Done()
	for j := range jobs {
		results <- j * 2
	}
}

func main() {
	const numJobs = 10
	jobs := make(chan int, numJobs)
	results := make(chan int, numJobs)

	var wg sync.WaitGroup
	for w := 1; w <= %d; w++ {
		wg.Add(1)
		go worker(w, jobs, results, &wg)
	}
	for j := 1; j <= numJobs; j++ {
		jobs <- j
	}
	close(jobs)
	wg.Wait()
	close(results)
	for r := range results {
		fmt.Println(r)
	}
}
`, workers)
	expl := "A WaitGroup lets main block on wg.Wait() until every worker finishes — without it the program could exit while jobs are still running. " +
		"Closing the jobs channel tells workers no more work is coming so their range loops end, and results is closed only after wg.Wait() so the final draining loop terminates instead of deadlocking."
	return recipe{code: code, explanation: expl}
}

func buildWriteFile(prompt string) recipe {
	filename := extractFilename(prompt, "output.txt")
	code := fmt.Sprintf(`package main

import (
	"fmt"
	"os"
)

func main() {
	data := []byte("hello, file\n")
	if err := os.WriteFile(%q, data, 0644); err != nil {
		fmt.Println("write error:", err)
		return
	}
	fmt.Println("wrote %s")
}
`, filename, filename)
	expl := "os.WriteFile handles open, write, and close in one call, which removes the usual places file code goes wrong. " +
		"We still check its error because disks fill up and permissions deny writes — silent failures here mean lost data."
	return recipe{code: code, explanation: expl}
}

func buildJSON(prompt string) recipe {
	_ = prompt
	code := `package main

import (
	"encoding/json"
	"fmt"
)

type User struct {
	Name string ` + "`json:\"name\"`" + `
	Age  int    ` + "`json:\"age\"`" + `
}

func main() {
	u := User{Name: "gopher", Age: 5}
	data, err := json.Marshal(u)
	if err != nil {
		fmt.Println("encode error:", err)
		return
	}
	fmt.Println(string(data))

	var back User
	if err := json.Unmarshal(data, &back); err != nil {
		fmt.Println("decode error:", err)
		return
	}
	fmt.Printf("%+v\n", back)
}
`
	expl := "Struct tags like `json:\"name\"` control the JSON field names, decoupling your Go naming from the wire format. " +
		"Both Marshal and Unmarshal return errors because not every value can be encoded and not every input is valid — checking them keeps corrupt data from spreading silently."
	return recipe{code: code, explanation: expl}
}

func buildMutexCounter(prompt string) recipe {
	_ = prompt
	code := `package main

import (
	"fmt"
	"sync"
)

func main() {
	var mu sync.Mutex
	count := 0
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mu.Lock()
			count++
			mu.Unlock()
		}()
	}
	wg.Wait()
	fmt.Println("count:", count)
}
`
	expl := "A Mutex serializes access to count so only one goroutine increments at a time; without it two goroutines could read-modify-write simultaneously and lose updates. " +
		"Each worker calls wg.Done() via defer so the WaitGroup is released even if the goroutine panics."
	return recipe{code: code, explanation: expl}
}

func buildCLIArgs(prompt string) recipe {
	_ = prompt
	code := `package main

import (
	"fmt"
	"os"
)

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Println("usage: greet <name>")
		return
	}
	fmt.Println("hello,", args[0])
}
`
	expl := "os.Args holds the command-line words with the program name first, so real arguments start at index 1. " +
		"We check the length before indexing because accessing a missing element panics — a usage message is friendlier than a crash."
	return recipe{code: code, explanation: expl}
}

func buildEnvVar(prompt string) recipe {
	_ = prompt
	code := `package main

import (
	"fmt"
	"os"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	fmt.Println("port:", port)
}
`
	expl := "os.Getenv reads configuration from the environment, which keeps secrets and ports out of source code. " +
		"We fall back to a default when PORT is unset so the program still runs locally without any setup."
	return recipe{code: code, explanation: expl}
}

func buildSortSlice(prompt string) recipe {
	_ = prompt
	code := `package main

import (
	"fmt"
	"sort"
)

func main() {
	names := []string{"gopher", "alice", "bob"}
	sort.Strings(names)
	fmt.Println(names)
}
`
	expl := "sort.Strings sorts the slice in place, which avoids allocating a second copy. " +
		"It only works because the sort package already knows how to order strings — for custom types you'd pass a 'less' function instead."
	return recipe{code: code, explanation: expl}
}

func buildStringBuilder(prompt string) recipe {
	_ = prompt
	code := `package main

import (
	"fmt"
	"strings"
)

func main() {
	var b strings.Builder
	for i := 0; i < 3; i++ {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	fmt.Print(b.String())
}
`
	expl := "strings.Builder accumulates text without creating a new string on every +=, which would copy all previous content each time. " +
		"fmt.Fprintf writes formatted text directly into the builder, and String() hands you the final result once."
	return recipe{code: code, explanation: expl}
}

func buildContextTimeout(prompt string) recipe {
	_ = prompt
	code := `package main

import (
	"context"
	"fmt"
	"time"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	select {
	case <-time.After(1 * time.Second):
		fmt.Println("finished in time")
	case <-ctx.Done():
		fmt.Println("timed out:", ctx.Err())
	}
}
`
	expl := "context.WithTimeout starts a countdown, and cancel (deferred so it always runs) releases the timer's resources. " +
		"The select waits on both the work and ctx.Done(), so a slow operation can't hang the program past its deadline."
	return recipe{code: code, explanation: expl}
}

func buildTicker(prompt string) recipe {
	_ = prompt
	code := `package main

import (
	"fmt"
	"time"
)

func main() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for i := 0; i < 3; i++ {
		<-ticker.C
		fmt.Println("tick", i)
	}
}
`
	expl := "time.NewTicker delivers a tick on a channel at a steady interval, which is simpler and more accurate than sleeping in a loop. " +
		"We defer ticker.Stop() because a forgotten ticker keeps a goroutine alive for the life of the program."
	return recipe{code: code, explanation: expl}
}

// ---------------------------------------------------------------------------
// Concept explainer
// ---------------------------------------------------------------------------

// ConceptExplanation is a beginner-friendly tour of one Go concept.
type ConceptExplanation struct {
	Concept  string   `json:"concept"`
	Analogy  string   `json:"analogy"`
	Snippet  string   `json:"snippet"`
	KeyRules []string `json:"keyRules"`
}

var concepts = map[string]ConceptExplanation{
	"interfaces": {
		Concept: "Interfaces",
		Analogy: "A wall outlet — it defines the plug's shape, not the appliance. Anything shaped right just works.",
		Snippet: `package main

import "fmt"

type Speaker interface {
	Speak() string
}

type Dog struct{ Name string }

func (d Dog) Speak() string { return d.Name + " says woof" }

func announce(s Speaker) {
	fmt.Println(s.Speak())
}

func main() {
	announce(Dog{Name: "Rex"})
}
`,
		KeyRules: []string{
			"A type implements an interface automatically by having all its methods — there is no 'implements' keyword.",
			"Accept interfaces as parameters, return concrete structs.",
			"Keep interfaces small; one or two methods is often enough.",
		},
	},
	"goroutines": {
		Concept: "Goroutines",
		Analogy: "Hiring an extra cook — the kitchen keeps working while they chop vegetables in the background.",
		Snippet: `package main

import (
	"fmt"
	"time"
)

func chop(veg string) {
	fmt.Println("chopping", veg)
}

func main() {
	go chop("carrots")
	go chop("onions")
	time.Sleep(100 * time.Millisecond)
	fmt.Println("dinner is served")
}
`,
		KeyRules: []string{
			"When main returns, the program exits — even if goroutines are still running.",
			"Use channels or a sync.WaitGroup to wait for goroutines to finish.",
			"Don't communicate by sharing memory; share memory by communicating (or guard it with a mutex).",
		},
	},
	"channels": {
		Concept: "Channels",
		Analogy: "A conveyor belt between workers — items go in one end and come out the other, in order.",
		Snippet: `package main

import "fmt"

func main() {
	ch := make(chan string)
	go func() {
		ch <- "hello"
	}()
	msg := <-ch
	fmt.Println(msg)
}
`,
		KeyRules: []string{
			"Sends on an unbuffered channel block until someone receives.",
			"Close a channel when no more values are coming; never send on a closed channel.",
			"Use 'for v := range ch' to drain values until the channel is closed.",
		},
	},
	"error handling": {
		Concept: "Error handling",
		Analogy: "A receipt that says what went wrong — you must read it before continuing with your day.",
		Snippet: `package main

import (
	"errors"
	"fmt"
)

func divide(a, b float64) (float64, error) {
	if b == 0 {
		return 0, errors.New("cannot divide by zero")
	}
	return a / b, nil
}

func main() {
	result, err := divide(10, 0)
	if err != nil {
		fmt.Println("oops:", err)
		return
	}
	fmt.Println(result)
}
`,
		KeyRules: []string{
			"Check every error where it happens; don't discard with '_' unless you truly don't care.",
			"Return errors to the caller instead of panicking inside libraries.",
			"Add context when wrapping: fmt.Errorf(\"reading config: %w\", err).",
		},
	},
	"structs": {
		Concept: "Structs",
		Analogy: "A form with labeled fields — one form per thing, filled out per instance.",
		Snippet: `package main

import "fmt"

type Person struct {
	Name string
	Age  int
}

func main() {
	p := Person{Name: "Ada", Age: 36}
	fmt.Println(p.Name, "is", p.Age)
}
`,
		KeyRules: []string{
			"Field names starting with a capital letter are exported (visible outside the package).",
			"Use struct tags to control JSON, database, or config mapping.",
			"Prefer small structs; compose big ones from small ones.",
		},
	},
	"slices": {
		Concept: "Slices",
		Analogy: "A stretchy tray — it grows with append and always knows how many items it holds.",
		Snippet: `package main

import "fmt"

func main() {
	tray := []string{"apple"}
	tray = append(tray, "banana", "cherry")
	fmt.Println(len(tray), tray)
}
`,
		KeyRules: []string{
			"Always use the result of append — it may return a new backing array.",
			"Slices share their backing array; copying a slice header does not copy the items.",
			"len is the item count, cap is the room before the next reallocation.",
		},
	},
	"maps": {
		Concept: "Maps",
		Analogy: "A labeled drawer cabinet — ask for a label, get the drawer; ask for a missing label, get an empty drawer.",
		Snippet: `package main

import "fmt"

func main() {
	scores := map[string]int{"ada": 95}
	scores["grace"] = 100
	v, ok := scores["alan"]
	fmt.Println(scores["ada"], v, ok)
}
`,
		KeyRules: []string{
			"Reading a missing key returns the zero value — use the comma-ok form to tell.",
			"Writing to a nil map panics; initialize with make or a literal first.",
			"Map iteration order is random; sort the keys if you need order.",
		},
	},
	"pointers": {
		Concept: "Pointers",
		Analogy: "A sticky note with a house's address — hand someone the note and they can repaint the actual house.",
		Snippet: `package main

import "fmt"

func repaint(house *string) {
	*house = "blue"
}

func main() {
	color := "red"
	repaint(&color)
	fmt.Println(color)
}
`,
		KeyRules: []string{
			"&x takes the address, *p follows it to the value.",
			"A nil pointer dereference panics — check for nil when a pointer might be absent.",
			"Use pointer receivers when a method needs to modify its struct.",
		},
	},
	"defer": {
		Concept: "Defer",
		Analogy: "A promise to tidy up that runs automatically when the function walks out the door.",
		Snippet: `package main

import (
	"fmt"
	"os"
)

func main() {
	f, err := os.Create("note.txt")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer f.Close()
	defer os.Remove("note.txt")
	fmt.Fprintln(f, "temporary")
	fmt.Println("wrote note.txt")
}
`,
		KeyRules: []string{
			"Deferred calls run in last-in-first-out order when the surrounding function returns.",
			"Defer resource cleanup (Close, Unlock, cancel) right after acquiring the resource.",
			"The arguments to a deferred call are evaluated immediately, not at return time.",
		},
	},
	"methods": {
		Concept: "Methods",
		Analogy: "A recipe card taped to one specific appliance — only that appliance can follow it.",
		Snippet: `package main

import "fmt"

type Counter struct{ n int }

func (c *Counter) Inc() { c.n++ }

func main() {
	c := &Counter{}
	c.Inc()
	c.Inc()
	fmt.Println(c.n)
}
`,
		KeyRules: []string{
			"Pointer receivers can modify the struct; value receivers work on a copy.",
			"Be consistent: if one method needs a pointer receiver, they all should.",
			"The receiver name is usually a short abbreviation of the type.",
		},
	},
	"packages": {
		Concept: "Packages",
		Analogy: "A labeled toolbox — related tools live together, and other toolboxes can borrow them.",
		Snippet: `package main

import (
	"fmt"
	"strings"
)

func main() {
	fmt.Println(strings.ToUpper("toolbox"))
}
`,
		KeyRules: []string{
			"Every Go file starts with a package clause; main packages build into programs.",
			"Import paths are quoted strings; unused imports are a compile error.",
			"Only capitalized names are visible to other packages.",
		},
	},
	"context": {
		Concept: "Context",
		Analogy: "A walkie-talkie passed down the call chain — anyone can announce 'stop, the caller gave up.'",
		Snippet: `package main

import (
	"context"
	"fmt"
	"time"
)

func work(ctx context.Context) {
	select {
	case <-time.After(5 * time.Second):
		fmt.Println("done")
	case <-ctx.Done():
		fmt.Println("cancelled")
	}
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	work(ctx)
}
`,
		KeyRules: []string{
			"Always call the cancel function, usually with defer right away.",
			"Pass ctx as the first parameter; never store it in a struct.",
			"Use timeouts and deadlines so a stuck dependency can't hang you forever.",
		},
	},
}

var conceptAliases = map[string]string{
	"goroutine": "goroutines",
	"channel":   "channels",
	"chan":      "channels",
	"error":     "error handling",
	"errors":    "error handling",
	"struct":    "structs",
	"slice":     "slices",
	"map":       "maps",
	"pointer":   "pointers",
	"method":    "methods",
	"package":   "packages",
	"ctx":       "context",
	"interface": "interfaces",
}

// ExplainGoConcept returns a beginner-friendly tour of one Go concept:
// analogy, runnable snippet, and key rules. Lookup is case-insensitive
// and understands common singular/plural forms.
func ExplainGoConcept(conceptName string) (*ConceptExplanation, error) {
	key := strings.ToLower(strings.TrimSpace(conceptName))
	if alias, ok := conceptAliases[key]; ok {
		key = alias
	}
	c, ok := concepts[key]
	if !ok {
		names := make([]string, 0, len(concepts))
		for name := range concepts {
			names = append(names, name)
		}
		sort.Strings(names)
		return nil, fmt.Errorf("beginner: unknown concept %q; I can explain: %s", conceptName, strings.Join(names, ", "))
	}
	formatted, err := format.Source([]byte(c.Snippet))
	if err != nil {
		return nil, fmt.Errorf("beginner: internal snippet error for %q: %w", key, err)
	}
	out := c
	out.Snippet = string(formatted)
	return &out, nil
}

// conceptNames returns the sorted list of known concept keys.
func conceptNames() []string {
	names := make([]string, 0, len(concepts))
	for name := range concepts {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// ---------------------------------------------------------------------------
// Visual map builder
// ---------------------------------------------------------------------------

// FuncCallInfo describes one declared function and what it calls.
type FuncCallInfo struct {
	Name  string   `json:"name"`
	Calls []string `json:"calls"`
	Line  int      `json:"line"`
}

// VisualMap is a structural map of a Go snippet: its imports, structs,
// functions with their call lists, and an ordered execution sequence
// starting from main (or the first function).
type VisualMap struct {
	Imports   []string       `json:"imports"`
	Structs   []string       `json:"structs"`
	Functions []FuncCallInfo `json:"functions"`
	CallSteps []string       `json:"callSteps"`
	Summary   string         `json:"summary"`
}

// BuildVisualMap parses a Go snippet (a full file or a fragment —
// fragments are wrapped in "package main") and maps its imports,
// structs, functions, and the call sequence starting from main.
func BuildVisualMap(codeSnippet string) (*VisualMap, error) {
	src := strings.TrimSpace(codeSnippet)
	if src == "" {
		return nil, fmt.Errorf("beginner: empty code snippet")
	}
	if !strings.HasPrefix(src, "package ") {
		src = "package main\n" + src
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "snippet.go", src, 0)
	if err != nil {
		return nil, fmt.Errorf("beginner: could not parse Go code: %w", err)
	}

	vm := &VisualMap{}
	for _, imp := range f.Imports {
		if p, uerr := strconv.Unquote(imp.Path.Value); uerr == nil {
			vm.Imports = append(vm.Imports, p)
		}
	}

	funcs := map[string]*ast.FuncDecl{}
	var order []string
	ast.Inspect(f, func(n ast.Node) bool {
		switch t := n.(type) {
		case *ast.TypeSpec:
			if _, ok := t.Type.(*ast.StructType); ok {
				vm.Structs = append(vm.Structs, t.Name.Name)
			}
		case *ast.FuncDecl:
			if _, seen := funcs[t.Name.Name]; !seen {
				order = append(order, t.Name.Name)
			}
			funcs[t.Name.Name] = t
		}
		return true
	})

	callsOf := map[string][]string{}
	for _, name := range order {
		var calls []string
		seen := map[string]bool{}
		ast.Inspect(funcs[name], func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			var cname string
			switch fn := call.Fun.(type) {
			case *ast.Ident:
				cname = fn.Name
			case *ast.SelectorExpr:
				if id, ok := fn.X.(*ast.Ident); ok {
					cname = id.Name + "." + fn.Sel.Name
				} else {
					cname = fn.Sel.Name
				}
			}
			if cname == "" || cname == name || seen[cname] {
				return true
			}
			seen[cname] = true
			calls = append(calls, cname)
			return true
		})
		sort.Strings(calls)
		callsOf[name] = calls
		vm.Functions = append(vm.Functions, FuncCallInfo{
			Name:  name,
			Calls: calls,
			Line:  fset.Position(funcs[name].Pos()).Line,
		})
	}

	// Execution sequence: depth-first from main over local calls.
	local := map[string]bool{}
	for _, fn := range vm.Functions {
		local[fn.Name] = true
	}
	start := ""
	if local["main"] {
		start = "main"
	} else if len(order) > 0 {
		start = order[0]
	}
	var steps []string
	visited := map[string]bool{}
	var dfs func(name string)
	dfs = func(name string) {
		if visited[name] || !local[name] {
			return
		}
		visited[name] = true
		steps = append(steps, name)
		for _, c := range callsOf[name] {
			base := c
			if i := strings.LastIndex(c, "."); i >= 0 {
				base = c[i+1:]
			}
			dfs(base)
		}
	}
	dfs(start)
	vm.CallSteps = steps

	reached := ""
	if len(steps) > 0 {
		reached = "; execution starts at " + steps[0] + " and reaches: " + strings.Join(steps, " -> ")
	}
	vm.Summary = fmt.Sprintf("%d import(s), %d struct(s), %d function(s)%s",
		len(vm.Imports), len(vm.Structs), len(vm.Functions), reached)
	return vm, nil
}

// checkCompiles parses and type-checks src as a standalone program.
// Type errors are collected (imports must resolve); it reports the
// first problem found.
func checkCompiles(src string) error {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "gen.go", src, 0)
	if err != nil {
		return err
	}
	var first error
	cfg := types.Config{
		Importer: importer.Default(),
		Error: func(e error) {
			if first == nil {
				first = e
			}
		},
	}
	_, _ = cfg.Check("beginner", fset, []*ast.File{f}, nil)
	return first
}

/*
Runnable example: translate a natural-language request into Go code,
print the code and its explanation.

package main

import (
	"fmt"

	"github.com/golangast/gollemer/pkg/beginner"
)

func main() {
	code, explanation, err := beginner.GenerateGoFromCommand(
		"Create an HTTP server listening on port 8080 with a health check endpoint")
	if err != nil {
		panic(err)
	}
	fmt.Println(code)
	fmt.Println("---- why this code ----")
	fmt.Println(explanation)

	concept, err := beginner.ExplainGoConcept("channels")
	if err != nil {
		panic(err)
	}
	fmt.Println("---- concept: " + concept.Concept + " ----")
	fmt.Println(concept.Analogy)
	for _, rule := range concept.KeyRules {
		fmt.Println("-", rule)
	}

	vm, err := beginner.BuildVisualMap(code)
	if err != nil {
		panic(err)
	}
	fmt.Println("---- visual map ----")
	fmt.Println(vm.Summary)
}
*/
