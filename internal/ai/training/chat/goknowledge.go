// Package chat — Go knowledge base.
//
// A tiny, dependency-free, keyword-indexed Go reference that the chat
// consults before the neural model when a message routes to the Go concept
// domain. The neural net is fluent but sometimes confidently wrong; these
// entries are curated, deterministic, and always correct. A lookup only
// fires on a strong keyword match — anything ambiguous falls through to
// the model.
package chat

import (
	"regexp"
	"sort"
	"strings"
)

// goKnowledgeEntry is one curated fact: the title names the topic,
// keywords drive retrieval, and body is the answer (plain prose,
// 1-3 sentences, no code blocks — the gocode brain owns code).
type goKnowledgeEntry struct {
	title    string
	keywords []string
	body     string
}

// goKnowledge is the curated reference. Keep bodies short, correct, and
// beginner-friendly; keep keywords distinctive (prefer terms like
// "goroutine" over "function").
var goKnowledge = []goKnowledgeEntry{
	{
		title:    "declaring variables",
		keywords: []string{"variable", "variables", "declare", "declaration", "var ", ":=", "zero value", "zero values"},
		body:     "Declare variables with var name type, or use := for short declaration with inference (x := 5). Every variable starts at its type's zero value: 0 for numbers, \"\" for strings, false for bools, nil for pointers, slices, and maps.",
	},
	{
		title:    "functions and multiple return values",
		keywords: []string{"function", "functions", "func", "return values", "multiple return", "parameters", "arguments"},
		body:     "Functions are declared with func name(params) returnType. Go functions can return multiple values, and by convention the last one is an error: result, err := doThing(). Callers check err before using result.",
	},
	{
		title:    "if for switch control flow",
		keywords: []string{"if statement", "for loop", "loops", "switch", "control flow", "range loop", "while"},
		body:     "Go has if, for, and switch — but no while or do-while. for covers everything: for i := 0; i < n; i++, for condition (acts as while), for {} (infinite), and for i, v := range slice. Switch cases don't fall through unless you write fallthrough.",
	},
	{
		title:    "slices",
		keywords: []string{"slice", "slices", "append", "make slice", "slice vs array"},
		body:     "Slices are dynamically-sized views over arrays: s := []int{1, 2, 3}. Use append(s, 4) to grow one — append may reallocate, so always keep its return value: s = append(s, 4). make([]int, len, cap) preallocates; len is the length, cap the capacity.",
	},
	{
		title:    "arrays",
		keywords: []string{"array", "arrays", "fixed size"},
		body:     "Arrays have a fixed size baked into their type: var a [4]int. They're values, so assigning or passing one copies all elements. In practice you almost always want slices instead.",
	},
	{
		title:    "maps",
		keywords: []string{"map", "maps", "dictionary", "hashmap", "hash map", "key value"},
		body:     "Maps are Go's hash tables: m := map[string]int{\"a\": 1}. Reading a missing key returns the zero value, so use the two-value form to test presence: v, ok := m[\"a\"]. Writing to a nil map panics — initialize with make or a literal first.",
	},
	{
		title:    "structs",
		keywords: []string{"struct", "structs", "fields", "record"},
		body:     "Structs group related fields: type Point struct { X, Y float64 }. Access fields with a dot: p.X. Struct literals can name fields (Point{X: 1}) or list them in order. Tags like `json:\"x\"` add metadata for libraries.",
	},
	{
		title:    "pointers",
		keywords: []string{"pointer", "pointers", "&", "dereference", "*", "address"},
		body:     "A pointer holds the address of a value: p := &x. Dereference with *p to read or write through it. Pointers let functions modify their arguments and avoid copying large structs. Go has no pointer arithmetic.",
	},
	{
		title:    "methods and receivers",
		keywords: []string{"method", "methods", "receiver", "pointer receiver", "value receiver"},
		body:     "Methods attach functions to types: func (p Point) Distance() float64. A value receiver gets a copy; a pointer receiver func (p *Point) can modify the original. Use pointer receivers when the method mutates state or the struct is large.",
	},
	{
		title:    "interfaces",
		keywords: []string{"interface", "interfaces", "implement", "polymorphism", "satisfies"},
		body:     "Interfaces declare method sets: type Writer interface { Write([]byte) (int, error) }. Types implement them implicitly — no implements keyword. This keeps coupling loose: depend on small interfaces, not concrete types. The empty interface any accepts any value.",
	},
	{
		title:    "type assertions and type switches",
		keywords: []string{"type assertion", "type switch", "type assertions", ".(type)"},
		body:     "Extract a concrete value from an interface with a type assertion: s := v.(string). The two-value form s, ok := v.(string) avoids a panic on mismatch. A type switch branches on dynamic type: switch t := v.(type) { case string: ... }.",
	},
	{
		title:    "generics",
		keywords: []string{"generics", "generic", "type parameter", "type parameters", "[T any]", "constraints"},
		body:     "Since Go 1.18, functions and types take type parameters: func Min[T constraints.Ordered](a, b T) T. any matches any type; comparable matches types supporting ==. Generics shine for containers and algorithms, but plain interfaces are often simpler.",
	},
	{
		title:    "packages and imports",
		keywords: []string{"package", "packages", "import", "imports", "exported", "unexported", "module path"},
		body:     "Every Go file starts with package name. Code is organized into packages imported by path: import \"fmt\". Names starting with a capital letter are exported (public); lowercase names are package-private. One package per directory.",
	},
	{
		title:    "init functions",
		keywords: []string{"init", "initialization", "package init"},
		body:     "A function named init runs automatically when its package loads, before main. It's for package-level setup. Keep init small and side-effect-light — heavy init logic makes programs hard to reason about and test.",
	},
	{
		title:    "constants and iota",
		keywords: []string{"constant", "constants", "const", "iota", "enum"},
		body:     "Constants are declared with const and can't change. iota generates sequences in const blocks: const ( Sunday = iota; Monday; Tuesday ) gives 0, 1, 2. It's Go's idiom for enumerations.",
	},
	{
		title:    "strings and runes",
		keywords: []string{"string", "strings", "rune", "runes", "unicode", "characters", "concatenation"},
		body:     "Strings are immutable sequences of bytes, usually UTF-8. Indexing gives bytes, not characters — use for _, r := range s to iterate runes (Unicode code points). Convert with []rune(s) when you need character counts: len(s) counts bytes.",
	},
	{
		title:    "goroutines",
		keywords: []string{"goroutine", "goroutines", "go statement", "concurrency", "concurrent", "parallel"},
		body:     "A goroutine is a lightweight thread managed by the Go runtime: go doWork(). Thousands of goroutines cost little. They run concurrently with the caller — use channels or sync primitives to coordinate, never shared memory without synchronization.",
	},
	{
		title:    "channels",
		keywords: []string{"channel", "channels", "chan", "buffered", "unbuffered", "send", "receive"},
		body:     "Channels move values between goroutines: ch := make(chan int). ch <- v sends, v := <-ch receives. Unbuffered channels block until both sides are ready (a rendezvous); buffered channels (make(chan int, 10)) block only when full or empty. Close with close(ch); range over a channel drains it until closed.",
	},
	{
		title:    "select",
		keywords: []string{"select", "multiplex", "multiple channels"},
		body:     "select waits on multiple channel operations, like a switch for channels: select { case v := <-ch1: ...; case ch2 <- x: ...; case <-time.After(d): ... }. If several cases are ready, one is picked at random. A default case makes it non-blocking.",
	},
	{
		title:    "sync.Mutex",
		keywords: []string{"mutex", "sync.mutex", "lock", "locking", "race", "data race", "mutual exclusion"},
		body:     "sync.Mutex guards shared state: mu.Lock() before touching it, mu.Unlock() after (defer the unlock). Keep critical sections tiny. For read-heavy data, sync.RWMutex lets many readers proceed while writers exclude everyone. Run go test -race to catch data races.",
	},
	{
		title:    "sync.WaitGroup",
		keywords: []string{"waitgroup", "wait group", "wait for goroutines", "wg.add", "wg.done"},
		body:     "sync.WaitGroup waits for goroutines to finish: wg.Add(1) before each go call, defer wg.Done() inside it, wg.Wait() to block until all are done. Add before launching, never copy a WaitGroup after use.",
	},
	{
		title:    "sync.Once",
		keywords: []string{"once", "sync.once", "singleton", "initialize once"},
		body:     "sync.Once runs a function exactly once, even across goroutines: once.Do(setup). It's the idiom for lazy singleton initialization without races.",
	},
	{
		title:    "context package",
		keywords: []string{"context", "context.Context", "cancellation", "cancel", "timeout", "deadline", "WithTimeout"},
		body:     "context.Context carries cancellation, deadlines, and request values: ctx, cancel := context.WithTimeout(parent, 5*time.Second). Pass ctx as the first parameter and check ctx.Done() in long work. Always call cancel to release resources. Never store contexts in structs.",
	},
	{
		title:    "worker pools",
		keywords: []string{"worker pool", "worker pools", "pool of workers", "fan-out", "fanout"},
		body:     "A worker pool bounds concurrency: launch N goroutines reading jobs from a channel and sending results to another, then close the jobs channel and wait. It keeps a flood of tasks from spawning a flood of goroutines.",
	},
	{
		title:    "fan-in fan-out pipelines",
		keywords: []string{"fan-in", "fanin", "pipeline", "pipelines", "stages"},
		body:     "Pipelines chain stages with channels: each stage reads from its input channel and writes to its output. Fan-out spreads work across many goroutines; fan-in merges multiple channels into one (often with a select loop or WaitGroup). Close outputs when inputs are drained.",
	},
	{
		title:    "goroutine leaks",
		keywords: []string{"goroutine leak", "leak", "leaking goroutines", "stuck goroutine"},
		body:     "A goroutine leak is a goroutine that never exits — usually blocked forever on a channel nobody will touch. Every goroutine should have a clear exit path: a done channel, context cancellation, or a closed input channel. Leaks slowly eat memory.",
	},
	{
		title:    "error values and handling",
		keywords: []string{"error", "errors", "error handling", "errors.New", "check error", "err !="},
		body:     "Errors are values of the error interface (a single method: Error() string). The idiom is explicit: result, err := f(); if err != nil { return ..., err }. Handle errors where you have context; add information as they propagate up.",
	},
	{
		title:    "wrapping errors",
		keywords: []string{"wrap", "wrapping", "fmt.Errorf", "%w", "errors.Is", "errors.As", "sentinel"},
		body:     "Wrap errors with context using %w: fmt.Errorf(\"open config: %w\", err). Callers unwrap with errors.Is(err, target) for sentinel comparison or errors.As for type extraction. Prefer wrapping over discarding the original error.",
	},
	{
		title:    "panic and recover",
		keywords: []string{"panic", "panics", "recover", "crash"},
		body:     "panic stops normal execution for unrecoverable problems (programmer bugs, corrupt state). recover inside a deferred function can catch it, but that's for boundaries like servers — ordinary code should return errors, not panic. Never use panic for normal control flow.",
	},
	{
		title:    "fmt package",
		keywords: []string{"fmt", "Println", "Printf", "format verbs", "printing", "Sprintf"},
		body:     "fmt prints: Println adds spaces and a newline, Printf formats with verbs (%v value, %d integer, %s string, %q quoted, %T type), Sprintf returns the string instead of printing. Check fmt docs for the full verb list.",
	},
	{
		title:    "strings and strconv",
		keywords: []string{"strings package", "strconv", "string manipulation", "split", "join", "contains", "Atoi", "Itoa", "parse int"},
		body:     "strings has Contains, HasPrefix, Split, Join, Replace, TrimSpace, and more. strconv converts: n, err := strconv.Atoi(\"42\"), s := strconv.Itoa(42), plus ParseBool/ParseFloat/FormatFloat. Build strings efficiently with strings.Builder.",
	},
	{
		title:    "os package",
		keywords: []string{"os package", "command line args", "os.Args", "environment variables", "Getenv", "exit code", "stdin stdout"},
		body:     "os bridges to the system: os.Args holds command-line arguments (index 0 is the program name), os.Getenv reads environment variables, os.Exit sets the exit code. os.Stdin/Stdout/Stderr are the standard streams.",
	},
	{
		title:    "reading and writing files",
		keywords: []string{"read file", "write file", "ReadFile", "WriteFile", "os.Open", "file i/o", "files"},
		body:     "os.ReadFile(name) reads a whole file into memory; os.WriteFile(name, data, 0644) writes it. For big files or streams, use os.Open with bufio.Scanner line by line, and always check errors and Close what you Open (defer f.Close()).",
	},
	{
		title:    "net/http servers",
		keywords: []string{"http server", "net/http", "web server", "http.HandleFunc", "ListenAndServe", "handler", "rest api"},
		body:     "net/http serves web traffic with no dependencies: http.HandleFunc(\"/\", handler) registers a route, http.ListenAndServe(\":8080\", nil) starts the server. Handlers get a ResponseWriter and *Request. For clients, http.Get or http.Post do the fetching.",
	},
	{
		title:    "encoding/json",
		keywords: []string{"json", "encoding/json", "marshal", "unmarshal", "json tags", "decode json", "encode json"},
		body:     "encoding/json converts between Go values and JSON: json.Marshal(v) encodes, json.Unmarshal(data, &v) decodes. Struct tags control field names: `json:\"name\"`. Unknown fields are ignored on decode; use json.Decoder for streams.",
	},
	{
		title:    "time package",
		keywords: []string{"time", "sleep", "duration", "ticker", "parse time", "time.After", "time format"},
		body:     "time handles clocks and durations: time.Sleep(2 * time.Second) pauses, time.Now() stamps, t.Add(d) shifts. time.After(d) returns a channel that fires once; time.NewTicker(d) repeats. Parse with the reference layout: time.Parse(\"2006-01-02\", s) — yes, that odd date is the template.",
	},
	{
		title:    "sort package",
		keywords: []string{"sort", "sorting", "sort.Slice", "order"},
		body:     "sort.Slice(s, func(i, j int) bool { return s[i] < s[j] }) sorts with a custom less function; sort.Ints and sort.Strings cover the basics. Since Go 1.21, the slices package offers a generic slices.Sort(s).",
	},
	{
		title:    "regexp",
		keywords: []string{"regexp", "regular expression", "regex", "pattern matching", "MatchString"},
		body:     "regexp compiles patterns: re := regexp.MustCompile(`\\d+`), re.MatchString(s) tests, re.FindAllString(s, -1) extracts. MustCompile panics on bad patterns — fine for constants, but use Compile and check the error for user-supplied patterns.",
	},
	{
		title:    "path and filepath",
		keywords: []string{"filepath", "path/filepath", "join paths", "file path", "directory"},
		body:     "path/filepath handles OS-correct paths: filepath.Join(\"a\", \"b\") inserts the right separator, filepath.Ext gives the extension, filepath.WalkDir traverses trees. Use path (not filepath) only for slash-separated paths like URLs.",
	},
	{
		title:    "log package",
		keywords: []string{"log package", "logging", "log.Fatal", "log.Println"},
		body:     "log prints timestamped lines: log.Println, log.Printf. log.Fatal prints then calls os.Exit(1) — handy in main, dangerous in libraries. For real applications, the standard slog package (Go 1.21+) adds levels and structured fields.",
	},
	{
		title:    "go run and go build",
		keywords: []string{"go run", "go build", "compile", "build binary", "run program"},
		body:     "go run . compiles and runs the package in the current directory without leaving a binary. go build . writes the executable (named after the directory). go build -o name sets the output filename.",
	},
	{
		title:    "go test",
		keywords: []string{"go test", "testing", "unit test", "table test", "table-driven", "benchmark", "test flags"},
		body:     "Tests live in _test.go files as func TestX(t *testing.T). go test ./... runs everything. Table-driven tests loop over input/expected pairs — the standard Go idiom. go test -run Name -v runs one test verbosely; -race detects data races; BenchmarkY functions measure speed.",
	},
	{
		title:    "go vet",
		keywords: []string{"go vet", "vet", "static analysis", "lint"},
		body:     "go vet statically scans for suspicious code: wrong Printf verbs, unreachable code, bad struct tags, copying locks. Run it before committing — CI should too. It's not a formatter (that's gofmt) and not a full linter, but it catches real bugs.",
	},
	{
		title:    "gofmt",
		keywords: []string{"gofmt", "formatting", "format code", "go fmt"},
		body:     "gofmt enforces Go's standard formatting: tabs, aligned braces, sorted imports. gofmt -l lists files needing formatting; gofmt -w rewrites them in place. Never hand-format — run gofmt and move on.",
	},
	{
		title:    "go modules",
		keywords: []string{"go.mod", "go mod", "modules", "dependencies", "go get", "go.sum", "module"},
		body:     "Go modules track dependencies: go mod init example.com/m creates go.mod; go get pkg@version adds or upgrades; go mod tidy adds missing and drops unused requirements. go.sum pins exact hashes so builds are reproducible.",
	},
	{
		title:    "godoc",
		keywords: []string{"godoc", "documentation", "doc comments", "go doc"},
		body:     "Documentation comes from comments: a comment starting with the name (// Add sums...) becomes the doc. go doc pkg.Symbol prints it in the terminal; pkg.go.dev hosts it online. Write doc comments on every exported name.",
	},
	{
		title:    "nil map write panic",
		keywords: []string{"nil map", "assignment to entry in nil map", "map panic"},
		body:     "Writing to a nil map panics with 'assignment to entry in nil map'. Reading is fine (zero value). Always initialize before writing: m := make(map[string]int) or a literal. This bites when a map field is left at its zero value.",
	},
	{
		title:    "slice append aliasing",
		keywords: []string{"append aliasing", "slice aliasing", "shared backing array", "append gotcha"},
		body:     "Appending to a slice can silently modify another slice sharing the same backing array when capacity allows. If two slices must stay independent, copy first or append to a fresh slice. When in doubt, full slice expressions (s[0:2:2]) cap capacity.",
	},
	{
		title:    "loop variable capture",
		keywords: []string{"loop variable", "closure", "goroutine loop", "capture", "range variable"},
		body:     "Before Go 1.22, loop variables were reused across iterations, so closures and goroutines launched in a loop all saw the last value — the classic fix was shadowing (x := x) inside the loop. Go 1.22+ gives each iteration fresh variables, but you'll still meet the old pattern in legacy code.",
	},
	{
		title:    "typed nil in interfaces",
		keywords: []string{"typed nil", "nil interface", "interface nil", "nil check interface"},
		body:     "An interface holding a typed nil pointer is NOT nil — it has a type but no value, so err != nil passes unexpectedly. Return explicit nil instead of a nil concrete value, and design so nil means nil. This is one of Go's most famous gotchas.",
	},
	{
		title:    "copying locks",
		keywords: []string{"copying locks", "copylocks", "mutex copy", "noCopy"},
		body:     "Copying a struct containing a sync.Mutex (or WaitGroup) copies the lock state — go vet flags it as 'copylocks'. Pass such structs by pointer. The noCopy convention (embedding sync.Mutex or a noCopy struct) makes vet catch it.",
	},
	{
		title:    "variable shadowing",
		keywords: []string{"shadowing", "shadow", ":= shadowing", "variable shadow"},
		body:     ":= in an inner scope creates a NEW variable that shadows the outer one — assignments then hit the inner copy and the outer never changes. go vet's shadow check (via golang.org/x/tools) catches it; otherwise watch for := where you meant =.",
	},
	{
		title:    "defer",
		keywords: []string{"defer", "deferred", "cleanup", "LIFO"},
		body:     "defer schedules a call to run when the function returns: defer f.Close(). Deferred calls run LIFO (last in, first out) and see the final values of named return variables — which is how recover-based patterns work. Use it for cleanup, not for hot-loop performance.",
	},
	{
		title:    "blank identifier",
		keywords: []string{"blank identifier", "_", "underscore", "unused import", "ignore value"},
		body:     "The blank identifier _ discards values: _, err := f() ignores the first result. It's also how you import for side effects (import _ \"pkg\") and satisfy the compiler about unused variables during development.",
	},
	{
		title:    "embedding",
		keywords: []string{"embedding", "embed", "composition", "inheritance", "promoted methods"},
		body:     "Go composes instead of inheriting: embedding a struct or interface promotes its fields and methods to the outer type. type Server struct { http.Server } gets all of http.Server's methods. It's composition with syntactic sugar, not a class hierarchy.",
	},
}

var goKnowledgeWord = regexp.MustCompile(`[a-z0-9+.#]+`)

// knowledgeStopwords are too common to drive retrieval; they are stripped
// before matching so "is" can't substring-match "polymorphism".
var knowledgeStopwords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true, "was": true,
	"were": true, "be": true, "been": true, "what": true, "how": true,
	"why": true, "when": true, "where": true, "which": true, "who": true,
	"do": true, "does": true, "did": true, "you": true, "me": true,
	"it": true, "its": true, "to": true, "of": true, "in": true, "on": true,
	"for": true, "with": true, "about": true, "tell": true, "explain": true,
	"describe": true, "please": true, "my": true, "your": true, "this": true,
	"that": true, "these": true, "those": true, "and": true, "or": true,
	"as": true, "at": true, "by": true, "from": true, "can": true,
	"could": true, "would": true, "should": true, "have": true, "has": true,
	"had": true, "want": true, "know": true, "like": true, "get": true,
	"got": true, "just": true, "very": true, "really": true, "much": true,
	"more": true, "most": true, "some": true, "any": true, "all": true,
	"there": true, "their": true, "they": true, "them": true, "then": true,
	"than": true, "so": true, "such": true, "into": true, "over": true,
	"under": true, "between": true, "through": true, "during": true,
}

// knowledgeTokens normalizes a query into matchable words.
func knowledgeTokens(s string) []string {
	s = strings.ToLower(s)
	words := goKnowledgeWord.FindAllString(s, -1)
	seen := map[string]bool{}
	out := make([]string, 0, len(words))
	for _, w := range words {
		if len(w) < 2 || seen[w] || knowledgeStopwords[w] {
			continue
		}
		seen[w] = true
		out = append(out, w)
	}
	return out
}

// stem strips a trailing plural s for lenient matching.
func stem(w string) string {
	if len(w) > 3 && strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss") {
		return w[:len(w)-1]
	}
	return w
}

// LookupGoKnowledge finds the best curated entry for a Go question.
// It returns the answer body and true on a strong keyword match;
// anything ambiguous returns false so the neural model handles it.
//
// A hit needs one of: a phrase-keyword match, a distinctive-term match
// (long or dotted keywords like "goroutine" or "sync.mutex"), or at
// least two distinct keyword hits. Single common words ("maps" in
// "i like maps") never fire on their own.
func LookupGoKnowledge(query string) (string, bool) {
	toks := knowledgeTokens(query)
	if len(toks) == 0 {
		return "", false
	}
	tokSet := map[string]bool{}
	for _, t := range toks {
		tokSet[t] = true
		tokSet[stem(t)] = true
	}
	lowerQuery := strings.ToLower(query)
	type scored struct {
		idx   int
		hit   bool
		score int
	}
	var ranked []scored
	for i, e := range goKnowledge {
		// matched collects stem-deduped keywords so "map"+"maps" count once.
		matched := map[string]bool{}
		phrases := 0
		distinctive := 0
		for _, kw := range e.keywords {
			kw = strings.ToLower(kw)
			if strings.Contains(kw, " ") {
				if strings.Contains(lowerQuery, kw) {
					phrases++
					matched["phrase:"+kw] = true
				}
				continue
			}
			st := stem(kw)
			if matched[st] {
				continue
			}
			isMatch := tokSet[kw] || tokSet[st]
			if !isMatch {
				// Substring only in the keyword-contains-token direction,
				// using original (unstemmed) tokens of real length: this
				// catches "mutex" in "sync.mutex" without letting "is"
				// match "polymorphism" or "map" match "hashmap".
				for _, t := range toks {
					if len(t) >= 4 && strings.Contains(kw, t) {
						isMatch = true
						break
					}
				}
			}
			if isMatch {
				matched[st] = true
				if len(kw) >= 6 || strings.Contains(kw, ".") {
					distinctive++
				}
			}
		}
		hit := phrases > 0 || distinctive > 0 || len(matched) >= 2
		if !hit {
			continue
		}
		score := len(matched) + distinctive + 2*phrases
		for _, tw := range knowledgeTokens(e.title) {
			if tokSet[tw] {
				score++
			}
		}
		ranked = append(ranked, scored{i, true, score})
	}
	if len(ranked) == 0 {
		return "", false
	}
	sort.Slice(ranked, func(a, b int) bool { return ranked[a].score > ranked[b].score })
	return goKnowledge[ranked[0].idx].body, true
}
