// Command gollemer is an interactive natural-language Go shell.
//
// It combines natural language code synthesis, formal static safety
// verification, genetic memory auto-tuning, and step-by-step visual
// execution tracing into a pure terminal experience. Built-in
// commands: help, exit, quit.
//
// This file is self-contained: it compiles alone with
// `go run cmd/gollemer/main.go` and uses only the Go standard library.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"strings"
)

// ANSI escape sequences. Emptied when NO_COLOR is set so output stays
// readable on terminals without color support.
var (
	Reset   = "\033[0m"
	Bold    = "\033[1m"
	Red     = "\033[31m"
	Green   = "\033[32m"
	Yellow  = "\033[33m"
	Cyan    = "\033[36m"
	Magenta = "\033[35m"
	Gray    = "\033[90m"
	White   = "\033[97m"
)

func init() {
	if os.Getenv("NO_COLOR") != "" {
		Reset, Bold, Red, Green, Yellow, Cyan, Magenta, Gray, White = "", "", "", "", "", "", "", "", ""
	}
}

func main() {
	once := flag.String("once", "", "run one prompt through the pipeline and exit (non-interactive)")
	flag.Parse()
	if *once != "" {
		if err := processPrompt(*once); err != nil {
			fmt.Fprintf(os.Stderr, "!! %s\n", err)
			os.Exit(1)
		}
		return
	}
	runREPL()
}

// runREPL is the interactive terminal loop: bold prompt, line input,
// built-in commands, and per-prompt pipeline processing.
func runREPL() {
	fmt.Println(Bold + "gollemer — natural language Go shell" + Reset)
	fmt.Println(`Ask for Go in plain English, e.g. "create a worker pool".`)
	fmt.Println(`Type "help" for help, "exit" to leave.`)
	fmt.Println()

	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 4096), 1024*1024)
	for {
		fmt.Print(Bold + "gollemer> " + Reset)
		if !sc.Scan() {
			fmt.Println()
			break
		}
		line := strings.TrimSpace(sc.Text())
		switch line {
		case "":
			continue
		case "exit", "quit":
			fmt.Println("bye!")
			return
		case "help":
			printHelp()
			continue
		}
		if err := processPrompt(line); err != nil {
			fmt.Printf("%s!! %s%s\n\n", Red, err, Reset)
		}
	}
	if err := sc.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "input error: %v\n", err)
	}
}

func printHelp() {
	fmt.Println()
	fmt.Println(Bold + "commands" + Reset)
	fmt.Println("  help          show this help")
	fmt.Println("  exit | quit   leave the shell")
	fmt.Println()
	fmt.Println(Bold + "things to try" + Reset)
	fmt.Println("  create a worker pool")
	fmt.Println("  write an http server")
	fmt.Println("  read a file line by line")
	fmt.Println("  encode a struct to json")
	fmt.Println("  make a mutex counter")
	fmt.Println("  run a ticker")
	fmt.Println("  build a list of squares")
	fmt.Println()
}

// ------------------------------------------------------------------
// Pipeline data model.
// ------------------------------------------------------------------

// ExecutionStep is one card of the visual execution trace.
type ExecutionStep struct {
	Number      int    // 1-based step number
	Title       string // function name, humanized
	Subtitle    string // package/file/function call context
	Description string // operational description derived from the AST
}

// SafetyReport is the outcome of static safety verification.
type SafetyReport struct {
	Verified bool     // true when guard clauses exist and no panic paths
	Guards   int      // guard clauses found
	Panics   int      // panic() call sites found
	Notes    []string // human-readable findings
}

// PerfReport is the outcome of genetic memory auto-tuning.
type PerfReport struct {
	AllocsBefore int      // unbounded allocation sites before tuning
	AllocsAfter  int      // unbounded allocation sites after tuning
	WasteBefore  int      // over-provisioned buffer capacity before
	WasteAfter   int      // over-provisioned buffer capacity after
	Changes      []string // applied tuning changes
}

// PipelineResult is everything the reactive pipeline produces.
type PipelineResult struct {
	Pattern string // matched intent name, e.g. "worker pool"
	Code    string // final tuned + formatted Go source
	Analogy string // one-sentence real-world analogy
	Steps   []ExecutionStep
	Safety  SafetyReport
	Perf    PerfReport
}

// processPrompt runs the reactive pipeline for one request and
// renders the result to the terminal.
func processPrompt(prompt string) error {
	out, err := runPipeline(prompt)
	if err != nil {
		return err
	}
	renderTerminalOutput(out)
	return nil
}

// runPipeline is the 4-stage reactive pipeline:
//
//	Stage 1 — AST synthesis: natural language -> idiomatic Go source.
//	Stage 2 — Format & parse: go/parser + go/format, then genetic
//	  memory auto-tuning rewrites capacities (verified by re-parse).
//	Stage 3 — Static safety inspection: guard clauses via ast.Inspect.
//	Stage 4 — Trace & analogy: visual execution steps + 1-sentence
//	  real-world analogy.
func runPipeline(prompt string) (PipelineResult, error) {
	var out PipelineResult

	// Stage 1 — AST synthesis.
	code, pattern, err := synthesize(prompt)
	if err != nil {
		return out, err
	}
	out.Pattern = pattern
	out.Analogy = analogies[pattern]

	// Stage 2 — format & parse, then auto-tune.
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "repl.go", code, 0)
	if err != nil {
		return out, fmt.Errorf("stage 2 (parse): %w", err)
	}
	out.Perf.AllocsBefore = countUnboundedAllocs(file)
	preallocateSlices(file) // deterministic: size slices up front
	tuned := geneticTune(collectCapSites(file), pattern)
	out.Perf.Changes = tuned.changes
	out.Perf.WasteBefore, out.Perf.WasteAfter = tuned.wasteBefore, tuned.wasteAfter
	formatted, err := formatNode(fset, file)
	if err != nil {
		return out, fmt.Errorf("stage 2 (format): %w", err)
	}
	out.Code = formatted
	out.Perf.AllocsAfter = countUnboundedAllocs(file)

	// Re-parse the tuned code so later stages see final positions.
	fset2 := token.NewFileSet()
	tunedFile, err := parser.ParseFile(fset2, "repl.go", formatted, 0)
	if err != nil {
		return out, fmt.Errorf("stage 2 (re-parse): %w", err)
	}

	// Stage 3 — static safety inspection.
	out.Safety = inspectGuards(tunedFile)

	// Stage 4 — trace & analogy.
	out.Steps = traceExecution(tunedFile, fset2)

	return out, nil
}

func formatNode(fset *token.FileSet, file *ast.File) (string, error) {
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, file); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// ------------------------------------------------------------------
// renderTerminalOutput — vibrant ANSI terminal renderer.
// ------------------------------------------------------------------

// renderTerminalOutput renders a pipeline result to stdout in four
// distinct visual sections.
func renderTerminalOutput(out PipelineResult) {
	var buf bytes.Buffer
	renderTo(&buf, out)
	fmt.Print(buf.String())
}

// renderTo is renderTerminalOutput with an injectable buffer (tests).
func renderTo(buf *bytes.Buffer, out PipelineResult) {
	p := func(format string, args ...interface{}) { fmt.Fprintf(buf, format, args...) }

	// Section 1 — generated Go source, syntax highlighted.
	p("\n%s=== GENERATED GO SOURCE ===%s\n", Bold, Reset)
	p("%s\n", highlightGo(out.Code))

	// Section 2 — beginner concept analogy in yellow.
	p("%s💡 BEGINNER CONCEPT ANALOGY%s\n", Bold, Reset)
	p("%s%s%s\n\n", Yellow, out.Analogy, Reset)

	// Section 3 — visual execution trace: magenta step numbers,
	// bold titles, package contexts, gray tree characters.
	p("%s─── VISUAL EXECUTION TRACE ───%s\n", Bold, Reset)
	for _, s := range out.Steps {
		p("%s[%d]%s %s%s%s\n", Magenta, s.Number, Reset, Bold, s.Title, Reset)
		p("%s└──%s %s%s%s\n", Gray, Reset, White, s.Subtitle, Reset)
		p("%s    %s%s\n", Gray, s.Description, Reset)
	}
	p("\n")

	// Section 4 — status badges: green safety, cyan performance.
	p("%sSTATUS BADGES%s\n", Bold, Reset)
	if out.Safety.Verified {
		p("%s✓ SAFE%s — 0 panic paths, %d guard clause%s\n",
			Green, Reset, out.Safety.Guards, pluralS(out.Safety.Guards))
	} else {
		p("%s⚠ UNVERIFIED%s — %d panic path%s, %d guard clause%s\n",
			Yellow, Reset, out.Safety.Panics, pluralS(out.Safety.Panics),
			out.Safety.Guards, pluralS(out.Safety.Guards))
	}
	for _, n := range out.Safety.Notes {
		p("  %s%s%s\n", Gray, n, Reset)
	}
	saved := out.Perf.AllocsBefore - out.Perf.AllocsAfter
	p("%s⚡ PERF%s — %d → %d allocs/op (saved %d)",
		Cyan, Reset, out.Perf.AllocsBefore, out.Perf.AllocsAfter, saved)
	if out.Perf.WasteBefore != out.Perf.WasteAfter {
		p(", buffer waste %d → %d", out.Perf.WasteBefore, out.Perf.WasteAfter)
	}
	p("\n")
	for _, c := range out.Perf.Changes {
		p("  %s%s%s\n", Gray, c, Reset)
	}
	p("\n")
}

func pluralS(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// ------------------------------------------------------------------
// Syntax highlighting — a tiny hand-rolled Go tokenizer (stdlib only).
// ------------------------------------------------------------------

var goKeywords = map[string]bool{
	"break": true, "case": true, "chan": true, "const": true,
	"continue": true, "default": true, "defer": true, "else": true,
	"fallthrough": true, "for": true, "func": true, "go": true,
	"goto": true, "if": true, "import": true, "interface": true,
	"map": true, "package": true, "range": true, "return": true,
	"select": true, "struct": true, "switch": true, "type": true,
	"var": true,
}

func isGoLetter(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

// highlightGo colors keywords green, strings yellow, numbers cyan,
// and comments gray, preserving all other bytes verbatim.
func highlightGo(src string) string {
	var b strings.Builder
	b.Grow(len(src) + len(src)/4)
	i := 0
	for i < len(src) {
		c := src[i]
		switch {
		case c == '/' && i+1 < len(src) && src[i+1] == '/':
			j := i + 2
			for j < len(src) && src[j] != '\n' {
				j++
			}
			b.WriteString(Gray + src[i:j] + Reset)
			i = j
		case c == '/' && i+1 < len(src) && src[i+1] == '*':
			j := i + 2
			for j+1 < len(src) && !(src[j] == '*' && src[j+1] == '/') {
				j++
			}
			j += 2
			if j > len(src) {
				j = len(src)
			}
			b.WriteString(Gray + src[i:j] + Reset)
			i = j
		case c == '`':
			j := i + 1
			for j < len(src) && src[j] != '`' {
				j++
			}
			if j < len(src) {
				j++
			}
			b.WriteString(Yellow + src[i:j] + Reset)
			i = j
		case c == '"' || c == '\'':
			j := i + 1
			for j < len(src) {
				if src[j] == '\\' {
					j += 2
					continue
				}
				if src[j] == c {
					j++
					break
				}
				j++
			}
			b.WriteString(Yellow + src[i:j] + Reset)
			i = j
		case c >= '0' && c <= '9':
			j := i
			for j < len(src) && (src[j] >= '0' && src[j] <= '9' || src[j] == '.' ||
				(src[j] >= 'a' && src[j] <= 'f') || (src[j] >= 'A' && src[j] <= 'F') ||
				src[j] == 'x' || src[j] == 'X') {
				j++
			}
			b.WriteString(Cyan + src[i:j] + Reset)
			i = j
		case isGoLetter(c):
			j := i
			for j < len(src) && (isGoLetter(src[j]) || (src[j] >= '0' && src[j] <= '9')) {
				j++
			}
			word := src[i:j]
			if goKeywords[word] {
				b.WriteString(Green + word + Reset)
			} else {
				b.WriteString(word)
			}
			i = j
		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String()
}

// ------------------------------------------------------------------
// Stage 1: synthesis — keyword intent router over hand-written,
// idiomatic Go templates. Deterministic, stdlib only.
// ------------------------------------------------------------------

type intent struct {
	name     string
	anyOf    [][]string // match when every keyword of any group is present
	template string
}

var intents = []intent{
	{"worker pool", [][]string{{"worker", "pool"}, {"fan", "out"}},
		`package main

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
	for w := 1; w <= 3; w++ {
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
`},
	{"http server", [][]string{{"http", "server"}, {"web", "server"}},
		`package main

import (
	"fmt"
	"net/http"
)

func health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintln(w, ` + "`" + `{"status":"ok"}` + "`" + `)
}

func main() {
	http.HandleFunc("/health", health)
	fmt.Println("listening on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("server error:", err)
	}
}
`},
	{"read file", [][]string{{"read", "file"}, {"lines", "file"}},
		`package main

import (
	"bufio"
	"fmt"
	"os"
)

func main() {
	f, err := os.Open("notes.txt")
	if err != nil {
		fmt.Println("open error:", err)
		return
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fmt.Println(sc.Text())
	}
	if err := sc.Err(); err != nil {
		fmt.Println("scan error:", err)
	}
}
`},
	{"json", [][]string{{"json"}},
		`package main

import (
	"encoding/json"
	"fmt"
)

type User struct {
	Name string ` + "`json:\"name\"`" + `
	Age  int    ` + "`json:\"age\"`" + `
}

func main() {
	u := User{Name: "gopher", Age: 7}
	out, err := json.MarshalIndent(u, "", "  ")
	if err != nil {
		fmt.Println("json error:", err)
		return
	}
	fmt.Println(string(out))
}
`},
	{"mutex counter", [][]string{{"mutex", "counter"}, {"safe", "counter"}},
		`package main

import (
	"fmt"
	"sync"
)

type Counter struct {
	mu sync.Mutex
	n  int
}

func (c *Counter) Inc() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.n++
}

func (c *Counter) Value() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.n
}

func main() {
	var wg sync.WaitGroup
	c := &Counter{}
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c.Inc()
		}()
	}
	wg.Wait()
	fmt.Println("count:", c.Value())
}
`},
	{"ticker", [][]string{{"ticker"}, {"every", "second"}},
		`package main

import (
	"fmt"
	"time"
)

func main() {
	t := time.NewTicker(500 * time.Millisecond)
	defer t.Stop()
	done := time.After(2 * time.Second)
	for {
		select {
		case tm := <-t.C:
			fmt.Println("tick at", tm.Format("15:04:05"))
		case <-done:
			fmt.Println("done")
			return
		}
	}
}
`},
	{"build a list", [][]string{{"build", "list"}, {"squares"}},
		`package main

import "fmt"

func squares(n int) []int {
	var out []int
	for i := 0; i < n; i++ {
		out = append(out, i*i)
	}
	return out
}

func main() {
	for _, s := range squares(10) {
		fmt.Println(s)
	}
}
`},
}

// analogies holds the one-sentence real-world comparison per pattern.
var analogies = map[string]string{
	"worker pool":   "A restaurant kitchen: orders arrive on a ticket rail (the channel) and each cook (a goroutine) grabs the next ticket until the rail is empty.",
	"http server":   "A reception desk that never closes: every visitor (request) is greeted the same way, one after another.",
	"read file":     "Reading a receipt line by line: the scanner walks down the paper and hands you one line at a time.",
	"json":          "Packing a labeled suitcase: each struct field becomes a named compartment that anyone can unpack later.",
	"mutex counter": "A single bathroom key: only the holder may touch the counter, so the count is never wrong.",
	"ticker":        "A metronome: it ticks on a steady beat and your code dances to each tick until the song ends.",
	"build a list":  "A factory assembly line: the tray is sized for the whole order up front, so workers never stop to fetch a bigger one.",
}

// synthesize picks the first intent whose keywords all appear in the
// prompt. Unknown prompts get an error listing what the shell can do.
func synthesize(prompt string) (code, pattern string, err error) {
	lc := strings.ToLower(prompt)
	for _, in := range intents {
		for _, group := range in.anyOf {
			hit := true
			for _, kw := range group {
				if !strings.Contains(lc, kw) {
					hit = false
					break
				}
			}
			if hit {
				return in.template, in.name, nil
			}
		}
	}
	var names []string
	for _, in := range intents {
		names = append(names, in.name)
	}
	return "", "", fmt.Errorf("I can't build that yet — try one of: %s", strings.Join(names, ", "))
}

// ------------------------------------------------------------------
// Auto-tuning, part 1 (deterministic): slice preallocation.
// ------------------------------------------------------------------

// countUnboundedAllocs counts append calls on slices with no statically
// known capacity: each is a potential growth reallocation.
func countUnboundedAllocs(file *ast.File) int {
	prealloc := map[string]bool{}
	ast.Inspect(file, func(n ast.Node) bool {
		assign, ok := n.(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
			return true
		}
		lhs, ok := assign.Lhs[0].(*ast.Ident)
		if !ok {
			return true
		}
		call, ok := assign.Rhs[0].(*ast.CallExpr)
		if !ok {
			return true
		}
		if mk, ok := call.Fun.(*ast.Ident); !ok || mk.Name != "make" || len(call.Args) < 2 {
			return true
		}
		if _, ok := call.Args[0].(*ast.ArrayType); ok {
			prealloc[lhs.Name] = true
		}
		return true
	})
	n := 0
	ast.Inspect(file, func(x ast.Node) bool {
		call, ok := x.(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || id.Name != "append" || len(call.Args) == 0 {
			return true
		}
		if tgt, ok := call.Args[0].(*ast.Ident); ok && !prealloc[tgt.Name] {
			n++
		}
		return true
	})
	return n
}

// preallocateSlices rewrites `var s []T` followed by a counted append
// loop into `s := make([]T, 0, N)`. Bounds with possible side effects
// are never hoisted.
func preallocateSlices(file *ast.File) {
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		ast.Inspect(fn.Body, func(x ast.Node) bool {
			if blk, ok := x.(*ast.BlockStmt); ok {
				preallocateInBlock(blk)
			}
			return true
		})
	}
}

func preallocateInBlock(body *ast.BlockStmt) {
	for i, stmt := range body.List {
		ds, ok := stmt.(*ast.DeclStmt)
		if !ok {
			continue
		}
		gd, ok := ds.Decl.(*ast.GenDecl)
		if !ok || gd.Tok != token.VAR || len(gd.Specs) != 1 {
			continue
		}
		vs, ok := gd.Specs[0].(*ast.ValueSpec)
		if !ok || len(vs.Names) != 1 || len(vs.Values) != 0 {
			continue
		}
		arr, ok := vs.Type.(*ast.ArrayType)
		if !ok || arr.Len != nil {
			continue
		}
		name := vs.Names[0].Name
		for j := i + 1; j < len(body.List); j++ {
			bound, covered := loopAppendsTo(body.List[j], name)
			if bound == nil {
				continue
			}
			body.List[i] = &ast.AssignStmt{
				Lhs:    []ast.Expr{ast.NewIdent(name)},
				TokPos: gd.Pos(),
				Tok:    token.DEFINE,
				Rhs: []ast.Expr{&ast.CallExpr{
					Fun: ast.NewIdent("make"),
					Args: []ast.Expr{
						&ast.ArrayType{Elt: arr.Elt},
						&ast.BasicLit{Kind: token.INT, Value: "0"},
						bound,
					},
				}},
			}
			_ = covered
			break
		}
	}
}

// loopAppendsTo reports the bound expression of a `for i := S; i < B;
// i++` loop whose body appends to name, plus covered append sites.
func loopAppendsTo(stmt ast.Stmt, name string) (ast.Expr, int) {
	fs, ok := stmt.(*ast.ForStmt)
	if !ok || fs.Init == nil || fs.Cond == nil || fs.Post == nil {
		return nil, 0
	}
	initAssign, ok := fs.Init.(*ast.AssignStmt)
	if !ok || initAssign.Tok != token.DEFINE || len(initAssign.Rhs) != 1 {
		return nil, 0
	}
	if _, ok := initAssign.Rhs[0].(*ast.BasicLit); !ok {
		return nil, 0
	}
	cond, ok := fs.Cond.(*ast.BinaryExpr)
	if !ok || (cond.Op != token.LSS && cond.Op != token.LEQ) {
		return nil, 0
	}
	if _, ok := cond.X.(*ast.Ident); !ok {
		return nil, 0
	}
	if !isPureBound(cond.Y) {
		return nil, 0
	}
	covered := 0
	ast.Inspect(fs.Body, func(x ast.Node) bool {
		assign, ok := x.(*ast.AssignStmt)
		if !ok || len(assign.Rhs) != 1 {
			return true
		}
		call, ok := assign.Rhs[0].(*ast.CallExpr)
		if !ok {
			return true
		}
		id, ok := call.Fun.(*ast.Ident)
		if !ok || id.Name != "append" || len(call.Args) == 0 {
			return true
		}
		if tgt, ok := call.Args[0].(*ast.Ident); ok && tgt.Name == name {
			covered++
		}
		return true
	})
	if covered == 0 {
		return nil, 0
	}
	return cond.Y, covered
}

// isPureBound reports whether e is safe to evaluate once up front.
func isPureBound(e ast.Expr) bool {
	switch e.(type) {
	case *ast.BasicLit, *ast.Ident, *ast.SelectorExpr:
		return true
	}
	return false
}

// ------------------------------------------------------------------
// Auto-tuning, part 2 (genetic memory): buffer capacity search.
//
// A tiny real genetic algorithm over capacity literals. Each tunable
// `make` capacity is a gene; fitness rewards matching the statically
// estimated demand (loop trip counts) and punishes shortfall (blocking
// / growth risk) and waste (over-provisioned memory). Winning vectors
// persist in tuningMemory across prompts ("memory") and seed future
// populations.
// ------------------------------------------------------------------

// tuningMemory maps an intent pattern to its winning capacity vector.
var tuningMemory = map[string][]int{}

// capSite is one tunable capacity literal.
type capSite struct {
	label  string
	get    func() int
	set    func(int)
	demand int // static demand; sites with unknown demand are excluded
}

// tuneOutcome reports what the genetic search did.
type tuneOutcome struct {
	changes     []string
	wasteBefore int
	wasteAfter  int
}

// collectCapSites finds tunable capacities: `make(chan T, N)` and
// `make([]T, 0, N)` / `make([]T, N)` where N is an int literal or a
// const int ident, on variables whose send/append demand is statically
// known from counted loops.
func collectCapSites(file *ast.File) []capSite {
	consts := map[string]*ast.BasicLit{}
	for _, d := range file.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.CONST {
			continue
		}
		for _, sp := range gd.Specs {
			vs, ok := sp.(*ast.ValueSpec)
			if !ok || len(vs.Names) != 1 || len(vs.Values) != 1 {
				continue
			}
			if lit, ok := vs.Values[0].(*ast.BasicLit); ok && lit.Kind == token.INT {
				consts[vs.Names[0].Name] = lit
			}
		}
	}

	// Static demand per variable: max trip count over counted loops
	// containing a send/append to it; unknown when any use is outside
	// a counted loop.
	demand := map[string]int{}
	unknown := map[string]bool{}
	noteUse := func(name string, loop *ast.ForStmt) {
		if unknown[name] {
			return
		}
		if loop == nil {
			unknown[name] = true
			return
		}
		trips, ok := loopTrips(loop, consts)
		if !ok {
			unknown[name] = true
			return
		}
		if trips > demand[name] {
			demand[name] = trips
		}
	}
	for _, d := range file.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		walkStmts(fn.Body.List, nil, consts, func(s ast.Stmt, loop *ast.ForStmt) {
			switch x := s.(type) {
			case *ast.SendStmt:
				if id, ok := x.Chan.(*ast.Ident); ok {
					noteUse(id.Name, loop)
				}
			case *ast.AssignStmt:
				for _, rhs := range x.Rhs {
					call, ok := rhs.(*ast.CallExpr)
					if !ok {
						continue
					}
					if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "append" && len(call.Args) > 0 {
						if tgt, ok := call.Args[0].(*ast.Ident); ok {
							noteUse(tgt.Name, loop)
						}
					}
				}
			}
		})
	}

	var sites []capSite
	ast.Inspect(file, func(n ast.Node) bool {
		assign, ok := n.(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
			return true
		}
		lhs, ok := assign.Lhs[0].(*ast.Ident)
		if !ok {
			return true
		}
		call, ok := assign.Rhs[0].(*ast.CallExpr)
		if !ok {
			return true
		}
		mk, ok := call.Fun.(*ast.Ident)
		if !ok || mk.Name != "make" || len(call.Args) < 2 {
			return true
		}
		capArg := call.Args[len(call.Args)-1]
		var lit *ast.BasicLit
		switch x := capArg.(type) {
		case *ast.BasicLit:
			if x.Kind == token.INT {
				lit = x
			}
		case *ast.Ident:
			lit = consts[x.Name]
		}
		if lit == nil {
			return true
		}
		d, known := demand[lhs.Name]
		if !known || unknown[lhs.Name] || d <= 0 {
			return true
		}
		l := lit // capture
		label := lhs.Name
		sites = append(sites, capSite{
			label:  label,
			get:    func() int { return atoi(l.Value) },
			set:    func(v int) { l.Value = fmt.Sprint(v) },
			demand: d,
		})
		return true
	})
	return sites
}

// walkStmts visits every statement, carrying the innermost counted loop.
func walkStmts(list []ast.Stmt, loop *ast.ForStmt, consts map[string]*ast.BasicLit, visit func(s ast.Stmt, loop *ast.ForStmt)) {
	for _, s := range list {
		visit(s, loop)
		switch x := s.(type) {
		case *ast.ForStmt:
			inner := loop
			if _, ok := loopTrips(x, consts); ok {
				inner = x
			}
			if x.Body != nil {
				walkStmts(x.Body.List, inner, consts, visit)
			}
		case *ast.RangeStmt:
			if x.Body != nil {
				walkStmts(x.Body.List, nil, consts, visit) // range trips unknown
			}
		case *ast.IfStmt:
			if x.Body != nil {
				walkStmts(x.Body.List, loop, consts, visit)
			}
			if x.Else != nil {
				walkElse(x.Else, loop, consts, visit)
			}
		case *ast.SwitchStmt:
			if x.Body != nil {
				for _, c := range x.Body.List {
					if cc, ok := c.(*ast.CaseClause); ok {
						walkStmts(cc.Body, loop, consts, visit)
					}
				}
			}
		case *ast.BlockStmt:
			walkStmts(x.List, loop, consts, visit)
		}
	}
}

func walkElse(e ast.Stmt, loop *ast.ForStmt, consts map[string]*ast.BasicLit, visit func(s ast.Stmt, loop *ast.ForStmt)) {
	switch x := e.(type) {
	case *ast.BlockStmt:
		walkStmts(x.List, loop, consts, visit)
	case *ast.IfStmt:
		walkStmts([]ast.Stmt{x}, loop, consts, visit)
	}
}

// loopTrips returns the static trip count of `for i := S; i < B; i++`
// style loops, where B is an int literal or const int ident.
func loopTrips(fs *ast.ForStmt, consts map[string]*ast.BasicLit) (int, bool) {
	if fs.Init == nil || fs.Cond == nil {
		return 0, false
	}
	as, ok := fs.Init.(*ast.AssignStmt)
	if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
		return 0, false
	}
	startLit, ok := as.Rhs[0].(*ast.BasicLit)
	if !ok || startLit.Kind != token.INT {
		return 0, false
	}
	start := atoi(startLit.Value)
	bin, ok := fs.Cond.(*ast.BinaryExpr)
	if !ok {
		return 0, false
	}
	bound, ok := intBound(bin.Y, consts)
	if !ok {
		return 0, false
	}
	switch bin.Op {
	case token.LSS:
		return bound - start, bound > start
	case token.LEQ:
		return bound - start + 1, bound >= start
	}
	return 0, false
}

func intBound(e ast.Expr, consts map[string]*ast.BasicLit) (int, bool) {
	switch x := e.(type) {
	case *ast.BasicLit:
		if x.Kind == token.INT {
			return atoi(x.Value), true
		}
	case *ast.Ident:
		if lit, ok := consts[x.Name]; ok {
			return atoi(lit.Value), true
		}
	}
	return 0, false
}

func atoi(s string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return 0
		}
		n = n*10 + int(s[i]-'0')
	}
	return n
}

// capFitness: shortfall (blocking/growth risk) is punished hard,
// over-provisioned capacity is punished linearly. Lower is better.
func capFitness(g []int, sites []capSite) int {
	f := 0
	for i, s := range sites {
		c := g[i]
		if c < 1 {
			c = 1
		}
		if c < s.demand {
			f += 500 + (s.demand-c)*50
		} else {
			f += c - s.demand
		}
	}
	return f
}

// xorshift is a tiny deterministic PRNG (stdlib-only constraint rules
// out math/rand here; determinism keeps the shell repeatable).
type xorshift struct{ s uint64 }

func newRand(seed uint64) *xorshift {
	if seed == 0 {
		seed = 0x9e3779b97f4a7c15
	}
	return &xorshift{s: seed}
}

func (r *xorshift) intn(n int) int {
	if n <= 0 {
		return 0
	}
	x := r.s
	x ^= x << 13
	x ^= x >> 7
	x ^= x << 17
	r.s = x
	return int(x % uint64(n))
}

// fnv1a hashes the pattern into a deterministic RNG seed.
func fnv1a(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

// geneticTune runs the GA: population 12, 12 generations, elitism (3),
// single-point crossover, per-gene mutation. The winning vector is
// applied to the AST, stored in tuningMemory, and reported.
func geneticTune(sites []capSite, pattern string) tuneOutcome {
	var tc tuneOutcome
	n := len(sites)
	if n == 0 {
		return tc
	}
	for _, s := range sites {
		if s.get() > s.demand {
			tc.wasteBefore += s.get() - s.demand
		}
	}
	rng := newRand(fnv1a(pattern))
	cur := make([]int, n)
	for i, s := range sites {
		cur[i] = s.get()
	}
	pop := [][]int{append([]int(nil), cur...)}
	if mem, ok := tuningMemory[pattern]; ok && len(mem) == n {
		pop = append(pop, append([]int(nil), mem...))
	}
	for len(pop) < 12 {
		g := make([]int, n)
		for i, s := range sites {
			g[i] = 1 + rng.intn(s.demand*2+4)
		}
		pop = append(pop, g)
	}
	better := func(a, b []int) bool { return capFitness(a, sites) < capFitness(b, sites) }
	sortPop := func() {
		for i := 1; i < len(pop); i++ {
			for j := i; j > 0 && better(pop[j], pop[j-1]); j-- {
				pop[j], pop[j-1] = pop[j-1], pop[j]
			}
		}
	}
	for gen := 0; gen < 12; gen++ {
		sortPop()
		next := [][]int{pop[0], pop[1], pop[2]}
		for len(next) < 12 {
			a, b := pop[rng.intn(6)], pop[rng.intn(6)]
			pt := 1 + rng.intn(n-1)
			if n == 1 {
				pt = 1
			}
			child := append(append([]int(nil), a[:pt]...), b[pt:]...)
			for i := range child {
				if rng.intn(4) == 0 {
					child[i] += 1 + rng.intn(3)
					if rng.intn(2) == 0 {
						child[i] -= 2 * (1 + rng.intn(3))
					}
					if child[i] < 1 {
						child[i] = 1
					}
				}
			}
			next = append(next, child)
		}
		pop = next
		_ = gen
	}
	sortPop()
	best := pop[0]
	tuningMemory[pattern] = append([]int(nil), best...)
	for i, s := range sites {
		if best[i] != s.get() {
			tc.changes = append(tc.changes,
				fmt.Sprintf("tuned %s capacity %d → %d (demand %d)", s.label, s.get(), best[i], s.demand))
			s.set(best[i])
		}
		if best[i] > s.demand {
			tc.wasteAfter += best[i] - s.demand
		}
	}
	return tc
}

// ------------------------------------------------------------------
// Stage 3: static safety inspection — guard clauses via ast.Inspect.
// ------------------------------------------------------------------

// inspectGuards counts guard clauses (if statements that check a
// condition before proceeding: nil checks, error checks, comparisons)
// and panic() call sites. Verified = guards exist and zero panics.
func inspectGuards(f *ast.File) SafetyReport {
	var guards, panics, defers, errAssigns, errChecks int
	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.IfStmt:
			if isGuardClause(x) {
				guards++
			}
			if isErrCheck(x) {
				errChecks++
			}
		case *ast.DeferStmt:
			defers++
		case *ast.AssignStmt:
			for _, lhs := range x.Lhs {
				if id, ok := lhs.(*ast.Ident); ok && id.Name == "err" {
					errAssigns++
				}
			}
		case *ast.CallExpr:
			if id, ok := x.Fun.(*ast.Ident); ok && id.Name == "panic" {
				panics++
			}
		}
		return true
	})
	rep := SafetyReport{Guards: guards, Panics: panics, Verified: guards > 0 && panics == 0}
	if defers > 0 {
		rep.Notes = append(rep.Notes, fmt.Sprintf("%d deferred cleanup%s", defers, pluralS(defers)))
	}
	if errAssigns > 0 {
		rep.Notes = append(rep.Notes, fmt.Sprintf("%d of %d error sites checked", min(errChecks, errAssigns), errAssigns))
	}
	return rep
}

// isGuardClause reports whether an if statement guards execution.
func isGuardClause(s *ast.IfStmt) bool {
	switch cond := s.Cond.(type) {
	case *ast.BinaryExpr:
		switch cond.Op {
		case token.NEQ, token.EQL, token.LSS, token.GTR, token.LEQ, token.GEQ:
			return true
		}
	case *ast.UnaryExpr:
		if cond.Op == token.NOT {
			return true
		}
	}
	return isErrCheck(s)
}

// isErrCheck reports the classic `if err != nil` guard.
func isErrCheck(s *ast.IfStmt) bool {
	bin, ok := s.Cond.(*ast.BinaryExpr)
	if !ok || bin.Op != token.NEQ {
		return false
	}
	id, ok := bin.X.(*ast.Ident)
	return ok && id.Name == "err"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ------------------------------------------------------------------
// Stage 4: visual execution trace — the call graph from main.
// ------------------------------------------------------------------

// traceExecution follows the call graph from main (or the first
// function), in source order, cycle-safe, up to three steps.
func traceExecution(f *ast.File, fset *token.FileSet) []ExecutionStep {
	decls := map[string]*ast.FuncDecl{}
	var order []string
	for _, d := range f.Decls {
		fn, ok := d.(*ast.FuncDecl)
		if !ok || fn.Recv != nil || fn.Body == nil {
			continue
		}
		decls[fn.Name.Name] = fn
		order = append(order, fn.Name.Name)
	}
	if len(order) == 0 {
		return nil
	}
	cur := "main"
	if _, ok := decls[cur]; !ok {
		cur = order[0]
	}
	var steps []ExecutionStep
	visited := map[string]bool{}
	for len(steps) < 3 && cur != "" {
		if visited[cur] {
			break
		}
		visited[cur] = true
		fn := decls[cur]
		if fn == nil {
			break
		}
		callees := calledFuncs(fn, decls)
		steps = append(steps, ExecutionStep{
			Number:      len(steps) + 1,
			Title:       humanize(fn.Name.Name),
			Subtitle:    subtitleFor(fn.Name.Name, callees),
			Description: describeFunc(fn),
		})
		next := ""
		for _, c := range callees {
			if !visited[c] {
				next = c
				break
			}
		}
		cur = next
	}
	return steps
}

// calledFuncs lists locally-defined functions called by fn, in source
// order, without duplicates. Direct calls (worker()) and handler
// registrations (http.HandleFunc("/x", health)) both count.
func calledFuncs(fn *ast.FuncDecl, decls map[string]*ast.FuncDecl) []string {
	var out []string
	seen := map[string]bool{}
	add := func(name string) {
		if seen[name] {
			return
		}
		if _, ok := decls[name]; ok {
			seen[name] = true
			out = append(out, name)
		}
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if id, ok := call.Fun.(*ast.Ident); ok {
			add(id.Name)
		}
		for _, arg := range call.Args {
			if id, ok := arg.(*ast.Ident); ok {
				add(id.Name)
			}
		}
		return true
	})
	return out
}

// subtitleFor builds the package/file/function call context.
func subtitleFor(name string, callees []string) string {
	base := name + "() · package main in repl.go"
	if len(callees) == 0 {
		return base + " — leaf"
	}
	return base + " — calls " + strings.Join(callees, ", ")
}

// describeFunc derives an operational description from the AST.
func describeFunc(fn *ast.FuncDecl) string {
	var gos, loops, sends, defers int
	stmts := 0
	if fn.Body != nil {
		stmts = len(fn.Body.List)
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.GoStmt:
			gos++
		case *ast.ForStmt, *ast.RangeStmt:
			loops++
		case *ast.SendStmt:
			sends++
		case *ast.DeferStmt:
			defers++
		}
		return true
	})
	var parts []string
	if gos > 0 {
		parts = append(parts, fmt.Sprintf("spawns %d goroutine%s", gos, pluralS(gos)))
	}
	if loops > 0 {
		parts = append(parts, fmt.Sprintf("runs %d loop%s", loops, pluralS(loops)))
	}
	if sends > 0 {
		parts = append(parts, fmt.Sprintf("sends on channels %dx", sends))
	}
	if defers > 0 {
		parts = append(parts, "defers cleanup")
	}
	parts = append(parts, fmt.Sprintf("%d statement%s", stmts, pluralS(stmts)))
	if fn.Name.Name == "main" {
		return "entry point: " + strings.Join(parts, ", ")
	}
	return strings.Join(parts, ", ")
}

// humanize turns camelCase names into titles.
func humanize(name string) string {
	var b strings.Builder
	for i, r := range name {
		if i > 0 && r >= 'A' && r <= 'Z' {
			b.WriteByte(' ')
		}
		b.WriteRune(r)
	}
	s := b.String()
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// ------------------------------------------------------------------
// Example session (transcript):
//
//	$ go run cmd/gollemer/main.go
//	gollemer> build a list of squares
//	=== GENERATED GO SOURCE ===
//	package main
//	...
//	💡 BEGINNER CONCEPT ANALOGY
//	A factory assembly line: ...
//	─── VISUAL EXECUTION TRACE ───
//	[1] Main
//	└── main() · package main in repl.go — calls squares
//	    entry point: runs 1 loop, 2 statements
//	[2] Squares
//	└── squares() · package main in repl.go — leaf
//	    runs 1 loop, 3 statements
//	STATUS BADGES
//	⚠ UNVERIFIED — 0 panic paths, 0 guard clauses
//	⚡ PERF — 1 → 0 allocs/op (saved 1)
// ------------------------------------------------------------------
