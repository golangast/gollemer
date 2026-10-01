package chat

// The training-data loop: the mechanism behind "remember old data if it is
// good data".
//
// All social training pairs live in ONE file: data/training/chat_pairs.jsonl
// (one JSON object per line). Adding data never fine-tunes on just the new
// pairs — every training run loads the whole file and retrains from scratch
// on old + new together (rehearsal), which is what prevents the model from
// forgetting what it already learned.
//
// New pairs pass through ValidatePair before they are admitted:
//   - non-empty input and output
//   - 1..40 whitespace-separated tokens on each side
//   - input and output must differ
//   - no exact duplicate of a pair already in the file
//   - no control characters
//   - no mixed chatter+code outputs ("I am func doing (x)"): an output that
//     looks like code must not contain natural-language chatter, and a
//     gocode-domain output must contain code at all
// Pairs that fail are quarantined (reported, never silently dropped) so bad
// data can never poison the pool.

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

// ChatPair is one training example: a user input and the response to learn.
// Domain tags what the pair teaches ("social", "go", "makefile", ...). The roadmap trains
// one stage at a time, so the trainer filters by domain — a social model never
// sees Go Q&A and vice versa.
type ChatPair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
	Domain string `json:"domain,omitempty"`
}

// SocialDomain is the default domain: everyday conversation.
const SocialDomain = "social"

// MakefileDomain is stage two of the roadmap: guessing makefile commands.
const MakefileDomain = "makefile"

// GocodeDomain is stage three of the roadmap: generating Go code from a
// natural-language request.
const GocodeDomain = "gocode"

// GoDomain is stage four of the roadmap: answering questions about Go
// concepts, terminology, and workflow (files/folders, main function,
// go commands). Distinct from GocodeDomain: this is prose explanations,
// not code generation.
const GoDomain = "go"

// GoCliDomain is stage five of the roadmap: the Go command domain. The
// input is a natural-language request to do something with the Go
// toolchain ("check if the dependencies are updated"); the output is the
// exact CLI command ("go mod tidy"), which unified chat can run on
// confirmation. Distinct from GoDomain (prose explanations of what
// commands do) and GocodeDomain (generating Go code): modes stay
// separate by design.
const GoCliDomain = "gocli"

// GoAnalyzeDomain is the codebase-reader brain: it parses a Go project
// with go/ast and reports how the codebase works — entry points, the
// most-called functions, package dependencies, a suggested reading order,
// and where to change things for a given task. Unlike the other brains
// it is fully deterministic (no neural model): analysis is exact, so a
// trained model would only add hallucinations.
const GoAnalyzeDomain = "goanalyze"

// UnifiedDomain is the unified chat mode: a single chat session that
// routes each message to the appropriate domain model (social, go,
// gocli, gocode, makefile) based on input intent.
const UnifiedDomain = "unified"

// makefileTerms marks the makefile-command domain: the word "makefile"
// itself or one of the repo's known make target names. Checked before
// goTerms on purpose: outputs like "run make train-real-seq2seq" contain
// the word "train", and we want the makefile tag to win.
var makefileTerms = regexp.MustCompile(`(?i:\bmakefile\b)|\b(train-real-seq2seq|real-chat|train-resume|train-fresh|train-small-seq2seq|test-small-seq2seq|chat-makefile|makefile-train|makefile-pb|import-pairs|reclassify-domains|conversing-pb|social-replies-pb|tech-multiturn-pb|clean-all|install-hooks|export-labels|all-pb)\b|(?i:\bmake\s+(clean|chat|train|test|sel|metrics)\b)`)

// gocodeTerms marks the code-generation domain: the input asks for code to
// be written, or the output is a Go snippet (a func declaration, a package
// clause, or a := assignment). Checked before goTerms on purpose: code
// outputs contain words like "func" and "struct" that would otherwise tag
// them as Go Q&A.
var gocodeTerms = regexp.MustCompile(`(?i:\bwrite\b[^.]{0,40}\b(function|func|code|program|method|struct)\b)|\bfunc\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(|package main|:=`)

// goTerms marks the Go-programming domain. Rule-based on purpose: data
// curation is a human judgment, and this keeps it visible and auditable.
// NOTE: the bare word "go" is matched case-sensitively only — the language
// name is capitalized in this dataset ("use Go"), while the verb is not
// ("how did your day go"). Plurals (channels, slices) are included.
var goTerms = regexp.MustCompile(`(?i:\b(goroutines?|closures?|defers?|structs?|interfaces?|slices?|channels?|modules?|cgo|generics?|packages?|funcs?|functions?|methods?|race conditions?|compil\w*|architect\w*|waitgroups?|pointers?|executables?|binar\w*|mutex\w*|documentation|profiling|builtins?|error handling|builds?|panics?|blank identifiers?|anonymous functions?|select|gomaxprocs|gofmt|godoc|delete|maps?|printf|sprintf|unmarshal|marshal|pipelines?|fan[- ]?out|fan[- ]?in|done (channels?|signals?)|context cancellation|errors?|goproxy|gosumdb|testmain|workspaces?|shadowing|loop variables?|zero values?|init functions?|benchmarks?|testdata|golden files?|work stealing|replace directive|minimal version|blank imports?|dot imports?|producers?|consumers?|config|standard input|stdin|logs|sync\.?\s*cond|time\.?\s*after|flaky|deferred|integer division|json|float64|slicing|concurrency|context|imports?|t\s+helper|t\s+cleanup|test coverage|from go|time values|string of an int|empty array|v2\b|test my code|tasks|configuration|cached|parallel|share.*value|read heavy|first of several|test pass|fail together|skip.*test|lines.*tests?)\b|\bin go\b|\bgo (program|programs|string|server|language|programming)\b|\b(can|does) go\b|\bwhat is go\b|\blearn(ing)? go\b|\bfmt\b|\berrors\.\w+|\bt\.(helper|cleanup)\b|\binit\b.*\bmain\b)|\bdepende|\bnew\(\)|\bmake\(\)|\bGo\b`)

// ClassifyDomain tags a pair by its content.
func ClassifyDomain(input, output string) string {
	if makefileTerms.MatchString(input) || makefileTerms.MatchString(output) {
		return MakefileDomain
	}
	// Command requests route to gocli before the prose domains: the
	// router predicate already carves out concept questions ("what does
	// go mod tidy do"), so reusing it keeps reclassify consistent.
	if isGoCliRequest(input) {
		return GoCliDomain
	}
	if gocodeTerms.MatchString(input) || gocodeTerms.MatchString(output) {
		return GocodeDomain
	}
	if goTerms.MatchString(input) || goTerms.MatchString(output) {
		return "go"
	}
	return SocialDomain
}

// ChatDatasetRelPath is the dataset location relative to the project root.
const ChatDatasetRelPath = "data/training/chat_pairs.jsonl"

// maxPairTokens bounds each side of a pair; longer texts are not social
// chat and would destabilize the tiny model.
const maxPairTokens = 40

// normalizePairKey canonicalizes a pair for duplicate detection.
func normalizePairKey(p ChatPair) string {
	norm := func(s string) string {
		return strings.Join(strings.Fields(strings.ToLower(s)), " ")
	}
	return norm(p.Input) + "\n" + norm(p.Output)
}

// codeOutputMarkers detects an output that looks like Go code: braces, a
// := assignment, or a func/package/type keyword.
var codeOutputMarkers = regexp.MustCompile(`[{}]|:=|\bfunc\b|\bpackage\b|\btype\b`)

// chatterPhrases detects natural-language chatter: first-person framing,
// offers, and pleasantries that have no place inside generated code.
var chatterPhrases = regexp.MustCompile(`(?i)\b(i am|i'm|here is|here's|sure|of course|hope this helps|let me know|you're welcome|no problem|happy to help)\b`)

// stringLiteral strips double-quoted string literals so a "hello" inside
// "hello world" is not mistaken for chatter.
var stringLiteral = regexp.MustCompile(`"[^"]*"`)

// stripStringLiterals removes "..." spans from s.
func stripStringLiterals(s string) string {
	return stringLiteral.ReplaceAllString(s, "")
}

// ValidatePair is the quality gate. seen holds normalizePairKey values of the
// pairs already admitted. It returns nil when the pair is good data worth
// remembering, or a human-readable reason when it must be quarantined.
func ValidatePair(p ChatPair, seen map[string]bool) error {
	in := strings.TrimSpace(p.Input)
	out := strings.TrimSpace(p.Output)
	if in == "" {
		return fmt.Errorf("empty input")
	}
	if out == "" {
		return fmt.Errorf("empty output")
	}
	inTok := len(strings.Fields(in))
	outTok := len(strings.Fields(out))
	if inTok < 1 || outTok < 1 {
		return fmt.Errorf("needs at least 1 token per side")
	}
	if inTok > maxPairTokens || outTok > maxPairTokens {
		return fmt.Errorf("too long (%d/%d tokens, max %d per side)", inTok, outTok, maxPairTokens)
	}
	if strings.EqualFold(in, out) {
		return fmt.Errorf("input and output are identical")
	}
	// NOTE: in and out are checked separately — the "\n" separator used in the
	// duplicate-detection key is itself a control character.
	for _, s := range in + out {
		if unicode.IsControl(s) {
			return fmt.Errorf("contains control characters")
		}
	}
	if seen[normalizePairKey(p)] {
		return fmt.Errorf("duplicate of an existing pair")
	}
	// Mode separation: an output must never blend natural-language chatter
	// with code ("I am func doing (x)"). String literals are stripped first
	// so the "hello" in "hello world" doesn't count as chatter.
	bare := stripStringLiterals(out)
	isCode := codeOutputMarkers.MatchString(bare)
	isChatter := chatterPhrases.MatchString(bare)
	if isCode && isChatter {
		return fmt.Errorf("output mixes natural-language chatter with code")
	}
	// A gocode-domain pair teaches code generation: its output must contain
	// code, not a chat reply.
	if p.Domain == GocodeDomain && !isCode {
		return fmt.Errorf("gocode output contains no code")
	}
	return nil
}

// LoadChatDataset reads the JSONL dataset. Missing file = empty dataset, nil error.
func LoadChatDataset(path string) ([]ChatPair, error) {
	return loadChatPairs(path, SocialDomain)
}

// loadChatPairs reads a JSONL pair file. When defaultDomain is non-empty,
// pairs without an explicit domain tag get it; when empty, the domain is
// left blank so callers (like the import gate) can classify for themselves.
func loadChatPairs(path string, defaultDomain string) ([]ChatPair, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()
	var pairs []ChatPair
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var p ChatPair
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			return nil, fmt.Errorf("dataset line %d: %w", lineNo, err)
		}
		if p.Domain == "" {
			p.Domain = defaultDomain
		}
		pairs = append(pairs, p)
	}
	return pairs, sc.Err()
}

// SaveChatDataset writes pairs as JSONL, creating parent directories.
func SaveChatDataset(path string, pairs []ChatPair) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	for _, p := range pairs {
		line, err := json.Marshal(p)
		if err != nil {
			return err
		}
		if _, err := w.Write(append(line, '\n')); err != nil {
			return err
		}
	}
	return w.Flush()
}

// ChatDatasetPath resolves the dataset file under the project root.
func ChatDatasetPath(projectRoot string) string {
	return filepath.Join(projectRoot, ChatDatasetRelPath)
}

// ReclassifyDomains re-runs ClassifyDomain over every pair in the dataset and
// saves the result. Used when the classifier itself improves — the data stays,
// the tags get smarter.
func ReclassifyDomains(projectRoot string) (map[string]int, error) {
	path := ChatDatasetPath(projectRoot)
	pairs, err := LoadChatDataset(path)
	if err != nil {
		return nil, err
	}
	counts := map[string]int{}
	for i := range pairs {
		pairs[i].Domain = ClassifyDomain(pairs[i].Input, pairs[i].Output)
		counts[pairs[i].Domain]++
	}
	if err := SaveChatDataset(path, pairs); err != nil {
		return nil, err
	}
	return counts, nil
}

// seedPair is a Q&A pair loaded from the legacy seed CSV.
type seedPair struct {
	Q string
	A string
}

// loadSeedPairs reads Q&A pairs from a CSV file (header + rows).
// It replaces the old moe-dependent loader; only used when seeding
// a missing dataset from the built-in CSV.
func loadSeedPairs(dataPath string) ([]seedPair, error) {
	f, err := os.Open(dataPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	records, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("seed dataset is empty")
	}
	pairs := make([]seedPair, 0, len(records)-1)
	for _, rec := range records[1:] {
		if len(rec) < 2 {
			continue
		}
		q, a := strings.TrimSpace(rec[0]), strings.TrimSpace(rec[1])
		if q == "" || a == "" {
			continue
		}
		pairs = append(pairs, seedPair{Q: q, A: a})
	}
	return pairs, nil
}

// seedDataPath resolves the built-in seed dataset.
func seedDataPath(projectRoot string) string {
	return filepath.Join(projectRoot, "data", "training", "trainingdata", "conversing.csv")
}

// SeedChatDataset populates the JSONL dataset from the built-in conversing.pb
// pairs the first time it is missing. It never overwrites an existing file:
// John's data is append-only by design.
func SeedChatDataset(projectRoot string) (int, error) {
	path := ChatDatasetPath(projectRoot)
	if _, err := os.Stat(path); err == nil {
		existing, err := LoadChatDataset(path)
		if err != nil {
			return 0, err
		}
		return len(existing), nil // already seeded; report current size
	}
	raw, err := loadSeedPairs(seedDataPath(projectRoot))
	if err != nil {
		return 0, fmt.Errorf("seed: load built-in pairs: %w", err)
	}
	seen := map[string]bool{}
	pairs := make([]ChatPair, 0, len(raw))
	for _, rp := range raw {
		p := ChatPair{Input: rp.Q, Output: rp.A, Domain: ClassifyDomain(rp.Q, rp.A)}
		if err := ValidatePair(p, seen); err != nil {
			continue // quarantine quietly at seed time; the gate logs on import
		}
		seen[normalizePairKey(p)] = true
		pairs = append(pairs, p)
	}
	if err := SaveChatDataset(path, pairs); err != nil {
		return 0, err
	}
	return len(pairs), nil
}

// ImportChatPairs validates every pair in an import JSONL file and appends the
// good ones to the dataset. It returns the number admitted and a list of
// human-readable quarantine reports ("line 12: too long ...") for the rest.
func ImportChatPairs(projectRoot, importPath string) (admitted int, quarantined []string, err error) {
	incoming, err := loadChatPairs(importPath, "")
	if err != nil {
		return 0, nil, fmt.Errorf("read import file: %w", err)
	}
	path := ChatDatasetPath(projectRoot)
	existing, err := LoadChatDataset(path)
	if err != nil {
		return 0, nil, fmt.Errorf("read dataset: %w", err)
	}
	seen := map[string]bool{}
	for _, p := range existing {
		seen[normalizePairKey(p)] = true
	}
	for i, p := range incoming {
		if p.Domain == "" {
			p.Domain = ClassifyDomain(p.Input, p.Output)
		} else {
			p.Domain = strings.ToLower(strings.TrimSpace(p.Domain))
		}
		if verr := ValidatePair(p, seen); verr != nil {
			quarantined = append(quarantined, fmt.Sprintf("pair %d: %v", i+1, verr))
			continue
		}
		seen[normalizePairKey(p)] = true
		existing = append(existing, ChatPair{
			Input:  strings.TrimSpace(p.Input),
			Output: strings.TrimSpace(p.Output),
			Domain: p.Domain,
		})
		admitted++
	}
	if admitted > 0 {
		if err := SaveChatDataset(path, existing); err != nil {
			return 0, quarantined, fmt.Errorf("save dataset: %w", err)
		}
	}
	return admitted, quarantined, nil
}

// goStdlibQuestion marks "how do i" questions about stdlib usage as
// concept questions: "how do i get the current time in go" asks for
// time.Now, not a program to be written. Checked before gocodeCodegen,
// which would otherwise catch these via its "how do i ... in go" clause.
var goStdlibQuestion = regexp.MustCompile(`(?i:\bhow do i\b.*\b(string|strings|current time|working directory|error|errors|file extension|cpus|timeout|channel receive|sorted)\b.*\bin go\b)`)

// gocodeCodegen marks explicit code-generation intent: the input asks for
// code to be written. This is a subset of gocodeTerms used for routing:
// a question like "what is package main" matches gocodeTerms (via the
// literal "package main") but is a concept question, not a code request,
// so it routes to GoDomain.
var gocodeCodegen = regexp.MustCompile(`(?i:^\s*(code|check|add|sum|build|compute|print|multiply|count|find|double|total)\b)|\b(write|give me|i need|i want|create|generate|define|declare|build|compute|print|multiply|count|find|make me|make a|make an)\b[^.]{0,40}\b(function|func|code|program|method|struct|main|check|checker|helper|type|adder|maker|factorial|loop)\b|\ba\s+program\s+that\b|\bfunction\s+that\b|\bprogram\b.*\bpackage main\b|\bsay hello\b.*\bin go\b|\bfactorial\b.*\bcode\b|\bcode\b.*\bfactorial\b|\btell me if\b.*\b(divides|is)\b|\bi want to\b.*\b(add|sum|divide|multiply)\b|\bfunction\s+for\b|\bwith\s+(a\s+)?function\b|\bwith\s+code\b|\bmaker\b.*\bin go\b|\b(checker)\b.*\bin go\b|\b(find|get)\b.*\b(largest|smallest|biggest|maximum|minimum)\b|\b(biggest|largest|smallest|greater|total)\b.*\bof\b|\bwhich of two\b|\b(division|addition|subtraction|multiplication)\b.*\bin go\b|\bhow do i\b.*\btest\b.*\bif\b|\bhow do i\b.*\b(get|find|compute|declare|add|sum|divide|total|multiply|count|check)\b.*\bin go\b|\bfunc\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(|:=|\bgo\s+(code|function|func)\b|\bin\s+go\b.*\b(function|func|code|type|struct)\b`)

// questionForm marks inputs phrased as questions about a concept rather
// than requests to produce code.
var questionForm = regexp.MustCompile(`(?i:^\s*(what|why|how|explain|tell me|describe|when|where|which|who)\b)`)

// goWorkflow marks Go workflow questions (modules, dependencies, build
// commands) that use action verbs like "build" or "add" but are NOT code
// generation requests. Checked before gocodeCodegen: "build every package
// in a Go module" is a `go build` question, not a "write a builder" request.
var goWorkflow = regexp.MustCompile(`\bGo module\b|\bGo binary\b|\bdependency\b|\bdependencies\b|\bbetween new and make\b`)

// goCommandLiteral marks an explicit Go toolchain command in the input.
// A bare invocation ("go mod tidy", "gofmt -w .") is a run request for
// GoCliDomain; the same literal inside a question ("what does go mod
// tidy do") is a concept question for GoDomain — see isGoCliRequest.
var goCommandLiteral = regexp.MustCompile(`(?i:\bgo\s+(mod\s+(init|tidy|download|verify|graph)|get\b|build\b|run\b|test\b|vet\b|list\b|doc\b|env\b|version\b|install\b|clean\b)|\bgofmt\b)`)

// runOneTestQuestion marks the explanatory "how do you run one test"
// phrasing, which asks what the command looks like (GoDomain) rather
// than asking to run the suite (GoCliDomain).
var runOneTestQuestion = regexp.MustCompile(`(?i:\brun\b.*\b(one|single)\b.*\btest\b)`)

// gocliQuestionCarveout marks explanatory questions about commands:
// "what does go mod tidy do", "what is the difference between go run
// and go build", "how do you run one test in Go", "why did go test say
// cached". These ask what a command does or why it behaves a way
// (GoDomain); they never ask to run one. Deliberately limited to
// what/how/why: "which go version is installed" and "where is
// the go module cache" ARE run requests (GoCliDomain).
var gocliQuestionCarveout = regexp.MustCompile(`(?i:^\s*(what|how|why)\b)`)

// gocliExplainMarker marks the phrasing that turns a command mention
// into a concept question: "what does go mod tidy do", "the difference
// between go run and go build", "what does gofmt mean". A bare "do" is
// deliberately excluded: "what go version do I have" is a run request
// (go version), not an explanation request.
var gocliExplainMarker = regexp.MustCompile(`(?i:\bdoes\b|\bdifference between\b|\bmeans?\b|\bmeant\b|\bexplain\b)`)

// isGoCommandQuestion reports whether the input asks what a Go command
// does ("what does go mod tidy do", "how do you run one test in Go").
// These are GoDomain concept questions, never run requests. A what/how
// question that merely names a command ("what go version do I have")
// stays a run request.
func isGoCommandQuestion(input string) bool {
	if !gocliQuestionCarveout.MatchString(input) {
		return false
	}
	if !goCommandLiteral.MatchString(input) && !runOneTestQuestion.MatchString(input) {
		return false
	}
	return gocliExplainMarker.MatchString(input) || runOneTestQuestion.MatchString(input)
}

// gocliIntent marks natural-language requests to RUN a Go toolchain
// command: "check if the dependencies are updated" -> go mod tidy.
// Deliberately narrow: bare verbs like "build" or "check" only count
// with a toolchain object, so "build a web server" (gocode) and "check
// the weather" (social) never match.
var gocliIntent = regexp.MustCompile(`(?i:` +
	`\btidy\b.*\b(dependenc|deps|go\.mod|module)\b|\bgo\.mod\b.*\btidy\b` +
	`|\b(dependenc(y|ies)|deps)\b.*\b(tidy|updated?|up to date|clean ?up|sync(ed)?|download(ed)?|fetch(ed)?|verify|tampered|intact|remove(d)?)\b` +
	`|\b(tidy|sync|download|fetch|verify|clean ?up|remove)\b.*\b(dependenc(y|ies)|deps)\b` +
	`|\b(show|list)\b.*\b(dependenc\w*|deps)\b` +
	`|\b(dependenc\w*|deps)\b.*\bgraph\b|\bgraph\b.*\b(dependenc\w*|deps)\b` +
	`|\blist\b.*\bpackages\b` +
	`|\buuid\b` +
	`|\bbuild\b.*\b(my|the|this|our|every|all)\b.*\b(program|project|binary|packages|module)\b` +
	`|\bcompile\b.*\b(program|project|packages|module|everything|code)\b` +
	`|\bcreate\b.*\bexecutable\b` +
	`|\brun\b.*\b(my|the|this)\b.*\b(program|package)\b|\bexecute\b.*\bprogram\b` +
	`|\b(run|execute)\b.*\btests?\b|\btest\b.*\bpackages\b` +
	`|\bcheck\b.*\btest\b.*\bcoverage\b|\bcoverage\b.*\b(check|run)\b` +
	`|\brace\b.*\btests?\b|\btests?\b.*\brace\b` +
	`|\bvet\b|\bcode\b.*\bfor\b.*\bproblems\b|\bsuspicious\b` +
	`|\bformat(s|ted)?\b.*\b(code|go files?|source files?)\b|\bcode\b.*\bformat(ted)?\b|\bformatting\b.*\bcode\b|\breformat\b` +
	`|\bgo doc\b|\bdocs?\b.*\bfor\b|\bdocumentation\b.*\bfor\b` +
	`|\bgo version\b|\bversion\b.*\binstalled\b` +
	`|\bgo env\b|\benvironment settings\b|\bmodule cache\b` +
	`|\binstall\b.*\b(binary|command|program|path)\b` +
	`|\b(clean|clear)\b.*\b(build )?cache\b` +
	`|\b(init|initiali[sz]e)\b.*\b(module|project)\b` +
	`|` + `^\s*test\s+my\s+code\b` +
	`)`)

// isGoCliRequest reports whether the input asks to RUN a Go toolchain
// command. Concept questions about commands ("what does go mod tidy do",
// "how do you run one test in Go", "why did go test say cached") are
// GoDomain, not run requests. But "what go version do I have" IS a run
// request (it wants the output of `go version`), so have/has/installed
// questions are never carved out. The bare imperative "test my code" is
// a run request; the question "How do I test my code?" stays where the
// dataset put it (social) because the pattern is anchored to the bare
// imperative.
func isGoCliRequest(input string) bool {
	if isGoCommandQuestion(input) {
		return false
	}
	// A what/how/why question that mentions Go tooling asks for an
	// explanation, not a command run: "why did go test say cached",
	// "how do i run tests in parallel". Exception: "what go version
	// do I have" wants the command output, not an explanation. And
	// "how do I run the tests" is a run request (gocli) unless it has
	// a concept qualifier like "parallel" — hence the goTerms check.
	if gocliQuestionCarveout.MatchString(input) &&
		(goCommandLiteral.MatchString(input) || gocliIntent.MatchString(input)) &&
		!gocliHaveRequest.MatchString(input) &&
		goTerms.MatchString(input) {
		return false
	}
	return goCommandLiteral.MatchString(input) || gocliIntent.MatchString(input)
}

// gocliHaveRequest marks "what go version do I have" style questions as
// run requests: the user wants the command output, not an explanation.
var gocliHaveRequest = regexp.MustCompile(`(?i:\b(have|has|installed)\b)`)

// routeDomain classifies a chat INPUT (not the output) into the domain
// model that should handle it. Used by unified chat. Tuned for perfect
// coverage on the training inputs: social, gocode, makefile, go, gocli.
func routeDomain(input string) string {
	// Definition questions about Gollemer itself ("what is gollemer",
	// "who made you") are conversational, not make-command requests,
	// even though they name Gollemer. Checked before makefileIntent.
	if gollemerDefinition.MatchString(input) {
		return SocialDomain
	}
	// "what does make chat do" asks for an explanation of a command,
	// not a request to run it. Checked before makefileTerms.
	if makeExplain.MatchString(input) {
		return SocialDomain
	}
	// "how do i add training data" asks for an explanation, not a run.
	if trainingDataQuestion.MatchString(input) {
		return SocialDomain
	}
	// "analyze this go project" / "where would I add a chat command" are
	// codebase-reading requests for the goanalyze brain, not makefile or
	// Go-concept questions.
	if analyzeIntent.MatchString(input) {
		return GoAnalyzeDomain
	}
	// "analyze ~/projects/foo", "analyze ./cmd/tool": the analyze verb
	// plus a path-shaped token is a codebase request even when the path
	// names no codebase noun ("tinyproj"). extractAnalyzePath only finds
	// real path shapes (/, ~, quoted, github.com), so "analyze the
	// training dataset" still falls through to the makefile brain.
	if analyzeVerb.MatchString(input) && extractAnalyzePath(input) != "" {
		return GoAnalyzeDomain
	}
	if makefileTerms.MatchString(input) || makefileIntent.MatchString(input) {
		return MakefileDomain
	}
	// A question about what a Go command does is a concept question,
	// not a run request: "what does go mod tidy do" -> GoDomain.
	if isGoCommandQuestion(input) {
		return GoDomain
	}
	// A request to run a Go toolchain command goes to the command model.
	// Checked before goWorkflow/gocodeCodegen: "check if the dependencies
	// are updated" is a `go mod tidy` request, not a concept question,
	// and "build my program" is a `go build` request, not codegen.
	if isGoCliRequest(input) {
		return GoCliDomain
	}
	// Go workflow questions use action verbs but aren't code requests.
	if goWorkflow.MatchString(input) {
		return GoDomain
	}
	// "how do i" questions about stdlib usage are concept questions,
	// not codegen: "how do i get the current time in go".
	if goStdlibQuestion.MatchString(input) {
		return GoDomain
	}
	// Explicit code-generation request always wins for code.
	if gocodeCodegen.MatchString(input) {
		return GocodeDomain
	}
	// A question about a Go concept (even one mentioning "package main"
	// or "func") goes to the concept model, not the code generator.
	if goTerms.MatchString(input) {
		return GoDomain
	}
	// Bare code fragments outside a question go to the code model.
	if gocodeTerms.MatchString(input) {
		return GocodeDomain
	}
	return SocialDomain
}

// analyzeIntentPatterns mark "read this codebase" requests: the user wants
// Gollemer to parse a Go project and explain its structure, entry points,
// and where to change things. Every pattern pairs a reading verb with a
// codebase noun so bare \bmap\b (a Go builtin all over the training data)
// can never match. Checked before makefileTerms and goTerms: "analyze
// this go project" names Go but is a reading request, not a concept
// question.
var analyzeIntentPatterns = []string{
	`\b(analyse|analyze|analyzing|analysing)\b.*\b(project|codebase|repo|repository|source|package|directory|folder|this)\b`,
	`\b(analyse|analyze)\b.*\bgithub\.com\b`,
	`\b(map|tour|walkthrough|overview)\b.*\b(codebase|repo|repository|project)\b`,
	`\bwalk me through\b.*\b(code|repo|project|codebase)\b`,
	`\bhow\b.*\b(codebase|repo|repository)\b.*\b(work|structured|organized|architected)\b`,
	`\bwhere\b.*\b(start|begin)\b.*\b(codebase|repo|repository|project)\b`,
	`\bread\b.*\b(codebase|repo|repository|source)\b`,
	// "where would I add a chat command" — a change-location question.
	// No training input asks where-to-change, so the bare form is safe;
	// the handler defaults to the current project when no path is given.
	`\bwhere\b.*\b(add|change|update|modify|fix|edit|put|implement|wire|hook)\b`,
	// "show me a visual of this project" / "draw a diagram of the repo":
	// a visual noun aimed at the codebase. "graph" stays scoped to
	// codebase nouns so "write a graph traversal" (gocode) never matches.
	`\b(visual|diagram|picture)\b.*\b(codebase|repo|repository|project|code)\b`,
	`\bgraph\b.*\b(codebase|repo|repository|project)\b`,
}

// analyzeVerb marks the bare reading verb: "analyze ~/projects/foo".
// Paired with extractAnalyzePath in routeDomain so a path that names no
// codebase noun ("tinyproj") still routes to the goanalyze brain.

// analyzeIntent is the compiled union of analyzeIntentPatterns.
var analyzeIntent = regexp.MustCompile(`(?i:` + strings.Join(analyzeIntentPatterns, "|") + `)`)

// analyzeVerb is the bare reading verb on its own.
var analyzeVerb = regexp.MustCompile(`(?i:\b(analyse|analyze|analysing|analyzing)\b)`)

// makeExplain marks "what does make X do" questions as conversational.
// The user wants an explanation of the command, not a request to run it.
var makeExplain = regexp.MustCompile(`(?i:^\s*what\s+does\s+make\s+\w+)`)

// trainingDataQuestion marks "how do i add training data" as
// conversational. The user wants an explanation of the process, not a
// request to run a command. Checked before makefileIntent, whose bare
// \btraining\b would otherwise catch it.
var trainingDataQuestion = regexp.MustCompile(`(?i:\bhow do i\b.*\btraining data\b)`)

// gollemerDefinition marks "what is gollemer" style questions as
// conversational. Without this, makefileIntent's \bgollemer\b would
// route them to the make-command model.
var gollemerDefinition = regexp.MustCompile(`(?i:\b(what is|what's|who made|who created|who built|tell me about|how many|how does)\b.*\bgollemer\b|\bgollemer\b.*\b(what is|who made|how many)\b|\b(get started|expand)\b.*\bgollemer\b|\bgollemer\b.*\b(learn|brains)\b)`)

// makefileIntent marks natural-language requests to operate the Gollemer
// repo itself (train the model, chat with it, manage checkpoints/dataset,
// list make targets). These rarely name a make target explicitly, so the
// explicit makefileTerms above miss them. Checked before the social
// fallback: social chatter never mentions training, checkpoints, or the
// dataset.
//
// The patterns are grouped by intent so new ones can be added without
// untangling a single giant expression. They're joined with | below.
var makefileIntentPatterns = []string{
	// Train/retrain the model: "retrain the neural model", "resume training".
	// NOTE: bare \btrain\b is deliberately NOT matched: it's a vehicle
	// ("train rides") as often as a verb. The specific patterns above
	// cover "train the model"; bare \btraining\b and \bretrain\b are
	// unambiguous (but see trainingDataQuestion: "how do i add training
	// data" is an explanation question, not a command).
	`\b(train|training|retrain|resume|continue)\b.*\b(model|tiny|small|neural|gollemer|checkpoint)\b`,
	`\b(resume|continue)\b.*\btraining\b`,
	`\btraining\b.*\b(resume|continue)\b`,
	`\btraining\b`,
	`\bretrain\b`,
	`\btrain\b.*\b(anew|from scratch)\b`,
	`\b(wipe|reset|scrap)\b.*\btrain\b`,
	`\b(tiny|small)\b.*\bmodel\b`,

	// Manage checkpoints: "clean the checkpoints", "delete old models".
	`\b(checkpoints?|models?)\b.*\b(clean|clear|wipe|delete|remove|purge|erase|tidy|old)\b`,
	`\b(clean|clear|wipe|delete|remove|purge|erase|tidy\s+up|throw away|get rid of)\b.*\b(checkpoints?|models?|files?|everything|all)\b`,

	// Dataset work: "import the pairs", "reclassify the dataset".
	`\b(dataset|reclassify|retag|pairs|yaml|protobuf|metrics|labels|domain\s+tags?)\b`,
	`\bimport\b.*\b(examples|pairs)\b`,
	`\b(examples|pairs)\b.*\bimport\b`,

	// Make targets: "list the commands", "what commands exist".
	`\bmake\b.*\b(targets?|commands?)\b`,
	`\b(targets?|commands?)\b.*\bavailable\b`,
	`\bavailable\b.*\b(commands?|targets?)\b`,
	`\b(list|show|display)\b.*\b(commands|targets)\b`,
	`\bcommands?\b.*\bexist\b`,
	`\bwith make\b`,
	`\brun\b.*\bwith make\b`,

	// Git hooks: "install the hooks", "hook up git".
	`\b(install|set\s+up|setup|enable)\b.*\bhooks?\b`,
	`\bhooks?\b.*\b(install|set\s+up|setup|enable)\b`,
	`\bhook\s+up\b.*\bgit\b`,
	`\bgit\b.*\bhook\s+up\b`,
	`\bprecommit\b`,

	// Talking to Gollemer itself: "talk to gollemer", "start a chat session".
	// NOTE: \bgollemer\b stays case-sensitive (lowercase only). A capitalized
	// "Gollemer" is someone greeting the bot by name ("Hi Gollemer!"), which
	// is social; lowercase "gollemer" means the command/system.
	`(?-i:\bgollemer\b)`,
	`\b(chat|talk|conversation)\b.*\b(model|session|interface)\b`,
	`\b(model|session|interface)\b.*\b(chat|talk|conversation)\b`,
	`\b(launch|open|start)\b.*\bchat\b`,

	// Fresh start: "start over", "begin again".
	`\bfresh\s+start\b`,
	`\bstart\b.*\bfresh\b`,
	`\bstart\s+over\b`,
	`\bbegin\s+(again|anew)\b`,

	// Start-here guide: "show the start here guide".
	`\bstart\b.*\b(guide|here)\b`,
	`\b(guide|here)\b.*\bstart\b`,

	// Debug chat: "show the thought process", "chat with debug prints".
	`\bthought\b.*\bprocess\b`,
	`\bdebug\b.*\b(mode|prints)\b`,

	// Improve/upgrade: "improve the models", "run the full upgrade".
	// NOTE: bare \bupgrade\b is NOT matched: "upgrade a Go module
	// dependency" is a go question and "upgrade the uuid dependency"
	// is gocli. Only the model-upgrade senses count.
	`\b(improve|upgrade)\b.*\bmodels?\b`,
	`\b(full|complete)\b.*\bupgrade\b`,

	// Explain project: "explain this project", "give me a project overview".
	`\bexplain\b.*\bproject\b`,
	`\bproject\b.*\b(overview|explain)\b`,

	// Command picker: "let me choose a command", "open the command picker".
	`\b(choose|pick|select)\b.*\bcommands?\b`,
	`\bcommands?\b.*\bpicker\b`,
	`\bpicker\b`,

	// Eval: "score all the models", "run the eval suite".
	`\beval\b.*\bsuite\b`,
	`\bscore\b.*\bmodels?\b`,
	`\bhow good\b.*\bmodels?\b`,

	// Update a model: "update the gocli model".
	`\bupdate\b.*\b(gocli|gocode|model)\b`,

	// Help: "show the help".
	`\bshow\b.*\bhelp\b`,
}

var makefileIntent = regexp.MustCompile(`(?i:` + strings.Join(makefileIntentPatterns, "|") + `)`)
