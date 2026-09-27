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
// ("how did your day go").
var goTerms = regexp.MustCompile(`(?i:\b(goroutine|closure|defer|struct|interface|slice|channel|module|cgo|generics?|package|func|race condition|compil|architect|waitgroup|pointer|executable|binary|mutex|documentation|profiling|builtin|error handling|builds?|panic|blank identifier|anonymous function)\b)|\bdepende|\bnew\(\)|\bmake\(\)|\bGo\b`)

// ClassifyDomain tags a pair by its content.
func ClassifyDomain(input, output string) string {
	if makefileTerms.MatchString(input) || makefileTerms.MatchString(output) {
		return MakefileDomain
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
			p.Domain = SocialDomain
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
	raw, err := loadTinyPairs(seq2SeqDataPath(projectRoot))
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
	incoming, err := LoadChatDataset(importPath)
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
