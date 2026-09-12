package makefile

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// MakeTarget represents a parsed make target and its metadata.
type MakeTarget struct {
	Name        string
	Description string
	Command     string
}

// RankedCommand represents a ranked makefile command match.
type RankedCommand struct {
	Name        string
	Description string
	Score       float32
}

// ParseOptions controls Makefile parsing behavior.
type ParseOptions struct {
	MakefilePath string
}

// ParseMakefile reads a Makefile and extracts targets with their descriptions and commands.
func ParseMakefile(opts ParseOptions) ([]MakeTarget, error) {
	if opts.MakefilePath == "" {
		return nil, fmt.Errorf("makefile path is required")
	}
	f, err := os.Open(opts.MakefilePath)
	if err != nil {
		return nil, fmt.Errorf("open makefile: %w", err)
	}
	defer f.Close()

	var targets []*MakeTarget
	var pendingDescription string
	var current *MakeTarget
	var inTarget bool

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)

		if trimmed == "" || strings.HasPrefix(trimmed, "#") && !strings.HasPrefix(trimmed, "##") {
			continue
		}

		if strings.HasPrefix(trimmed, "##") {
			desc := strings.TrimSpace(strings.TrimPrefix(trimmed, "##"))
			desc = strings.TrimSuffix(desc, ":")
			desc = strings.TrimSpace(desc)
			if desc != "" {
				pendingDescription = desc
			}
			continue
		}

		if strings.Contains(trimmed, ":") && !strings.HasPrefix(trimmed, "\t") && !strings.HasPrefix(trimmed, "-") {
			parts := strings.SplitN(trimmed, ":", 2)
			name := strings.TrimSpace(parts[0])

			if strings.Contains(name, " ") || strings.Contains(name, "=") || strings.HasPrefix(name, ".") {
				continue
			}

			current = &MakeTarget{Name: name}
			if pendingDescription != "" {
				current.Description = pendingDescription
				pendingDescription = ""
			} else if len(parts) > 1 {
				rest := strings.TrimSpace(parts[1])
				if rest != "" && !strings.HasPrefix(rest, "=") {
					current.Description = rest
				}
			}
			if current.Description != "" {
				current.Description = strings.TrimPrefix(current.Description, name)
				current.Description = strings.TrimSpace(current.Description)
				current.Description = strings.TrimPrefix(current.Description, ":")
				current.Description = strings.TrimSpace(current.Description)
				current.Description = strings.TrimSuffix(current.Description, ":")
				current.Description = strings.TrimSpace(current.Description)
				current.Description = normalizeSpaces(current.Description)
			}
			targets = append(targets, current)
			inTarget = true
			continue
		}

		if inTarget && current != nil {
			cmd := strings.TrimSpace(line)
			if cmd != "" && !strings.HasPrefix(cmd, "#") {
				if current.Command == "" {
					current.Command = cmd
				} else {
					current.Command += "\n" + cmd
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan makefile: %w", err)
	}

	return derefPointers(targets), nil
}

func derefPointers(ptrs []*MakeTarget) []MakeTarget {
	out := make([]MakeTarget, len(ptrs))
	for i, p := range ptrs {
		out[i] = *p
	}
	return out
}

// SaveTrainingData writes the parsed make targets as a YAML file compatible with the
// existing YAML training pipeline. The generated conversations focus on command
// names and simple social descriptions only; raw bash command bodies are omitted.
func SaveTrainingData(targets []MakeTarget, outPath string) error {
	if len(targets) == 0 {
		return fmt.Errorf("no make targets to save")
	}

	simpleDescriptions := map[string]string{
		"install-hooks":       "install hooks",
		"train":               "train the model",
		"train-resume":        "resume training",
		"train-fresh":         "fresh training",
		"train-small":         "small training",
		"train-small-seq2seq": "small seq2seq training",
		"test-small-seq2seq":  "test seq2seq",
		"seq2seq-prompt":      "send a prompt",
		"seq2seq-chat":        "chat with seq2seq",
		"chat":                "start chatting",
		"metrics":             "run metrics",
		"export-labels":       "export labels",
		"clean":               "clean models",
		"clean-all":           "clean everything",
		"conversing-pb":       "convert conversing data",
		"social-replies-pb":   "convert social replies",
		"tech-multiturn-pb":   "convert tech multiturn",
		"all-pb":              "convert all data",
		"makefile-pb":         "convert makefile data",
		"sel":                 "select a target",
		"help":                "show help",
	}

	responseTemplates := []string{
		"you can use make %s for this",
		"use make %s",
		"try make %s",
		"run make %s to do this",
		"make %s will help with that",
		"you should run make %s",
		"the command is make %s",
	}

	var sb strings.Builder
	sb.WriteString("conversations:\n")
	for _, t := range targets {
		desc := t.Description
		if desc == "" {
			desc = t.Name
		}
		if simpleDesc, ok := simpleDescriptions[t.Name]; ok {
			desc = simpleDesc
		}
		desc = normalizeSpaces(strings.ToLower(desc))

		for qIdx, q := range []string{
			fmt.Sprintf("how do i %s", desc),
			fmt.Sprintf("what is the command for %s", desc),
			fmt.Sprintf("i want to %s", desc),
		} {
			template := responseTemplates[qIdx]
			a := normalizeSpaces(fmt.Sprintf(template, t.Name))

			sb.WriteString(fmt.Sprintf("  - conversation_id: \"make_%s_q%d\"\n", t.Name, qIdx+1))
			sb.WriteString("    turns:\n")
			sb.WriteString("      - turn_sequence: 1\n")
			sb.WriteString("        role: \"user\"\n")
			sb.WriteString(fmt.Sprintf("        content: \"%s\"\n", escapeYAMLString(q)))
			sb.WriteString("      - turn_sequence: 2\n")
			sb.WriteString("        role: \"assistant\"\n")
			sb.WriteString(fmt.Sprintf("        content: \"%s\"\n", escapeYAMLString(a)))
		}
	}

	if err := os.WriteFile(outPath, []byte(sb.String()), 0644); err != nil {
		return fmt.Errorf("write yaml: %w", err)
	}
	return nil
}

func escapeYAMLString(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	s = strings.ReplaceAll(s, "\n", `\n`)
	return s
}

func normalizeSpaces(s string) string {
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	return strings.TrimSpace(s)
}

// DefaultMakefilePaths returns common Makefile paths relative to a project root.
func DefaultMakefilePaths(projectRoot string) []string {
	return []string{
		filepath.Join(projectRoot, "Makefile"),
		filepath.Join(projectRoot, "makefile"),
		filepath.Join(projectRoot, "GNUmakefile"),
		filepath.Join(projectRoot, "makefile.mak"),
	}
}

// FindMakefile searches for a Makefile in the given project root.
func FindMakefile(projectRoot string) (string, error) {
	for _, p := range DefaultMakefilePaths(projectRoot) {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("no Makefile found in %s", projectRoot)
}

// Tokenize splits text into lowercase word tokens.
func Tokenize(text string) []string {
	text = strings.ToLower(text)
	fields := strings.Fields(text)
	out := make([]string, 0, len(fields))
	for _, w := range fields {
		w = strings.Trim(w, ".,!?;:\"'`()[]{}")
		if w != "" {
			out = append(out, w)
		}
	}
	return out
}

// stopWords are common English words excluded from query matching to reduce noise.
var stopWords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "it": true,
	"do": true, "to": true, "for": true, "and": true, "or": true,
	"in": true, "at": true, "by": true, "on": true, "of": true,
	"with": true, "all": true, "this": true, "that": true, "from": true,
	"run": true, "start": true, "use": true, "make": true,
}

// splitName breaks a target name like "train-small-seq2seq" into component words.
func splitName(name string) []string {
	parts := strings.FieldsFunc(name, func(r rune) bool { return r == '-' || r == '_' })
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p != "" {
			out = append(out, strings.ToLower(p))
		}
	}
	return out
}

// filterStopWords removes stop words from a token list.
func filterStopWords(tokens []string) []string {
	out := make([]string, 0, len(tokens))
	for _, t := range tokens {
		if !stopWords[t] {
			out = append(out, t)
		}
	}
	return out
}

// TopKMakefileCommands returns the top K matching makefile targets for a query.
// Scores are normalized to [0,1] against the best match.
func TopKMakefileCommands(targets []MakeTarget, query string, k int) []RankedCommand {
	if k <= 0 {
		k = 3
	}
	if len(targets) == 0 || strings.TrimSpace(query) == "" {
		return nil
	}

	rawTokens := Tokenize(query)
	qTokens := filterStopWords(rawTokens)
	if len(qTokens) == 0 {
		// If all words were stop words, fall back to raw tokens
		qTokens = rawTokens
	}
	if len(qTokens) == 0 {
		return nil
	}

	qSet := make(map[string]bool, len(qTokens))
	for _, t := range qTokens {
		qSet[t] = true
	}

	type scored struct {
		target MakeTarget
		score  float64
	}
	scoredList := make([]scored, 0, len(targets))

	for _, t := range targets {
		// Name words (split on - and _) get 3x weight
		nameWords := splitName(t.Name)
		nameSet := make(map[string]bool, len(nameWords))
		for _, w := range nameWords {
			nameSet[w] = true
		}

		// Description + command tokens get 1x weight
		descDoc := strings.ToLower(t.Description + " " + t.Command)
		descTokens := filterStopWords(Tokenize(descDoc))
		descSet := make(map[string]bool, len(descTokens))
		for _, w := range descTokens {
			descSet[w] = true
		}

		// Count weighted hits: how many query tokens match name vs description
		var weightedHits float64
		for _, qt := range qTokens {
			if nameSet[qt] {
				weightedHits += 3.0 // name match is worth 3x
			} else if descSet[qt] {
				weightedHits += 1.0
			}
		}

		// Also reward when the name words are well covered by the query
		// (prevents "train" from matching "makefile-train" as well as "train")
		nameHits := 0
		for _, nw := range nameWords {
			if qSet[nw] {
				nameHits++
			}
		}
		nameCoverage := float64(0)
		if len(nameWords) > 0 {
			nameCoverage = float64(nameHits) / float64(len(nameWords))
		}

		// Final score: weighted hits normalised by query length, boosted by name coverage
		queryLen := float64(len(qTokens))
		score := (weightedHits/queryLen)*0.7 + nameCoverage*0.3

		scoredList = append(scoredList, scored{target: t, score: score})
	}

	sort.SliceStable(scoredList, func(i, j int) bool {
		return scoredList[i].score > scoredList[j].score
	})

	if len(scoredList) > k {
		scoredList = scoredList[:k]
	}

	best := float64(0)
	for _, s := range scoredList {
		if s.score > best {
			best = s.score
		}
	}

	out := make([]RankedCommand, 0, len(scoredList))
	for _, s := range scoredList {
		pct := float32(0)
		if best > 0 {
			pct = float32(s.score / best)
		}
		out = append(out, RankedCommand{
			Name:        s.target.Name,
			Description: s.target.Description,
			Score:       pct,
		})
	}
	return out
}
