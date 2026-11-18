package chat

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// bookresources.go — "I need resources for <topic>", answered
// from the Go knowledge book the chat carries.
//
// Two resource claims live side by side and own different
// shapes. resources.go owns the line-initial asks ("resources
// for databases") and the build-goal statements, and answers
// with link packs from the user's own Advocates doc. This
// claim owns the shapes that file's anchored patterns cannot
// see — a resources ask wearing a first-person or "any"
// lead-in ("i need resources for a database", "i want
// resources for testing", "any resources for channels") — and
// answers from the curated Go book (goknowledge.go): the entry
// itself, plus one line pointing at the piece the chat can
// build for the topic when the capability catalog has one.
//
// The lead-in shapes mattered the hard way (2026-11-09): "i
// need resources for a database" matched no resources shape,
// fell to the intent classifier, and was answered with a
// varflow report about a local variable named `need`. The
// claim is seated directly beside tryBuildResources in every
// loop — deterministic, ahead of the intent layer — and its
// shapes demand the word "resources", so a build ask that
// merely needs something ("i need a database in jim") declines
// and keeps the routing it has today.
//
// "Resources" here means the book's own material. The chat
// makes no external calls and invents no reading lists: a
// topic the book doesn't cover gets an honest miss, with the
// closest book topics named when the book has near ones.

// bookResourcesRes are the claim's shapes: an explicit
// resources ask behind a lead-in the Advocates claim's
// line-initial patterns miss. The capture is the topic,
// cleaned by cleanGoal (resources.go) the same way a
// build goal is.
var bookResourcesRes = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^\s*(?:please[,\s]+)?i (?:need|want) resources? (?:for|on|about)\s+(.+?)\s*$`),
	regexp.MustCompile(`(?i)^\s*(?:please[,\s]+)?i'?m looking for resources? (?:for|on|about)\s+(.+?)\s*$`),
	regexp.MustCompile(`(?i)^\s*(?:please[,\s]+)?(?:are there\s+|do you have\s+)?any resources? (?:for|on|about)\s+(.+?)\s*$`),
}

// parseBookResourceTopic extracts the topic from a
// book-resources line, or "" when the line is not one of the
// claim's shapes. The guards mirror parseBuildGoal's: a repo
// or URL keeps its own route, a path is not a topic, a single
// dotted token is a symbol reference, and a whole sentence is
// not a topic either.
func parseBookResourceTopic(line string) string {
	if strings.Contains(line, "://") || strings.Contains(line, "github.com/") {
		return ""
	}
	var raw string
	for _, re := range bookResourcesRes {
		if m := re.FindStringSubmatch(line); m != nil {
			raw = m[1]
			break
		}
	}
	if raw == "" {
		return ""
	}
	// A trailing "please" is wrapper, not topic.
	raw = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(raw), "please"))
	topic := cleanGoal(raw)
	if topic == "" || strings.Contains(topic, "/") {
		return ""
	}
	if !strings.Contains(topic, " ") && strings.Contains(topic, ".") {
		return ""
	}
	if len(strings.Fields(topic)) > 6 {
		return ""
	}
	return topic
}

// nearBookTopics names the book entries nearest a topic the
// strong lookup declined: entries sharing at least one content
// word (stemmed) with the topic, best overlap first, book
// order breaking ties, capped at limit. Sharing nothing earns
// nothing — a bare miss is never dressed up with unrelated
// topics and called a suggestion.
func nearBookTopics(topic string, limit int) []string {
	want := map[string]bool{}
	for _, t := range knowledgeTokens(topic) {
		want[stem(t)] = true
	}
	if len(want) == 0 {
		return nil
	}
	type cand struct {
		title string
		score int
	}
	var cands []cand
	seen := map[string]bool{}
	for _, e := range goKnowledge {
		words := map[string]bool{}
		add := func(text string) {
			for _, tw := range knowledgeTokens(text) {
				for _, part := range strings.Split(tw, ".") {
					if part != "" {
						words[stem(part)] = true
					}
				}
			}
		}
		add(e.title)
		for _, kw := range e.keywords {
			add(kw)
		}
		score := 0
		for w := range want {
			if words[w] {
				score++
			}
		}
		if score > 0 && !seen[e.title] {
			seen[e.title] = true
			cands = append(cands, cand{e.title, score})
		}
	}
	sort.SliceStable(cands, func(a, b int) bool { return cands[a].score > cands[b].score })
	var out []string
	for _, c := range cands {
		if len(out) >= limit {
			break
		}
		out = append(out, c.title)
	}
	return out
}

// buildPointerForTopic finds the capability the chat can build
// for a topic: the catalog entry whose text names the topic's
// principal word — its longest content word — earliest. The
// pointer is earned by the catalog's own words (the same data
// the capability seat answers from, drift-tested against the
// claims), so it can only ever point at a piece that exists;
// a topic no entry is about gets no pointer, never a vapor
// offer.
func buildPointerForTopic(topic string) (capabilityEntry, bool) {
	principal := ""
	for _, t := range knowledgeTokens(topic) {
		if len(t) > len(principal) {
			principal = t
		}
	}
	if len(principal) < 4 {
		return capabilityEntry{}, false
	}
	want := stem(principal)
	bestPos := -1
	var best capabilityEntry
	for _, g := range capabilityCatalog {
		for _, e := range g.entries {
			for i, w := range knowledgeTokens(e.text) {
				if stem(w) == want {
					if bestPos < 0 || i < bestPos {
						bestPos, best = i, e
					}
					break
				}
			}
		}
	}
	if bestPos < 0 {
		return capabilityEntry{}, false
	}
	return best, true
}

// lowerFirst drops a leading capital for mid-sentence use.
func lowerFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToLower(s[:1]) + s[1:]
}

// tryBookResources claims a need/want/any resources ask and
// answers it from the Go book. See the file header for the
// ownership split with tryBuildResources.
func tryBookResources(line string) bool {
	topic := parseBookResourceTopic(line)
	if topic == "" {
		return false
	}
	entry, ok := lookupGoKnowledgeEntry(topic)
	var b strings.Builder
	if ok {
		fmt.Fprintf(&b, "Here's what my Go book says about %s:\n\n%s\n", entry.title, entry.body)
		if ptr, has := buildPointerForTopic(topic); has {
			fmt.Fprintf(&b, "\nWant to build with it here? Say %q — %s.\n", ptr.example, lowerFirst(ptr.text))
		}
	} else {
		fmt.Fprintf(&b, "I don't have resources on %q in the Go book I carry — and I don't fetch anything from the web, so I won't invent a list of sites.\n", topic)
		if near := nearBookTopics(topic, 3); len(near) > 0 {
			quoted := make([]string, len(near))
			for i, t := range near {
				quoted[i] = fmt.Sprintf("%q", t)
			}
			fmt.Fprintf(&b, "Closest topics the book does cover: %s. Ask me about one and I'll show you what it says.\n", strings.Join(quoted, ", "))
		}
	}
	printPaced(b.String())
	recordBrain("analyst", true, "book resources: "+topic)
	return true
}
