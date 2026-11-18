package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// friend.go — THE FRIEND. A personal brain that watches how the USER
// works across all the other brains: it reads the Coordinator's outcome
// records, remembers topics and habits across sessions (a local file,
// the user's own), and speaks up RARELY — a trigger pattern (stuck,
// repeating, circling) plus a long cooldown, never twice in a row.
// Silence is its default. Asked directly ("any suggestions"), it reports
// from its notebook.

// friendMiss is one learn-mode check question the user got wrong:
// the concept it was about, the question as asked, the correct fact,
// and where and when it happened — so a later feature can bring the
// missed ones back.
type friendMiss struct {
	Concept  string `json:"concept"`
	Question string `json:"question"`
	Fact     string `json:"fact"`
	Project  string `json:"project"` // project root the question was about
	Day      string `json:"day"`     // 2006-01-02
}

type friendMemory struct {
	FuncAsks   map[string]int    `json:"func_asks"`
	IntentSeen map[string]int    `json:"intent_seen"`
	Sessions   int               `json:"sessions"`
	LastTopic  string            `json:"last_topic"`
	LastDay    string            `json:"last_day"`
	Spoken     map[string]string `json:"spoken"`               // trigger key -> day it last spoke
	WatchRead  string            `json:"watch_read,omitempty"` // RFC3339 cursor into the Watcher log
	Guided     map[string]int    `json:"guided,omitempty"`     // project root -> next guided step
	Taught     int               `json:"taught,omitempty"`     // training-tour lessons completed
	Misses     []friendMiss      `json:"misses,omitempty"`     // learn-mode questions missed
}

var (
	friendMem           *friendMemory
	friendTurn          int
	friendLastSpokeTurn = -1000
	friendPending       string
	friendPendingKey    string
	friendFiredSession  = map[string]bool{}
	friendSessFuncs     = map[string]int{}
	friendSessIntents   = map[string]int{}
	friendFailStreak    = map[string]int{}
	friendSessionBegun  bool
)

// friendCooldownTurns is the restraint: at least this many handled
// turns between spoken suggestions.
const friendCooldownTurns = 10

func friendPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".gollemer_friend.json")
}

func friendLoad() *friendMemory {
	if friendMem != nil {
		return friendMem
	}
	m := &friendMemory{
		FuncAsks:   map[string]int{},
		IntentSeen: map[string]int{},
		Spoken:     map[string]string{},
	}
	if data, err := os.ReadFile(friendPath()); err == nil {
		_ = json.Unmarshal(data, m)
		if m.FuncAsks == nil {
			m.FuncAsks = map[string]int{}
		}
		if m.IntentSeen == nil {
			m.IntentSeen = map[string]int{}
		}
		if m.Spoken == nil {
			m.Spoken = map[string]string{}
		}
		if m.Guided == nil {
			m.Guided = map[string]int{}
		}
	}
	friendMem = m
	return m
}

func friendSave() {
	if friendMem == nil {
		return
	}
	data, err := json.MarshalIndent(friendMem, "", " ")
	if err != nil {
		return
	}
	_ = os.WriteFile(friendPath(), data, 0600)
}

func friendToday() string { return time.Now().Format("2006-01-02") }

// friendWatchNotes composes the Watcher's notable events since the
// read cursor: signature changes, untested new exports, churn, removed
// functions with callers, behavior changes. Plain new functions are
// only noted if nothing more notable happened.
func friendWatchNotes() []string {
	m := friendLoad()
	var notable, plain []string
	for _, e := range readWatchLog() {
		if m.WatchRead != "" && e.T <= m.WatchRead {
			continue
		}
		text := e.Func
		if e.Detail != "" {
			text += " — " + e.Detail
		}
		switch e.Kind {
		case "sig-changed", "untested-new", "churn", "removed-func", "behavior-changed":
			notable = append(notable, text)
		default:
			plain = append(plain, text)
		}
	}
	out := notable
	if len(out) == 0 {
		out = plain
	}
	if len(out) > 3 {
		out = out[:3]
	}
	return out
}

func friendAdvanceWatch() {
	m := friendLoad()
	m.WatchRead = time.Now().Format(time.RFC3339)
	friendSave()
}

// friendBeginSession runs once per process: counts the session and, on
// a new day, queues a welcome-back note about the last topic.
func friendBeginSession() {
	if friendSessionBegun {
		return
	}
	friendSessionBegun = true
	m := friendLoad()
	m.Sessions++
	if notes := friendWatchNotes(); len(notes) > 0 {
		friendSetPending("watch:"+friendToday(),
			"while you were away, the Watcher noticed: "+strings.Join(notes, "; ")+".")
		friendAdvanceWatch()
	}
	if m.LastDay != "" && m.LastDay != friendToday() && m.LastTopic != "" {
		friendSetPending("return", fmt.Sprintf(
			"welcome back — last time you were deep in %s. Say \"what does %s do\" to pick it back up, or start anywhere you like.",
			m.LastTopic, m.LastTopic))
	}
	m.LastDay = friendToday()
	friendSave()
}

// friendSetPending queues a suggestion unless one is already waiting,
// it already fired this session, or (for durable keys) it spoke today.
func friendSetPending(key, text string) {
	if friendPending != "" || friendFiredSession[key] {
		return
	}
	if strings.Contains(key, ":") { // durable key: at most once a day
		if friendLoad().Spoken[key] == friendToday() {
			return
		}
	}
	friendPending, friendPendingKey = text, key
}

// friendMaybeSpeak delivers the queued suggestion if the cooldown has
// passed. Called once per handled turn.
func friendMaybeSpeak() {
	friendBeginSession()
	if friendPending == "" {
		return
	}
	if friendTurn-friendLastSpokeTurn < friendCooldownTurns {
		return
	}
	fmt.Printf("\nThe Friend: %s\n", friendPending)
	friendFiredSession[friendPendingKey] = true
	if strings.Contains(friendPendingKey, ":") {
		friendLoad().Spoken[friendPendingKey] = friendToday()
		friendSave()
	}
	friendPending, friendPendingKey = "", ""
	friendLastSpokeTurn = friendTurn
}

// friendObserveBrain is fed by the Coordinator's recordBrain: it watches
// per-brain failure streaks.
func friendObserveBrain(brain string, ok bool) {
	if ok {
		friendFailStreak[brain] = 0
		return
	}
	friendFailStreak[brain]++
	if friendFailStreak[brain] == 2 {
		name := brain
		if b := brainByKey(brain); b != nil {
			name = b.Name
		}
		advice := "a different phrasing may reach a different brain."
		switch brain {
		case "storyteller":
			advice = "that is the Verifier doing its job — the Analyst's verified wording is already covering for it, and those rejections are on the scoreboard."
		case "why":
			advice = "it needs a function in the open project — \"what does X do\" first will tell you the exact name."
		case "comparer":
			advice = "it needs two exact function names from the open project."
		}
		friendSetPending("stuck:"+brain, fmt.Sprintf(
			"%s has missed twice in a row — %s", name, advice))
	}
}

// friendObserveIntent is fed by the intent layer on every classified
// line: counts the turn, tracks habits, evaluates habit triggers.
func friendObserveIntent(intent string) {
	friendBeginSession()
	friendTurn++
	m := friendLoad()
	m.IntentSeen[intent]++
	friendSessIntents[intent]++

	if friendSessIntents["tell_more"] == 3 {
		friendSetPending("deepdiver",
			"you keep asking for more depth — this project's ideas are actually ordered in a learn path. Say \"what should I learn first\" and take them in order instead of digging at random.")
	}
	if len(friendSessFuncs) >= 5 && friendSessIntents["run_project"] == 0 && friendTurn >= 8 {
		friendSetPending("explorer",
			"you've looked at a lot of this project but haven't seen it move yet. Say \"run it\" — watching it run tends to tie the pieces together.")
	}
	friendMaybeSpeak()
}

// friendNoteFunc records a function the user asked a brain about.
func friendNoteFunc(name string) {
	if name == "" {
		return
	}
	// Canonicalize the casing through the open project ("add" -> "Add").
	if lastAnalyzeProject != nil {
		if fn := findFuncByQuery(lastAnalyzeProject, name); fn != nil {
			name = fn.Name
		}
	}
	friendBeginSession()
	m := friendLoad()
	m.FuncAsks[name]++
	m.LastTopic = name
	friendSessFuncs[name]++
	friendSave()
	if m.FuncAsks[name] >= 3 {
		friendSetPending("topic:"+name, fmt.Sprintf(
			"you've come back to %s %d times now. If it's the thing you keep needing, say \"walk me through %s\" once and get the whole picture — or \"comment %s\" and its doc comment gets drafted, so the next visit is shorter.",
			name, m.FuncAsks[name], name, name))
	}
}

// friendNoteMiss records a learn-mode check question the user got
// wrong, with the correct fact — the flow owns this write (it happens
// as part of answering, never as a side effect of rendering).
func friendNoteMiss(concept, question, fact, project string) {
	m := friendLoad()
	m.Misses = append(m.Misses, friendMiss{
		Concept:  concept,
		Question: question,
		Fact:     fact,
		Project:  project,
		Day:      friendToday(),
	})
	friendSave()
}

// friendReport answers a direct ask from the notebook.
func friendReport() {
	friendBeginSession()
	m := friendLoad()
	var b strings.Builder
	b.WriteString("The Friend's notebook on how you work:\n")
	fmt.Fprintf(&b, "• Sessions together: %d. Last topic: %s.\n", m.Sessions, orNone(m.LastTopic))

	type kv struct {
		k string
		v int
	}
	var asks []kv
	for k, v := range m.FuncAsks {
		asks = append(asks, kv{k, v})
	}
	sort.Slice(asks, func(i, j int) bool { return asks[i].v > asks[j].v })
	if len(asks) > 0 {
		var parts []string
		for i := 0; i < len(asks) && i < 3; i++ {
			parts = append(parts, fmt.Sprintf("%s (%d×)", asks[i].k, asks[i].v))
		}
		fmt.Fprintf(&b, "• Functions you return to: %s.\n", strings.Join(parts, ", "))
	} else {
		b.WriteString("• No repeat topics yet.\n")
	}

	counts := brainLogCounts()
	type bk struct {
		name string
		ok   int
		fail int
	}
	var brains []bk
	for _, info := range brainRegistry {
		if st := counts[info.Key]; st != nil && st.OK+st.Fail > 0 {
			brains = append(brains, bk{info.Name, st.OK, st.Fail})
		}
	}
	sort.Slice(brains, func(i, j int) bool { return brains[i].ok+brains[i].fail > brains[j].ok+brains[j].fail })
	if len(brains) > 0 {
		var parts []string
		for i := 0; i < len(brains) && i < 3; i++ {
			parts = append(parts, fmt.Sprintf("%s %d ok / %d failed", brains[i].name, brains[i].ok, brains[i].fail))
		}
		fmt.Fprintf(&b, "• Brains carrying your work: %s.\n", strings.Join(parts, "; "))
	}

	if notes := friendWatchNotes(); len(notes) > 0 {
		fmt.Fprintf(&b, "• The Watcher noticed since last time: %s.\n", strings.Join(notes, "; "))
		friendAdvanceWatch()
	}

	// A warm suggestion, if one is queued — direct asks bypass cooldown.
	if friendPending != "" {
		fmt.Fprintf(&b, "• One suggestion: %s\n", friendPending)
		friendFiredSession[friendPendingKey] = true
		if strings.Contains(friendPendingKey, ":") {
			m.Spoken[friendPendingKey] = friendToday()
			friendSave()
		}
		friendPending, friendPendingKey = "", ""
	} else {
		b.WriteString("• Nothing stands out right now — no loops, no repeats, nobody stuck. I'll speak up if I spot a pattern.\n")
	}
	friendLastSpokeTurn = friendTurn // being asked resets the clock too
	fmt.Print(b.String())
}

func orNone(s string) string {
	if s == "" {
		return "(none yet)"
	}
	return s
}

// friendResetForTest restores pristine state (tests only).
func friendResetForTest() {
	friendMem = nil
	friendTurn = 0
	friendLastSpokeTurn = -1000
	friendPending, friendPendingKey = "", ""
	friendFiredSession = map[string]bool{}
	friendSessFuncs = map[string]int{}
	friendSessIntents = map[string]int{}
	friendFailStreak = map[string]int{}
	friendSessionBegun = false
}
