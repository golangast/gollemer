package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// brains.go — THE COORDINATOR. One registry of every brain in the cast:
// what its job is, when it works, and when it fails. The coordinator routes
// work to brains (via the intent layer), RECORDS every dispatch and its
// outcome, and can answer for itself: "brains" lists the cast, "brain
// status" reports who has been carrying the work and who has been
// failing, and "why did you do that" explains the last routing decision
// from the recorded evidence — not from a script.

type BrainInfo struct {
	Key   string
	Name  string
	Job   string
	Works string
	Fails string
}

var brainRegistry = []BrainInfo{
	{"coordinator", "The Coordinator", "routes your words to the right brain and keeps score",
		"always on; owns the routing table and the outcome log",
		"it knows nothing itself — if no brain fits, it must say so"},
	{"friend", "The Friend", "watches how YOU work across all the brains and speaks up rarely, when it spots a pattern worth naming",
		"patterns over many turns and sessions; its notebook is a local file (~/.gollemer_friend.json), yours alone",
		"it is quiet by design — long cooldowns, never twice in a row; for advice on demand, ask \"any suggestions\""},
	{"analyst", "The Analyst", "reads a project's structure: tours, examples, what-is, walkthroughs, where a value's data comes from and where it changes, the big picture",
		"any Go project it can parse; every claim is computed, so unseen code is fine",
		"structure only — it cannot judge taste, quality, or intent"},
	{"why", "The Reasoner", "explains WHY a function is shaped the way it is",
		"reasons the code itself shows: locks, errors, loops, callers, the author's comment",
		"intent the author never wrote down — it says so instead of guessing"},
	{"comparer", "The Comparer", "puts two functions side by side and names the real difference",
		"both functions in the analyzed project",
		"feelings about which is 'better' — it reports facts, you judge"},
	{"lexicon", "The Lexicon", "Go words in plain English (goroutine, channel, slice…)",
		"the standard vocabulary a beginner meets first",
		"project-specific jargon it has never seen defined"},
	{"teacher", "The Teacher", "orders a project's concepts into a path a beginner can follow",
		"sequences by real dependency order in the code",
		"knowing how YOU learn best — it follows the code's order, not yours"},
	{"runner", "The Runner", "actually runs the project — make target, entry point, or tests — and narrates",
		"projects that build; capped at 60 seconds, always asks first",
		"interactive programs, slow builds, anything needing your input"},
	{"watcher", "The Watcher", "watches a folder while you code and diffs the structure on every save",
		"opt-in per folder (gollemer -watch <folder>); reports only meaningful changes — signatures, behaviors, callers at risk — and leaves them for the Friend",
		"structure, not intent; and it sees nothing while it is not running"},
	{"detective", "The Detective", "given a GitHub issue, finds where the edit belongs",
		"localization: it has pointed at the exact files maintainers changed",
		"writing the fix itself — its drafts are plain skeletons"},
	{"composer", "The Composer", "assembles code from verified pieces: APIs, tests, getters, comments",
		"shapes it has a schema for; go build checks the result",
		"novel algorithms from a description — it has no schema for those"},
	{"critic", "The Critic", "reviews a function or file for smells: dead code, ignored errors, stubs",
		"anomalies the structure exposes",
		"deep logic bugs that need running the code to see"},
	{"storyteller", "The Storyteller (trained model)", "phrases the Analyst's facts in its own words",
		"nothing yet, unsupervised — the Verifier gates every sentence",
		"measured: v1 memorized its lessons and invented facts on unseen code (rejected 3/3); v2's verdict is still out"},
	{"verifier", "The Verifier", "checks the Storyteller's prose against the facts and rejects lies",
		"caught all 3 of v1's inventions on held-out code",
		"it can only reject — when it does, the Analyst's template speaks instead"},
	{"social", "The Companion (trained model)", "small talk: greetings, feelings, identity",
		"the conversation it rehearsed — it scored 10/10 on its targeted set",
		"open-ended chat beyond its rehearsal; it has no picture of the world"},
	{"maker", "The Maker (trained model)", "turns 'run the tests' into the right make command, then asks permission",
		"129 commands; conservative — it asks before anything runs",
		"commands outside its training set"},
	{"memory", "The Memory", "holds the thread: what you're looking at, what 'tell me more' continues",
		"this session's last 30 turns and last subject",
		"across sessions — it forgets when the chat closes (unless the profile file is on)"},
}

func brainByKey(key string) *BrainInfo {
	for i := range brainRegistry {
		if brainRegistry[i].Key == key {
			return &brainRegistry[i]
		}
	}
	return nil
}

// intentBrain is the coordinator's routing table: intent → brain key.
func intentBrain(intent string) string {
	switch intent {
	case "summarize", "analyze", "tour", "example", "whatis", "show_func",
		"walkthrough", "arch", "why_import", "dataflow_q":
		return "analyst"
	case "why_func":
		return "why"
	case "compare":
		return "comparer"
	case "glossary":
		return "lexicon"
	case "learn_path":
		return "teacher"
	case "run_project":
		return "runner"
	case "where":
		return "detective"
	case "comment", "test_scaffold":
		return "composer"
	case "review":
		return "critic"
	case "uexplain":
		return "storyteller"
	case "suggest":
		return "friend"
	case "tell_more":
		return "memory"
	case "what_now", "help", "brains", "why_brain":
		return "coordinator"
	}
	return ""
}

// --- outcome recording -------------------------------------------------

type brainStat struct{ OK, Fail int }

var (
	brainMu     sync.Mutex
	brainStats  = map[string]*brainStat{}
	lastBrain   string
	lastBrainOK bool
	lastNote    string
)

type brainLogEntry struct {
	T     string `json:"t"`
	Brain string `json:"brain"`
	OK    bool   `json:"ok"`
	Note  string `json:"note,omitempty"`
}

func brainLogPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".gollemer_brains.jsonl")
}

// recordBrain is called on every dispatch the coordinator makes: which
// brain, whether it delivered, and a short note (intent, fallback, gate
// verdict). Persisted to ~/.gollemer_brains.jsonl (0600, local only).
func recordBrain(key string, ok bool, note string) {
	if key == "" {
		return
	}
	brainMu.Lock()
	st := brainStats[key]
	if st == nil {
		st = &brainStat{}
		brainStats[key] = st
	}
	if ok {
		st.OK++
	} else {
		st.Fail++
	}
	lastBrain, lastBrainOK, lastNote = key, ok, note
	brainMu.Unlock()
	friendObserveBrain(key, ok)

	if path := brainLogPath(); path != "" {
		line, err := json.Marshal(brainLogEntry{
			T: time.Now().Format(time.RFC3339), Brain: key, OK: ok, Note: note,
		})
		if err == nil {
			if f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600); err == nil {
				f.Write(append(line, '\n'))
				f.Close()
			}
		}
	}
}

func brainLogCounts() map[string]*brainStat {
	out := map[string]*brainStat{}
	data, err := os.ReadFile(brainLogPath())
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var e brainLogEntry
		if json.Unmarshal([]byte(line), &e) != nil {
			continue
		}
		st := out[e.Brain]
		if st == nil {
			st = &brainStat{}
			out[e.Brain] = st
		}
		if e.OK {
			st.OK++
		} else {
			st.Fail++
		}
	}
	return out
}

// --- the coordinator answers for itself --------------------------------

func printBrains() {
	fmt.Println("The cast — every brain, its job, and its honest limits:")
	for _, b := range brainRegistry {
		fmt.Printf("  %-14s %s\n", b.Name, b.Job)
		fmt.Printf("  %-14s   works: %s\n", "", b.Works)
		fmt.Printf("  %-14s   fails: %s\n", "", b.Fails)
	}
	fmt.Println("Ask \"how are your brains doing\" for the score so far, or \"why did you do that\" after any answer.")
}

func printBrainStatus() {
	fmt.Println("Brain scoreboard:")
	allTime := brainLogCounts()
	var idle []string
	for _, b := range brainRegistry {
		brainMu.Lock()
		sess := brainStats[b.Key]
		brainMu.Unlock()
		all := allTime[b.Key]
		sOK, sF, aOK, aF := 0, 0, 0, 0
		if sess != nil {
			sOK, sF = sess.OK, sess.Fail
		}
		if all != nil {
			aOK, aF = all.OK, all.Fail
		}
		if sOK+sF+aOK+aF == 0 {
			idle = append(idle, b.Name)
			continue
		}
		fmt.Printf("  %-14s this session: %d ok, %d failed · all time: %d ok, %d failed\n",
			b.Name, sOK, sF, aOK, aF)
	}
	if len(idle) > 0 {
		fmt.Printf("\n%d other brain(s) haven't been needed yet: %s.\n", len(idle), strings.Join(idle, ", "))
	}
	if st := allTime["storyteller"]; st != nil && st.Fail > 0 {
		fmt.Printf("\nNote: the Storyteller has failed the Verifier %d time(s) all time — when it does, the Analyst's verified wording speaks instead. That is the system working as designed.\n", st.Fail)
	}
}

func explainLastBrain() {
	brainMu.Lock()
	key, ok, note := lastBrain, lastBrainOK, lastNote
	brainMu.Unlock()
	if key == "" {
		fmt.Println("No brain has answered yet this session — ask me something about a project and then ask again.")
		return
	}
	b := brainByKey(key)
	name := key
	job := ""
	if b != nil {
		name = b.Name
		job = " (" + b.Job + ")"
	}
	verdict := "and it delivered"
	if !ok {
		verdict = "and it FAILED — a fallback answered instead"
	}
	fmt.Printf("That came from %s%s. I routed it there %s.", name, job, verdict)
	if note != "" {
		fmt.Printf(" Recorded note: %s.", note)
	}
	if b != nil {
		fmt.Printf("\nIts known limit: %s.", b.Fails)
	}
	fmt.Println()
}
