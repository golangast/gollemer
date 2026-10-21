package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// watch.go — THE WATCHER. An opt-in mode (gollemer -watch <folder>)
// that sits beside the editor: on every save it re-analyzes the project
// and diffs the STRUCTURE against the last snapshot. Only meaningful
// changes become events — a signature change with callers at stake, a
// function that started locking, a new exported function no test
// touches. Events print in this terminal and are appended to
// ~/.gollemer_watch.jsonl (0600, local), where the Friend reads them
// and decides what is worth mentioning in conversation.

type watchLogLine struct {
	T      string `json:"t"`
	Kind   string `json:"kind"`
	Func   string `json:"func"`
	Detail string `json:"detail"`
	Root   string `json:"root"`
}

func WatchLogPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".gollemer_watch.jsonl")
}

func watchLogAppend(e analyze.WatchEvent, root string) {
	path := WatchLogPath()
	if path == "" {
		return
	}
	line, err := json.Marshal(watchLogLine{
		T: time.Now().Format(time.RFC3339), Kind: e.Kind, Func: e.Func,
		Detail: e.Detail, Root: root,
	})
	if err != nil {
		return
	}
	if f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600); err == nil {
		f.Write(append(line, '\n'))
		f.Close()
	}
}

func readWatchLog() []watchLogLine {
	data, err := os.ReadFile(WatchLogPath())
	if err != nil {
		return nil
	}
	var out []watchLogLine
	for _, line := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var e watchLogLine
		if json.Unmarshal([]byte(line), &e) == nil {
			out = append(out, e)
		}
	}
	return out
}

func scanGoMtimes(root string) map[string]int64 {
	out := map[string]int64{}
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", "vendor", "node_modules":
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".go") {
			if info, err := d.Info(); err == nil {
				out[path] = info.ModTime().UnixNano()
			}
		}
		return nil
	})
	return out
}

func sameMtimes(a, b map[string]int64) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

// RunWatch is the -watch mode: poll, debounce, diff, report.
func RunWatch(dir string) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		fmt.Println("watch: bad folder:", err)
		return
	}
	proj, err := analyze.Analyze(abs)
	if err != nil {
		fmt.Println("watch: cannot analyze", abs, "—", err)
		return
	}
	fmt.Printf("The Watcher is on %s — %d functions. Meaningful changes only; Ctrl-C to stop.\n",
		abs, len(proj.AllFuncs()))
	fmt.Println("Events are also kept in ~/.gollemer_watch.jsonl for the Friend.")

	snap := scanGoMtimes(abs)
	var dirtySince time.Time
	saves, eventCount := 0, 0
	churn := map[string]int{}
	churnSaid := map[string]bool{}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt)
	tick := time.NewTicker(2 * time.Second)
	defer tick.Stop()

	for {
		select {
		case <-stop:
			fmt.Printf("\nWatcher off. %d save(s), %d meaningful event(s) this session.\n", saves, eventCount)
			return
		case now := <-tick.C:
			cur := scanGoMtimes(abs)
			if !sameMtimes(snap, cur) {
				snap = cur
				if dirtySince.IsZero() {
					dirtySince = now
				}
				continue
			}
			if dirtySince.IsZero() || now.Sub(dirtySince) < 2*time.Second {
				continue
			}
			dirtySince = time.Time{}
			saves++
			next, err := analyze.Analyze(abs)
			if err != nil {
				fmt.Println("  (that save doesn't parse yet — still watching)")
				continue
			}
			for _, e := range analyze.DiffProjects(proj, next) {
				fmt.Printf("  %s  %s\n", now.Format("15:04:05"), e.String())
				watchLogAppend(e, abs)
				eventCount++
				if e.Kind == "sig-changed" || e.Kind == "behavior-changed" {
					churn[e.Func]++
					if churn[e.Func] >= 4 && !churnSaid[e.Func] {
						churnSaid[e.Func] = true
						ce := analyze.WatchEvent{Kind: "churn", Func: e.Func,
							Detail: "edited 4+ times this session — a spot worth a second look"}
						fmt.Printf("  %s  %s\n", now.Format("15:04:05"), ce.String())
						watchLogAppend(ce, abs)
						eventCount++
					}
				}
			}
			proj = next
		}
	}
}
