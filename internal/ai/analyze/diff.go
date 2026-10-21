package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// diff.go — the WATCHER's eyes. DiffProjects compares two snapshots of a
// project and reports only MEANINGFUL structural changes: new functions,
// removed functions that had callers, signature changes (with the
// caller count at stake), and behavior changes (gained a lock, started
// returning errors). Line edits that change nothing structural are not
// events — the watcher stays quiet about them by design.

type WatchEvent struct {
	Kind   string `json:"kind"` // new-func | removed-func | sig-changed | behavior-changed | untested-new
	Func   string `json:"func"`
	Detail string `json:"detail"`
}

func (e WatchEvent) String() string {
	if e.Detail != "" {
		return fmt.Sprintf("%s: %s", e.Func, e.Detail)
	}
	return e.Func
}

func behaviorWords(set map[string]bool) []string {
	var out []string
	if set[behLocks] {
		out = append(out, "locks")
	}
	if set[behReadLocks] {
		out = append(out, "read-locks")
	}
	if set[behIterates] {
		out = append(out, "iterates")
	}
	if set[behMayFail] {
		out = append(out, "can fail")
	}
	return out
}

func hasTestCaller(p *Project, fn *Func) bool {
	for _, id := range fn.Callers {
		if cf := p.FuncByID(id); cf != nil && cf.IsTest {
			return true
		}
	}
	return false
}

// DiffProjects returns the meaningful changes from old to new, in a
// stable order (removals, signature changes, behavior changes, new
// functions, untested-new last within each group by name).
func DiffProjects(old, new *Project) []WatchEvent {
	var events []WatchEvent
	if old == nil || new == nil {
		return events
	}
	oldByID := map[string]*Func{}
	for _, fn := range old.AllFuncs() {
		oldByID[fn.ID] = fn
	}
	newByID := map[string]*Func{}
	for _, fn := range new.AllFuncs() {
		newByID[fn.ID] = fn
	}

	var removed, sigs, behs, news, untested []WatchEvent
	for id, ofn := range oldByID {
		nfn, ok := newByID[id]
		if !ok {
			if len(ofn.Callers) > 0 {
				removed = append(removed, WatchEvent{Kind: "removed-func", Func: dispName(ofn),
					Detail: fmt.Sprintf("removed — it had %d caller(s)", len(ofn.Callers))})
			}
			continue
		}
		if ofn.Sig != nfn.Sig {
			sigs = append(sigs, WatchEvent{Kind: "sig-changed", Func: dispName(nfn),
				Detail: fmt.Sprintf("signature is now %s — %d caller(s) use it", sigShape(nfn), len(nfn.Callers))})
		}
		ob, nb := behaviorSet(ofn), behaviorSet(nfn)
		var gained, lost []string
		for _, w := range behaviorWords(nb) {
			if !containsWord(behaviorWords(ob), w) {
				gained = append(gained, w)
			}
		}
		for _, w := range behaviorWords(ob) {
			if !containsWord(behaviorWords(nb), w) {
				lost = append(lost, w)
			}
		}
		if len(gained) > 0 || len(lost) > 0 {
			var parts []string
			if len(gained) > 0 {
				parts = append(parts, "now "+strings.Join(gained, ", "))
			}
			if len(lost) > 0 {
				parts = append(parts, "no longer "+strings.Join(lost, ", "))
			}
			behs = append(behs, WatchEvent{Kind: "behavior-changed", Func: dispName(nfn),
				Detail: strings.Join(parts, "; ")})
		}
	}
	for id, nfn := range newByID {
		if _, ok := oldByID[id]; ok {
			continue
		}
		news = append(news, WatchEvent{Kind: "new-func", Func: dispName(nfn),
			Detail: "new function in " + nfn.File})
		if nfn.Exported && !nfn.IsTest && !hasTestCaller(new, nfn) {
			untested = append(untested, WatchEvent{Kind: "untested-new", Func: dispName(nfn),
				Detail: "new exported function — no test exercises it yet"})
		}
	}
	for _, group := range [][]WatchEvent{removed, sigs, behs, news, untested} {
		sort.Slice(group, func(i, j int) bool { return group[i].Func < group[j].Func })
		events = append(events, group...)
	}
	return events
}

// sigShape renders a signature from the function's name onward:
// "func (c *Counter) Add(n int) error" -> "Add(n int) error".
func sigShape(fn *Func) string {
	if i := strings.Index(fn.Sig, fn.Name+"("); i >= 0 {
		return fn.Sig[i:]
	}
	return fn.Sig
}

func containsWord(list []string, w string) bool {
	for _, s := range list {
		if s == w {
			return true
		}
	}
	return false
}
