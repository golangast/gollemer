package chat

import (
	"strings"
	"testing"
)

// TestIntentDispatchVetoLeavesBookHowTos pins the how-to veto
// in dispatchIntent (measured 2026-11-07, cookbook expansion):
// a how-to question the knowledge book answers must fall
// through to the deterministic seat that owns it, never be
// dispatched as a beginner-guidance intent — "how do I serve
// static files" classified as what_now and got the what-now
// guidance while the book's answer sat behind the layer. A
// how-to-shaped directions ask the book does NOT answer
// still dispatches, and a non-guidance intent is untouched
// by the veto.
func TestIntentDispatchVetoLeavesBookHowTos(t *testing.T) {
	if dispatchIntent("what_now", "how do I serve static files", nil) {
		t.Error("dispatch claimed a book how-to as what_now, want the veto to fall through")
	}
	if dispatchIntent("example", "how do I write an example test for this function", nil) {
		t.Error("dispatch claimed a book how-to as example, want the veto to fall through")
	}

	var handled bool
	out := captureStdout(t, func() { handled = dispatchIntent("what_now", "how do I get started", nil) })
	if !handled || !strings.Contains(out, "tour") {
		t.Errorf("genuine directions ask changed: handled=%v out=%q", handled, out)
	}
}
