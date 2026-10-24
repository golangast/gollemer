package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// dottedChatFixture builds the ugorji-shaped project from the
// reported failure: mapEnd on several receivers in package codec
// (long generated-style file names) plus one more in package wire.
func dottedChatFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/dotted\n\ngo 1.21\n")
	write("codec/binc.mono.generated.go", `package codec

type decoderBincBytes struct{ n int }

// mapEnd finishes reading a binc map.
func (d *decoderBincBytes) mapEnd() bool { return d.n > 0 }
`)
	write("codec/msgpack.mono.generated.go", `package codec

type decoderMsgpackIO struct{ n int }

// mapEnd finishes reading a msgpack map.
func (d *decoderMsgpackIO) mapEnd() bool { return d.n > 0 }
`)
	write("codec/json.mono.generated.go", `package codec

type decoderJsonIO struct{ n int }

// mapEnd finishes reading a json map.
func (d *decoderJsonIO) mapEnd() bool { return d.n > 0 }
`)
	write("wire/wire.go", `package wire

type reader struct{ n int }

// mapEnd finishes reading a wire map.
func (r *reader) mapEnd() bool { return r.n > 0 }
`)
	return dir
}

// withPickState isolates the pending pick/clarification globals for
// a test, like withRealIntentModel does for the classifier.
func withPickState(t *testing.T) {
	t.Helper()
	oldPick, oldClarify := pendingPick, pendingClarify
	pendingPick, pendingClarify = nil, nil
	t.Cleanup(func() { pendingPick, pendingClarify = oldPick, oldClarify })
}

// Defect A, bare-line shape: pasting the exact dotted name the
// pick-list suggested — original case or retyped lowercase — must
// resolve to that function. Before the fix this was claimed by the
// intent layer as a find search and answered "No matches for
// "codec.decoderbincbytes.mapend"."
func TestDottedSymbolBareLineResolves(t *testing.T) {
	withPickState(t)
	withEditsProject(t, dottedChatFixture(t))
	for _, line := range []string{
		"codec.decoderBincBytes.mapEnd",
		"codec.decoderbincbytes.mapend",
		"wire.reader.mapEnd",
	} {
		var handled bool
		out := captureStdout(t, func() { handled = tryIntentDispatch(line, nil) })
		if !handled {
			t.Errorf("bare dotted line %q not handled, output:\n%s", line, out)
			continue
		}
		if strings.Contains(out, "No matches for") {
			t.Errorf("bare dotted line %q got the find failure:\n%s", line, out)
		}
		if !strings.Contains(out, "mapEnd") {
			t.Errorf("bare dotted line %q did not answer about mapEnd:\n%s", line, out)
		}
	}
	out := captureStdout(t, func() { tryIntentDispatch("codec.decoderBincBytes.mapEnd", nil) })
	if !strings.Contains(out, "finishes reading a binc map") {
		t.Errorf("expected the binc mapEnd card, got:\n%s", out)
	}
}

// Defect A, question shape: "what does <dotted> do" must reach the
// function's card even though the classifier runs first.
func TestDottedSymbolQuestionResolves(t *testing.T) {
	withPickState(t)
	withEditsProject(t, dottedChatFixture(t))
	for _, line := range []string{
		"what does codec.decoderBincBytes.mapEnd do",
		"what does codec.decoderbincbytes.mapend do",
	} {
		var handled bool
		out := captureStdout(t, func() { handled = tryIntentDispatch(line, nil) })
		if !handled {
			t.Errorf("question %q not handled, output:\n%s", line, out)
			continue
		}
		if !strings.Contains(out, "finishes reading a binc map") {
			t.Errorf("question %q did not resolve to binc mapEnd:\n%s", line, out)
		}
		if strings.Contains(out, "which one did you mean") {
			t.Errorf("question %q fell back to the pick-list:\n%s", line, out)
		}
	}
}

// A dotted token that names nothing must not be claimed: the line
// falls through exactly as before.
func TestDottedSymbolUnknownFallsThrough(t *testing.T) {
	withPickState(t)
	dir := dottedChatFixture(t)
	withEditsProject(t, dir)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze fixture: %v", err)
	}
	if fn := exactDottedFunc(p, "codec.nothere.mapEnd"); fn != nil {
		t.Errorf("unknown dotted name resolved to %v", fn)
	}
	if tryDottedSymbol("codec.nothere.mapEnd", nil) {
		t.Error("unknown dotted line claimed by the exact-symbol path")
	}
	if tryDottedSymbol("github.com/ugorji/go", nil) {
		t.Error("URL-shaped line claimed by the exact-symbol path")
	}
}

// The numeric pick: an ambiguous question prints the numbered
// pick-list (project-relative paths, no ../..), and replying with
// the number answers with that candidate's card.
func TestFuncPickNumericReply(t *testing.T) {
	withPickState(t)
	withEditsProject(t, dottedChatFixture(t))

	out, ok := tryCodebaseQuestion("what does mapEnd do")
	if !ok {
		t.Fatal("ambiguous question not answered")
	}
	if !strings.Contains(out, "matches 4 functions") || !strings.Contains(out, "codec/binc.mono.generated.go:") {
		t.Errorf("expected numbered pick-list with relative paths, got:\n%s", out)
	}
	if strings.Contains(out, "../..") {
		t.Errorf("pick-list must not contain ../.. paths, got:\n%s", out)
	}
	if pendingPick == nil || len(pendingPick.cands) != 4 {
		t.Fatalf("pick state not recorded: %+v", pendingPick)
	}
	want := pendingPick.cands[1]
	var handled bool
	pickOut := captureStdout(t, func() { handled = tryIntentDispatch("2", nil) })
	if !handled {
		t.Fatalf("numeric pick not handled, output:\n%s", pickOut)
	}
	if !strings.Contains(pickOut, want.Name) || !strings.Contains(pickOut, want.Summary()) {
		t.Errorf("pick 2 should answer with %s's card, got:\n%s", want.Display(), pickOut)
	}
	if pendingPick != nil {
		t.Error("pick state should be consumed by the reply")
	}
}

// Any other line forgets the pending pick instead of letting a
// stray number hijack an old list later.
func TestFuncPickForgottenByOtherLine(t *testing.T) {
	withPickState(t)
	withEditsProject(t, dottedChatFixture(t))
	if _, ok := tryCodebaseQuestion("what does mapEnd do"); !ok {
		t.Fatal("ambiguous question not answered")
	}
	if pendingPick == nil {
		t.Fatal("pick state not recorded")
	}
	captureStdout(t, func() { tryIntentDispatch("what is a goroutine", nil) })
	if pendingPick != nil {
		t.Error("unrelated line should clear the pending pick")
	}
}
