package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// dottedFixture builds the ugorji-shaped project from the reported
// failure: one package (codec) where the same method name (mapEnd)
// lives on several receivers, in long generated-style files, plus a
// second package with the same method name again.
func dottedFixture(t *testing.T) *Project {
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
	p, err := Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze fixture: %v", err)
	}
	return p
}

// The bare name is genuinely ambiguous: the pick-list appears, is
// numbered, and prints project-relative paths (never the cwd-relative
// "../.." form that wrapped mid-token on the Chromebook).
func TestAnswerAmbiguousDottedPickList(t *testing.T) {
	p := dottedFixture(t)
	out, ok := p.Answer("what does mapEnd do")
	if !ok {
		t.Fatal("expected an answer for the ambiguous name")
	}
	if !strings.Contains(out, "matches 4 functions") {
		t.Errorf("expected 4 candidates, got:\n%s", out)
	}
	if !strings.Contains(out, "1. codec.decoderBincBytes.mapEnd") {
		t.Errorf("expected a numbered candidate list, got:\n%s", out)
	}
	if !strings.Contains(out, "codec/binc.mono.generated.go:") {
		t.Errorf("expected project-relative path in pick-list, got:\n%s", out)
	}
	if strings.Contains(out, "../..") {
		t.Errorf("pick-list must not contain cwd-relative ../.. paths, got:\n%s", out)
	}
	if !strings.Contains(out, "number") {
		t.Errorf("pick-list should invite a numeric reply, got:\n%s", out)
	}
}

// Pasting the full dotted name from the list — in the question
// shape, in original case or retyped lowercase — resolves to exactly
// that function, never back to the pick-list.
func TestAnswerDottedNameResolves(t *testing.T) {
	p := dottedFixture(t)
	for _, q := range []string{
		"what does codec.decoderBincBytes.mapEnd do",
		"what does codec.decoderbincbytes.mapend do",
		"what does wire.reader.mapEnd do",
	} {
		out, ok := p.Answer(q)
		if !ok {
			t.Fatalf("expected an answer for %q", q)
		}
		if strings.Contains(out, "which one did you mean") {
			t.Errorf("%q should resolve, got the pick-list:\n%s", q, out)
		}
	}
	out, _ := p.Answer("what does codec.decoderBincBytes.mapEnd do")
	if !strings.Contains(out, "finishes reading a binc map") {
		t.Errorf("expected the binc mapEnd card, got:\n%s", out)
	}
	out, _ = p.Answer("what does wire.reader.mapEnd do")
	if !strings.Contains(out, "finishes reading a wire map") {
		t.Errorf("expected the wire mapEnd card, got:\n%s", out)
	}
}

// AnswerCands exposes the pick-list candidates so the chat layer can
// resolve a numeric reply; a resolved question yields none.
func TestAnswerCands(t *testing.T) {
	p := dottedFixture(t)
	_, cands, ok := p.AnswerCands("what does mapEnd do")
	if !ok || len(cands) != 4 {
		t.Fatalf("AnswerCands ambiguous: ok=%v cands=%d", ok, len(cands))
	}
	_, cands, ok = p.AnswerCands("what does codec.decoderBincBytes.mapEnd do")
	if !ok || cands != nil {
		t.Fatalf("AnswerCands resolved: ok=%v cands=%v", ok, cands)
	}
}
