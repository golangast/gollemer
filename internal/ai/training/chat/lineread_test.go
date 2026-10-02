package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Completion is pure logic over the buffer: verbs, slash commands, make
// targets, file paths, and analyzed symbols.

func TestCompleteVerb(t *testing.T) {
	r := NewLineReader(strings.NewReader(""))
	cands, tokStart := r.complete([]rune("wal"), 3)
	if tokStart != 0 || len(cands) != 1 || cands[0] != "walk me through " {
		t.Errorf("complete(wal) = %v @%d, want [walk me through ] @0", cands, tokStart)
	}
}

func TestCompleteVerbCaseInsensitive(t *testing.T) {
	r := NewLineReader(strings.NewReader(""))
	cands, _ := r.complete([]rune("WAL"), 3)
	if len(cands) != 1 || cands[0] != "walk me through " {
		t.Errorf("complete(WAL) = %v, want [walk me through ]", cands)
	}
}

func TestCompleteSlash(t *testing.T) {
	r := NewLineReader(strings.NewReader(""))
	cands, tokStart := r.complete([]rune("/fl"), 3)
	if tokStart != 0 || len(cands) != 1 || cands[0] != "/flow " {
		t.Errorf("complete(/fl) = %v @%d, want [/flow ] @0", cands, tokStart)
	}
}

func TestCompleteMakeTargets(t *testing.T) {
	old := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{"eval": true, "chat": true, "explain": true}
	defer func() { makeTargetsAllowlist = old }()

	r := NewLineReader(strings.NewReader(""))
	cands, tokStart := r.complete([]rune("run make e"), 10)
	if tokStart != 9 {
		t.Fatalf("tokStart = %d, want 9", tokStart)
	}
	if len(cands) != 2 || cands[0] != "eval" || cands[1] != "explain" {
		t.Errorf("complete(run make e) = %v, want [eval explain]", cands)
	}
}

func TestCompleteMakeTargetsCommonPrefix(t *testing.T) {
	old := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{"eval": true, "evaluate": true}
	defer func() { makeTargetsAllowlist = old }()

	r := NewLineReader(strings.NewReader(""))
	buf, pos, listed := r.doComplete("you> ", []rune("run make ev"), 11, false)
	if listed || string(buf) != "run make eval" || pos != 13 {
		t.Errorf("doComplete = %q @%d listed=%v, want %q @13", string(buf), pos, listed, "run make eval")
	}
}

func TestCompletePath(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "alpha.go"), []byte("x"), 0o644)
	os.Mkdir(filepath.Join(dir, "alphabet"), 0o755)

	r := NewLineReader(strings.NewReader(""))
	cands, _ := r.complete([]rune("analyze "+dir+"/alph"), len("analyze "+dir+"/alph"))
	if len(cands) != 2 {
		t.Fatalf("complete(path) = %v, want 2 candidates", cands)
	}
}

func TestCompleteSymbols(t *testing.T) {
	// symbolNames needs an analyzed project; without one it yields nil.
	last := lastAnalyzeProject
	lastAnalyzeProject = nil
	defer func() { lastAnalyzeProject = last }()
	if got := symbolNames(); got != nil {
		t.Errorf("symbolNames() without a project = %v, want nil", got)
	}
}

func TestCommonPrefix(t *testing.T) {
	if got := commonPrefix([]string{"eval", "explain", "exit"}); got != "e" {
		t.Errorf("commonPrefix = %q, want e", got)
	}
	if got := commonPrefix([]string{"eval"}); got != "eval" {
		t.Errorf("commonPrefix single = %q, want eval", got)
	}
	if got := commonPrefix([]string{"abc", "xyz"}); got != "" {
		t.Errorf("commonPrefix disjoint = %q, want empty", got)
	}
}

// History: in-memory ordering for up/down, no consecutive duplicates,
// and piped (non-terminal) readers never record.
func TestAddHistory(t *testing.T) {
	r := &LineReader{term: os.Stdin, histPos: -1} // non-nil term: in-memory path only
	r.addHistory("first")
	r.addHistory("second")
	r.addHistory("second") // duplicate: dropped
	r.addHistory("   ")    // blank: dropped
	if len(r.history) != 2 || r.history[0] != "first" || r.history[1] != "second" {
		t.Errorf("history = %v, want [first second]", r.history)
	}
}

func TestPipedReaderRecordsNoHistory(t *testing.T) {
	r := NewLineReader(strings.NewReader("hello\n"))
	if r.term != nil {
		t.Fatal("strings.Reader must not be treated as a terminal")
	}
	line, err := r.ReadLine("you> ")
	if err != nil || line != "hello" {
		t.Errorf("ReadLine = %q, %v; want hello, nil", line, err)
	}
}

// Raw key parsing is exercised through a pipe: escape sequences decode
// to logical keys without needing a real terminal.
func TestReadKeyEscapes(t *testing.T) {
	if !rawSupported {
		t.Skip("raw mode is linux-only")
	}
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	defer w.Close()
	cases := []struct {
		in   []byte
		want keyCode
	}{
		{[]byte{'\r'}, keyEnter},
		{[]byte{'\t'}, keyTab},
		{[]byte{127}, keyBackspace},
		{[]byte{27, '[', 'A'}, keyUp},
		{[]byte{27, '[', 'B'}, keyDown},
		{[]byte{27, '[', 'C'}, keyRight},
		{[]byte{27, '[', 'D'}, keyLeft},
		{[]byte{27, '[', 'H'}, keyHome},
		{[]byte{27, '[', 'F'}, keyEnd},
		{[]byte{27, '[', '3', '~'}, keyDelete},
		{[]byte{27, 'O', 'A'}, keyUp},
		{[]byte{3}, keyCtrlC},
		{[]byte{4}, keyCtrlD},
		{[]byte{1}, keyCtrlA},
		{[]byte{5}, keyCtrlE},
		{[]byte{21}, keyCtrlU},
		{[]byte{11}, keyCtrlK},
		{[]byte{23}, keyCtrlW},
	}
	for _, tc := range cases {
		if _, err := w.Write(tc.in); err != nil {
			t.Fatal(err)
		}
		got, _ := readKey(r)
		if got != tc.want {
			t.Errorf("readKey(%v) = %v, want %v", tc.in, got, tc.want)
		}
	}
	// Lone ESC (nothing follows) is ignored, not an error.
	if _, err := w.Write([]byte{27}); err != nil {
		t.Fatal(err)
	}
	if got, _ := readKey(r); got != keyIgnore {
		t.Errorf("lone ESC = %v, want keyIgnore", got)
	}
	// UTF-8 decodes to a rune.
	if _, err := w.Write([]byte("é")); err != nil {
		t.Fatal(err)
	}
	if got, ch := readKey(r); got != keyChar || ch != 'é' {
		t.Errorf("utf-8 = %v %q, want keyChar 'é'", got, ch)
	}
}
