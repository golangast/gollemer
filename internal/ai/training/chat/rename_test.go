package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// Every documented rename sentence form parses to the request it
// names.
func TestParseRenameSentenceForms(t *testing.T) {
	cases := []struct {
		line string
		want analyze.EditRequest
	}{
		{"rename Describe to Explain", analyze.EditRequest{Kind: analyze.EditRename, RenameFrom: "Describe", RenameTo: "Explain"}},
		{"rename the function Describe to Explain", analyze.EditRequest{Kind: analyze.EditRename, RenameKind: "function", RenameFrom: "Describe", RenameTo: "Explain"}},
		{"rename the type State to Status", analyze.EditRequest{Kind: analyze.EditRename, RenameKind: "type", RenameFrom: "State", RenameTo: "Status"}},
		{"rename ModeFast to ModeQuick in other", analyze.EditRequest{Kind: analyze.EditRename, RenameFrom: "ModeFast", RenameTo: "ModeQuick", RenameScope: "other"}},
		{"rename shop.Describe to Explain", analyze.EditRequest{Kind: analyze.EditRename, RenameFrom: "shop.Describe", RenameTo: "Explain"}},
		{"rename Config.Host to Hostname", analyze.EditRequest{Kind: analyze.EditRename, RenameFrom: "Config.Host", RenameTo: "Hostname"}},
		{"rename Describe to Explain.", analyze.EditRequest{Kind: analyze.EditRename, RenameFrom: "Describe", RenameTo: "Explain"}},
	}
	for _, c := range cases {
		got, ok := parseRenameSentence(c.line)
		if !ok {
			t.Errorf("parseRenameSentence(%q) not claimed", c.line)
			continue
		}
		if got != c.want {
			t.Errorf("parseRenameSentence(%q) = %+v, want %+v", c.line, got, c.want)
		}
	}
}

// Rename sentences and other brains' sentences stay in their own
// lanes, in both directions.
func TestRenameNonInterference(t *testing.T) {
	// The assembled-edit parser must not claim rename sentences.
	for _, line := range []string{"rename Describe to Explain", "rename the type State to Status"} {
		if req, ok := parseAssembledEdit(line); ok {
			t.Errorf("parseAssembledEdit(%q) claimed a rename as %+v", line, req)
		}
	}
	// The rename parser/flow must not claim other brains' lines:
	// the 13-sentence edits probe set, the symbolpick dotted line,
	// and questions ABOUT renaming.
	for _, line := range []string{
		"where does Items change",
		"what changes Items",
		"where does Items get its data",
		"where does counts come from",
		"where would I add a retry helper",
		"what does Describe do",
		"where is Describe",
		"add a getter for Timeout",
		"add a test for Describe",
		"add a comment to Describe",
		"add an import for fmt",
		"how do I add a case to a switch",
		"add it to the list of things to do",
		"codec.decoderBincBytes.mapEnd",
		"how do I rename a variable in Go",
		"rename",
		"rename Describe",
	} {
		if req, ok := parseRenameSentence(line); ok {
			t.Errorf("parseRenameSentence(%q) claimed as %+v — that sentence belongs to another brain", line, req)
		}
		if runRenameEdit(line, yes) {
			t.Errorf("runRenameEdit(%q) claimed a non-rename sentence", line)
		}
	}
}

// A cross-package rename end to end: the const in `other` and its
// selector use in shop change together, behind the one gate.
func TestRenameEndToEndCrossPackage(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	if !runRenameEdit("rename ModeFast to ModeQuick", yes) {
		t.Fatal("rename sentence not claimed")
	}
	if got := readBack(t, dir, "other/other.go"); !strings.Contains(got, "ModeQuick = 1") {
		t.Errorf("definition not renamed:\n%s", got)
	}
	if got := readBack(t, dir, "shop/defs.go"); !strings.Contains(got, "other.ModeQuick") {
		t.Errorf("cross-package use not renamed:\n%s", got)
	}
}

// Decline and EOF at the gate leave every byte as it was.
func TestRenameDeclineAndEOFLeaveBytes(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	before := readBack(t, dir, "shop/shop.go")
	if !runRenameEdit("rename Describe to Explain", no) {
		t.Fatal("rename sentence not claimed")
	}
	if after := readBack(t, dir, "shop/shop.go"); after != before {
		t.Fatal("declined rename changed the file")
	}
	runRenameEdit("rename Describe to Explain", eof)
	if after := readBack(t, dir, "shop/shop.go"); after != before {
		t.Fatal("EOF at the rename gate must not apply")
	}
}

// Ambiguity prints a numbered list and arms the pick; the number
// answers it — first declining (bytes identical), then accepting
// (only the chosen candidate changes).
func TestRenameAmbiguousPickFlow(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	oneBefore := readBack(t, dir, "one/one.go")
	twoBefore := readBack(t, dir, "two/two.go")
	asked := false
	flagAsk := func(string) (string, error) { asked = true; return "y", nil }

	if !runRenameEdit("rename State to Status", flagAsk) {
		t.Fatal("rename sentence not claimed")
	}
	if asked {
		t.Fatal("an ambiguous rename reached the [y/n] gate instead of offering the pick list")
	}
	// "2" picks two.State; declining there changes nothing.
	if !runRenameEdit("2", no) {
		t.Fatal("numeric pick not claimed")
	}
	if readBack(t, dir, "one/one.go") != oneBefore || readBack(t, dir, "two/two.go") != twoBefore {
		t.Fatal("a declined pick changed a file")
	}
	// Offer again, then "1" + yes renames only one.State.
	if !runRenameEdit("rename State to Status", flagAsk) {
		t.Fatal("rename sentence not claimed on re-offer")
	}
	if !runRenameEdit("1", yes) {
		t.Fatal("numeric pick not claimed on re-offer")
	}
	if got := readBack(t, dir, "one/one.go"); !strings.Contains(got, "type Status struct") {
		t.Errorf("the picked candidate was not renamed:\n%s", got)
	}
	if got := readBack(t, dir, "two/two.go"); got != twoBefore {
		t.Errorf("the unpicked candidate changed:\n%s", got)
	}
}

// A collision refuses before the gate; a generated file too.
func TestRenameRefusalsNeverReachGate(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	asked := false
	flagAsk := func(string) (string, error) { asked = true; return "y", nil }
	before := readBack(t, dir, "shop/shop.go")
	runRenameEdit("rename Describe to ModeName", flagAsk) // ModeName already exists in shop
	if asked {
		t.Fatal("a colliding rename reached the gate")
	}
	runRenameEdit("rename GeneratedPick to GenPick", flagAsk) // declared in a generated file
	if asked {
		t.Fatal("a generated-file rename reached the gate")
	}
	if after := readBack(t, dir, "shop/shop.go"); after != before {
		t.Fatal("a refused rename changed a file")
	}
}

// A rename the planner can't foresee breaking the build — the
// method satisfies io.Reader only outside the project — is caught
// by the spine's build check and rolled back byte-identical.
func TestRenameBuildFailureRollsBack(t *testing.T) {
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
	write("go.mod", "module example.com/readrename\n\ngo 1.21\n")
	write("lib/lib.go", `package lib

type Source struct{}

func (Source) Read(p []byte) (int, error) { return 0, nil }
`)
	write("app/app.go", `package app

import (
	"io"

	"example.com/readrename/lib"
)

var _ io.Reader = lib.Source{}
`)
	withEditsProject(t, dir)
	libBefore := readBack(t, dir, "lib/lib.go")
	appBefore := readBack(t, dir, "app/app.go")
	if !runRenameEdit("rename Read to ReadAll", yes) {
		t.Fatal("rename sentence not claimed")
	}
	if got := readBack(t, dir, "lib/lib.go"); got != libBefore {
		t.Errorf("build-failing rename was not rolled back:\n%s", got)
	}
	if got := readBack(t, dir, "app/app.go"); got != appBefore {
		t.Errorf("app file changed by a rolled-back rename:\n%s", got)
	}
}
