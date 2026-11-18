package chat

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The basic-Linux-commands suite. The family's boundary is
// its first word, so the parse table pins both directions:
// command words claim (in their exact shapes), and every
// other sentence — including ones that merely contain a
// command word, or start with a word that contains one —
// falls through untouched. The behavior tests drive the
// real claim chain against fixture projects.

// withLinuxSession points the chat globals at dir (cold
// session) and resets the family's own session state, so
// each test captures its home root from the fixture.
func withLinuxSession(t *testing.T, dir string) {
	t.Helper()
	withChatSession(t, dir, false)
	oldHome, oldWork := linuxHomeDir, linuxWorkDir
	linuxHomeDir, linuxWorkDir = "", ""
	t.Cleanup(func() { linuxHomeDir, linuxWorkDir = oldHome, oldWork })
}

func linuxFixtureRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/linuxfixture\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"hi\")\n}\n")
	writeFile(t, dir, "notes.txt", "first line\nsecond line\n")
	writeFile(t, dir, "jake/main.go", "package main\n\nfunc main() {\n\tprintln(\"jake\")\n}\n")
	writeFile(t, dir, "jake/deep/deep.txt", "deep content\n")
	writeFile(t, dir, "jake/deep/deep.go", "package deep\n")
	writeFile(t, dir, "assets/logo.txt", "not code\n")
	return dir
}

func TestParseLinuxLine(t *testing.T) {
	cases := []struct {
		line  string
		shape linuxShape
		word  string
	}{
		// The allowlist, well-formed.
		{"pwd", linuxRun, "pwd"},
		{"PWD", linuxRun, "pwd"},
		{"pwd.", linuxRun, "pwd"},
		{"ls", linuxRun, "ls"},
		{"ls jake", linuxRun, "ls"},
		{"cd", linuxRun, "cd"},
		{"cd jake", linuxRun, "cd"},
		{"cd ..", linuxRun, "cd"},
		{"cat notes.txt", linuxRun, "cat"},
		{"echo", linuxRun, "echo"},
		{"echo hello world", linuxRun, "echo"},
		{"echo -n hello", linuxRun, "echo"},
		{"clear", linuxRun, "clear"},
		{"date", linuxRun, "date"},
		{"whoami", linuxRun, "whoami"},
		// Arity v1 doesn't take.
		{"pwd please", linuxBadArgs, "pwd"},
		{"date night", linuxBadArgs, "date"},
		{"ls a b", linuxBadArgs, "ls"},
		{"cat", linuxBadArgs, "cat"},
		{"cat a b", linuxBadArgs, "cat"},
		{"cd a b", linuxBadArgs, "cd"},
		// Shell tricks: claimed and refused by name.
		{"ls jake; rm all", linuxRefuseMeta, "ls"},
		{"echo $HOME", linuxRefuseMeta, "echo"},
		{"cat a > b", linuxRefuseMeta, "cat"},
		{"ls | less", linuxRefuseMeta, "ls"},
		{"echo `date`", linuxRefuseMeta, "echo"},
		{"cd .. && ls", linuxRefuseMeta, "cd"},
		// The refusal list: claimed for guidance. A refusal
		// word wins even with metacharacters after it.
		{"rm notes.txt", linuxRefuseWord, "rm"},
		{"rm -rf / ; echo x", linuxRefuseWord, "rm"},
		{"mkdir jake2", linuxRefuseWord, "mkdir"},
		{"mv a b", linuxRefuseWord, "mv"},
		{"cp a b", linuxRefuseWord, "cp"},
		{"touch new.txt", linuxRefuseWord, "touch"},
		{"rmdir jake", linuxRefuseWord, "rmdir"},
		{"chmod +x main.go", linuxRefuseWord, "chmod"},
		{"sudo ls", linuxRefuseWord, "sudo"},
		{"curl http://localhost:8080", linuxRefuseWord, "curl"},
		{"wget http://example.com/x", linuxRefuseWord, "wget"},
		// Not this family's business: falls through.
		{"what is the date today", linuxNone, ""},
		{"what does ls do", linuxNone, ""},
		{"list the files", linuxNone, ""},
		{"catalog the project", linuxNone, ""},
		{"echoes of the past", linuxNone, ""},
		{"show me the catalog", linuxNone, ""},
		{"the cat sat", linuxNone, ""},
		{"run the program jake/main.go", linuxNone, ""},
		{"add a print line in main to main.go", linuxNone, ""},
		{"create a folder jane", linuxNone, ""},
		{"", linuxNone, ""},
	}
	for _, tc := range cases {
		cmd := parseLinuxLine(tc.line)
		if cmd.shape != tc.shape || (tc.shape != linuxNone && cmd.word != tc.word) {
			t.Errorf("parseLinuxLine(%q) = (%v, %q), want (%v, %q)", tc.line, cmd.shape, cmd.word, tc.shape, tc.word)
		}
	}
	// The metachar refusal names the first character found.
	if cmd := parseLinuxLine("ls ; pwd"); cmd.meta != ';' {
		t.Errorf("meta for \"ls ; pwd\" = %q, want ';'", cmd.meta)
	}
	if cmd := parseLinuxLine("echo $HOME"); cmd.meta != '$' {
		t.Errorf("meta for \"echo $HOME\" = %q, want '$'", cmd.meta)
	}
}

// Fall-through at the claim level: the family itself
// returns false, so the rest of the chain sees the line
// untouched.
func TestLinuxCommandFallThrough(t *testing.T) {
	for _, line := range []string{
		"what is the date today", "list the files please",
		"catalog the project", "hello there",
	} {
		if runLinuxCommand(line, yes, &UserProfile{}, false) {
			t.Errorf("runLinuxCommand(%q) claimed a line outside the first-word boundary", line)
		}
	}
}

func TestLinuxPwdAndLs(t *testing.T) {
	dir := linuxFixtureRoot(t)
	writeFile(t, dir, ".hidden", "secret\n")
	withLinuxSession(t, dir)
	profile := &UserProfile{}
	root := mustAbs(t, dir)

	out := captureStdout(t, func() { claimChain("pwd", yes, profile) })
	if !strings.Contains(out, root) {
		t.Errorf("pwd should print the project root %q:\n%s", root, out)
	}

	out = captureStdout(t, func() { claimChain("ls", yes, profile) })
	for _, want := range []string{"go.mod", "jake/", "main.go", "notes.txt"} {
		if !strings.Contains(out, want) {
			t.Errorf("ls missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, ".hidden") {
		t.Errorf("ls showed a dotfile:\n%s", out)
	}
	// Sorted, one per line: go.mod before jake/ before main.go.
	gi, ji, mi := strings.Index(out, "go.mod"), strings.Index(out, "jake/"), strings.Index(out, "main.go")
	if !(gi >= 0 && gi < ji && ji < mi) {
		t.Errorf("ls entries not sorted (go.mod=%d jake/=%d main.go=%d):\n%s", gi, ji, mi, out)
	}

	out = captureStdout(t, func() { claimChain("ls jake", yes, profile) })
	if !strings.Contains(out, "deep/") || !strings.Contains(out, "main.go") {
		t.Errorf("ls jake wrong:\n%s", out)
	}

	// ls of a file prints the file's own name.
	out = captureStdout(t, func() { claimChain("ls notes.txt", yes, profile) })
	if !strings.Contains(out, "notes.txt") {
		t.Errorf("ls of a file wrong:\n%s", out)
	}

	// Missing and escaping targets refuse by name.
	out = captureStdout(t, func() { claimChain("ls ghost", yes, profile) })
	if !strings.Contains(out, "no such file or folder") {
		t.Errorf("ls ghost wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("ls ../outside", yes, profile) })
	if !strings.Contains(out, "climbs out of the project folder") {
		t.Errorf("ls escape wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("ls /etc", yes, profile) })
	if !strings.Contains(out, "isn't inside the project folder") {
		t.Errorf("ls absolute escape wrong:\n%s", out)
	}
}

func TestLinuxCat(t *testing.T) {
	dir := linuxFixtureRoot(t)
	withLinuxSession(t, dir)
	profile := &UserProfile{}

	out := captureStdout(t, func() { claimChain("cat notes.txt", yes, profile) })
	if !strings.Contains(out, "first line\nsecond line") {
		t.Errorf("cat wrong:\n%s", out)
	}

	out = captureStdout(t, func() { claimChain("cat jake/deep/deep.txt", yes, profile) })
	if !strings.Contains(out, "deep content") {
		t.Errorf("cat nested wrong:\n%s", out)
	}

	out = captureStdout(t, func() { claimChain("cat ghost.txt", yes, profile) })
	if !strings.Contains(out, "no such file") {
		t.Errorf("cat missing wrong:\n%s", out)
	}

	out = captureStdout(t, func() { claimChain("cat jake", yes, profile) })
	if !strings.Contains(out, "is a folder") {
		t.Errorf("cat a folder wrong:\n%s", out)
	}

	// Binary content is refused plainly, not dumped.
	writeFile(t, dir, "bin.dat", "a\x00b")
	out = captureStdout(t, func() { claimChain("cat bin.dat", yes, profile) })
	if !strings.Contains(out, "doesn't look like a text file") || strings.Contains(out, "a\x00b") {
		t.Errorf("cat binary wrong:\n%q", out)
	}

	// A big file is cut at the cap, and the cut is announced.
	big := strings.Repeat("x", linuxCatCap+5000)
	writeFile(t, dir, "big.txt", big)
	out = captureStdout(t, func() { claimChain("cat big.txt", yes, profile) })
	if !strings.Contains(out, "stopped there") || !strings.Contains(out, fmt.Sprint(linuxCatCap+5000)) {
		t.Errorf("cat truncation note wrong (out len %d)", len(out))
	}
	if len(out) > linuxCatCap+1000 {
		t.Errorf("cat printed past the cap (out len %d)", len(out))
	}
}

func TestLinuxEchoDateWhoamiClear(t *testing.T) {
	dir := linuxFixtureRoot(t)
	withLinuxSession(t, dir)
	profile := &UserProfile{}

	out := captureStdout(t, func() { claimChain("echo hello there world", yes, profile) })
	if !strings.Contains(out, "hello there world") {
		t.Errorf("echo wrong:\n%s", out)
	}
	// Flag-looking tokens print literally in v1.
	out = captureStdout(t, func() { claimChain("echo -n no newline", yes, profile) })
	if !strings.Contains(out, "-n no newline") {
		t.Errorf("echo flags should print literally:\n%s", out)
	}
	// Quotes are not shell quotes: they print as typed.
	out = captureStdout(t, func() { claimChain("echo \"hi there\"", yes, profile) })
	if !strings.Contains(out, "\"hi there\"") {
		t.Errorf("echo quotes should print as typed:\n%s", out)
	}

	out = captureStdout(t, func() { claimChain("date", yes, profile) })
	dateRe := regexp.MustCompile(`\w{3} \w{3} [ \d]\d \d{2}:\d{2}:\d{2} \S+ \d{4}`)
	if !dateRe.MatchString(out) {
		t.Errorf("date shape wrong:\n%s", out)
	}

	name := linuxWhoami()
	if name == "" {
		t.Skip("no OS user name available in this environment")
	}
	out = captureStdout(t, func() { claimChain("whoami", yes, profile) })
	if !strings.Contains(out, name) {
		t.Errorf("whoami wrong: want %q in:\n%s", name, out)
	}

	// clear off a terminal does nothing at all; on one it
	// emits the clear sequence (and nothing else).
	out = captureStdout(t, func() { runLinuxCommand("clear", yes, profile, false) })
	if strings.Contains(out, "\x1b[2J") {
		t.Errorf("clear off-TTY should not emit escapes:\n%q", out)
	}
	out = captureStdout(t, func() { runLinuxCommand("clear", yes, profile, true) })
	if out != "\x1b[H\x1b[2J" {
		t.Errorf("clear on-TTY = %q, want exactly the clear sequence", out)
	}
}

func TestLinuxGateDeclineAndAutoYes(t *testing.T) {
	dir := linuxFixtureRoot(t)
	withLinuxSession(t, dir)
	no := func(string) (string, error) { return "n", nil }

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("pwd", no, &UserProfile{}) })
	if !handled || !strings.Contains(out, "OK — not run.") || strings.Contains(out, dir) {
		t.Errorf("declined pwd should print nothing of the path:\n%s", out)
	}

	// The 'do all' directive auto-yeses without consulting
	// the ask at all (this ask errors if it is ever called).
	boom := func(string) (string, error) { return "", fmt.Errorf("ask must not be called") }
	profile := &UserProfile{Directives: map[string]bool{"do_all_means_all": true}}
	out = captureStdout(t, func() { handled = claimChain("pwd", boom, profile) })
	if !handled || !strings.Contains(out, "[auto-yes: your 'do all' directive]") || !strings.Contains(out, mustAbs(t, dir)) {
		t.Errorf("auto-yes pwd wrong:\n%s", out)
	}
}

func TestLinuxRefusals(t *testing.T) {
	dir := linuxFixtureRoot(t)
	withLinuxSession(t, dir)
	profile := &UserProfile{}

	cases := []struct{ line, want string }{
		{"rm notes.txt", "doesn't delete files"},
		{"mkdir jake2", "create a folder <name>"},
		{"mv a b", "I don't run mv"},
		{"cp a b", "I don't run cp"},
		{"touch new.txt", "I don't run touch"},
		{"rmdir jake", "I don't run rmdir"},
		{"chmod +x main.go", "I don't run chmod"},
		{"sudo ls", "I don't run sudo"},
		{"curl http://localhost:8080", "I don't run curl"},
		{"wget http://example.com/x", "I don't run wget"},
	}
	for _, tc := range cases {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(tc.line, yes, profile) })
		if !handled {
			t.Errorf("%q not claimed", tc.line)
			continue
		}
		if !strings.Contains(out, tc.want) {
			t.Errorf("%q: output missing %q:\n%s", tc.line, tc.want, out)
		}
	}
	// The refusal happened instead of any effect.
	if _, err := os.Stat(filepath.Join(dir, "notes.txt")); err != nil {
		t.Error("rm must never run: notes.txt is gone")
	}
	if _, err := os.Stat(filepath.Join(dir, "jake2")); err == nil {
		t.Error("mkdir must never run: jake2 exists")
	}

	// Metacharacters are named, and nothing runs.
	out := captureStdout(t, func() { claimChain("ls jake; cat notes.txt", yes, profile) })
	if !strings.Contains(out, "';'") && !strings.Contains(out, "\";\"") && !strings.Contains(out, ";") {
		t.Errorf("meta refusal should name the character:\n%s", out)
	}
	if strings.Contains(out, "first line") {
		t.Errorf("a meta line ran something:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("echo $HOME", yes, profile) })
	if !strings.Contains(out, "$") || !strings.Contains(out, "no shell") {
		t.Errorf("echo $ refusal wrong:\n%s", out)
	}
}

func TestLinuxCdFlow(t *testing.T) {
	dir := linuxFixtureRoot(t)
	withLinuxSession(t, dir)
	profile := &UserProfile{}
	root := mustAbs(t, dir)
	jake := filepath.Join(root, "jake")
	deep := filepath.Join(jake, "deep")

	// Into a subfolder: announced in project terms, and the
	// session's project follows the move.
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("cd jake", yes, profile) })
	if !handled || !strings.Contains(out, "Now working in jake/") {
		t.Fatalf("cd jake wrong:\n%s", out)
	}
	if linuxWorkDir != jake {
		t.Errorf("linuxWorkDir = %q, want %q", linuxWorkDir, jake)
	}
	if lastAnalyzeRoot != jake || lastAnalyzeProject == nil || lastAnalyzeProject.Root != jake {
		t.Errorf("analyzer did not follow cd: root=%q", lastAnalyzeRoot)
	}

	// pwd and ls now speak about jake/.
	out = captureStdout(t, func() { claimChain("pwd", yes, profile) })
	if !strings.Contains(out, jake) {
		t.Errorf("pwd after cd wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("ls", yes, profile) })
	if !strings.Contains(out, "deep/") || strings.Contains(out, "notes.txt") {
		t.Errorf("ls after cd wrong:\n%s", out)
	}

	// Deeper, then .. one step, then bare cd home.
	out = captureStdout(t, func() { claimChain("cd deep", yes, profile) })
	if !strings.Contains(out, "Now working in jake/deep/") || linuxWorkDir != deep {
		t.Errorf("cd deep wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("cd ..", yes, profile) })
	if !strings.Contains(out, "Now working in jake/") || linuxWorkDir != jake {
		t.Errorf("cd .. wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("cd", yes, profile) })
	if !strings.Contains(out, "Now working in") || linuxWorkDir != root || lastAnalyzeRoot != root {
		t.Errorf("bare cd home wrong:\n%s", out)
	}

	// cd .. at the top stops there, with the note, and the
	// session stays home.
	out = captureStdout(t, func() { claimChain("cd ..", yes, profile) })
	if !strings.Contains(out, "stops here") || linuxWorkDir != root {
		t.Errorf("cd .. at root wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("cd ../../..", yes, profile) })
	if !strings.Contains(out, "stops here") || linuxWorkDir != root {
		t.Errorf("cd ../../.. at root wrong:\n%s", out)
	}

	// A folder with no Go code in it can't become the
	// working folder — there's no project to re-aim at —
	// and the session stays home.
	out = captureStdout(t, func() { claimChain("cd assets", yes, profile) })
	if !strings.Contains(out, "has no Go code in it") || linuxWorkDir != root {
		t.Errorf("cd into a no-Go folder wrong:\n%s", out)
	}

	// A sideways escape refuses by name; a file isn't a
	// folder; a missing folder is named.
	out = captureStdout(t, func() { claimChain("cd ../outside", yes, profile) })
	if !strings.Contains(out, "climbs out of the project folder") {
		t.Errorf("cd escape wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("cd main.go", yes, profile) })
	if !strings.Contains(out, "is a file, not a folder") {
		t.Errorf("cd onto file wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("cd ghost", yes, profile) })
	if !strings.Contains(out, "There's no folder ghost") {
		t.Errorf("cd missing wrong:\n%s", out)
	}

	// An absolute path inside the project works; one
	// outside refuses.
	out = captureStdout(t, func() { claimChain("cd "+jake, yes, profile) })
	if !strings.Contains(out, "Now working in jake/") || linuxWorkDir != jake {
		t.Errorf("cd absolute inside wrong:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("cd /etc", yes, profile) })
	if !strings.Contains(out, "isn't inside the project folder") || linuxWorkDir != jake {
		t.Errorf("cd absolute outside wrong:\n%s", out)
	}
}

// cd while a program runs: the program was started with
// its own folder and binary, so the move doesn't touch it —
// top still lists it and stop still stops it.
func TestLinuxCdWhileProgramRuns(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", analyze.WebServerSource)
	writeFile(t, dir, "sub/keep.txt", "x\n")
	writeFile(t, dir, "sub/keep.go", "package sub\n")
	opened := withRunSession(t, dir)
	oldHome, oldWork := linuxHomeDir, linuxWorkDir
	linuxHomeDir, linuxWorkDir = "", ""
	t.Cleanup(func() { linuxHomeDir, linuxWorkDir = oldHome, oldWork })
	_ = opened
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("server did not start:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("cd sub", yes, profile) })
	if !handled || !strings.Contains(out, "Now working in sub/") {
		t.Fatalf("cd while running wrong:\n%s", out)
	}
	if !programRunning() {
		t.Fatal("cd stopped the running program")
	}

	out = captureStdout(t, func() { handled = claimChain("top", yes, profile) })
	if !handled || !strings.Contains(out, "srv/main.go — running for") {
		t.Errorf("top after cd wrong:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || !strings.Contains(out, "Stopped srv/main.go") {
		t.Errorf("stop after cd wrong:\n%s", out)
	}
	if !portFree("8080") {
		t.Error("port 8080 still bound after stop")
	}
}
