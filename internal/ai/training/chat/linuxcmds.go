package chat

// Basic Linux commands (2026-11-01, John's "and do pwd and
// all the basic Linux commands", sent right after his
// run/stop screenshots): the chat answers the beginner
// terminal vocabulary itself — pwd, ls, cd, cat, echo,
// clear, date, whoami — instead of letting the makefile
// brain reinterpret "ls" as a make target or the social
// brain make small talk out of "pwd".
//
// The family is seated with the run family, directly after
// run-the-program and ahead of the beginner five, the
// intent layer, and the brains. Its boundary is the first
// word: a line is this family's only when its first word is
// a command word it knows — one of the allowlist below, or
// one of the refusal words it answers with guidance. Every
// other sentence falls through untouched, pinned both ways
// in the tests.
//
// The safety idiom is the product's own:
//
//   - NO SHELL EVER. Nothing here is executed through sh or
//     bash, and no command is a subprocess at all: each one
//     is implemented natively in-process (a directory read,
//     a file read, the clock, the OS user). A line whose
//     first word is a command word but which contains a
//     shell metacharacter (; & | > < ` $) is claimed and
//     refused, naming the character — there are no pipes,
//     redirects, globs, or chaining to have.
//   - The established run gate: every command that runs
//     runs behind the same [y/n] as run-the-program, and a
//     profile 'do all' directive auto-yeses with the same
//     announcement. Refusals and guidance are not gated —
//     nothing runs.
//   - Confinement: every path resolves inside the session's
//     project root (the folder the chat's session is rooted
//     at when the family first engages). A path that climbs
//     out — ../ past the root, or an absolute path outside
//     it — is refused by name. cd can move the working
//     folder anywhere inside that root, and no further.
//
// cd is the one stateful command: it re-roots the session.
// The current project becomes the new folder (analyzed on
// the spot through the same setCurrentProject machinery
// every project switch uses), so creations, edits, and
// run-the-program follow the move, and the analyzer's
// answers are about the new folder. The family's own home
// root never moves: bare cd returns to it, and cd .. stops
// at it with a plain note. A program the run family already
// started is untouched by cd — it was started with its own
// folder and binary, and top/stop keep working on it.

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/user"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// ---------------------------------------------------------------------
// Parsing: the first-word boundary.

// linuxAllowWords are the commands v1 runs. linuxRefuseWords
// are first words the family claims only to answer with
// guidance — they never execute and never fall through.
var linuxAllowWords = map[string]bool{
	"pwd": true, "ls": true, "cd": true, "cat": true,
	"echo": true, "clear": true, "date": true, "whoami": true,
}

var linuxRefuseWords = map[string]bool{
	"rm": true, "mkdir": true, "mv": true, "cp": true,
	"touch": true, "rmdir": true, "chmod": true, "sudo": true,
	"curl": true, "wget": true,
}

type linuxShape int

const (
	linuxNone       linuxShape = iota
	linuxRun                   // an allowlisted command, well-formed
	linuxRefuseWord            // a refusal word: guidance, never a run
	linuxRefuseMeta            // allowlisted word + shell metacharacters
	linuxBadArgs               // allowlisted word, arguments v1 doesn't take
)

// linuxCmd is one parsed line: the command word, its
// arguments as typed, and — for the refusal shapes — the
// facts the answer names (the word itself, the metachar,
// the argument problem).
type linuxCmd struct {
	shape linuxShape
	word  string
	args  []string
	meta  byte   // linuxRefuseMeta: the character found
	note  string // linuxBadArgs: what the command takes instead
}

// linuxFoldCmd folds a typed first word for matching: case
// doesn't matter for a command word, and neither does a
// trailing sentence mark.
func linuxFoldCmd(s string) string {
	return strings.ToLower(strings.TrimRight(s, ".!?"))
}

// linuxMetaChar returns the first shell metacharacter in
// the line, or 0. The set is the brief's: ; & | > < plus
// backtick and $ (which covers $( too).
func linuxMetaChar(line string) byte {
	for i := 0; i < len(line); i++ {
		switch line[i] {
		case ';', '&', '|', '>', '<', '`', '$':
			return line[i]
		}
	}
	return 0
}

// parseLinuxLine classifies one chat line for this family.
// It claims only on the first word; a refusal word wins
// over the metachar check (its guidance is the more useful
// answer), and the metachar check wins over arity (the line
// isn't one command at all until the trick is out of it).
func parseLinuxLine(line string) linuxCmd {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 {
		return linuxCmd{shape: linuxNone}
	}
	word := linuxFoldCmd(fields[0])
	cmd := linuxCmd{word: word, args: fields[1:]}
	switch {
	case linuxRefuseWords[word]:
		cmd.shape = linuxRefuseWord
		return cmd
	case !linuxAllowWords[word]:
		return linuxCmd{shape: linuxNone}
	}
	if c := linuxMetaChar(line); c != 0 {
		cmd.shape = linuxRefuseMeta
		cmd.meta = c
		return cmd
	}
	// Arity, per command. echo takes anything (including
	// nothing); the rest are exact in v1.
	switch word {
	case "pwd", "date", "whoami", "clear":
		if len(cmd.args) > 0 {
			cmd.shape = linuxBadArgs
			cmd.note = fmt.Sprintf("%s doesn't take any extra words here — say just \"%s\"", word, word)
			return cmd
		}
	case "ls":
		if len(cmd.args) > 1 {
			cmd.shape = linuxBadArgs
			cmd.note = "ls takes one folder at a time here — say \"ls\" or \"ls <folder>\""
			return cmd
		}
	case "cd":
		if len(cmd.args) > 1 {
			cmd.shape = linuxBadArgs
			cmd.note = "cd takes one folder at a time here — say \"cd <folder>\", or just \"cd\" to go back to the folder the chat started in"
			return cmd
		}
	case "cat":
		if len(cmd.args) == 0 {
			cmd.shape = linuxBadArgs
			cmd.note = "cat needs a file: say \"cat <file>\", like \"cat main.go\""
			return cmd
		}
		if len(cmd.args) > 1 {
			cmd.shape = linuxBadArgs
			cmd.note = "cat takes one file at a time here — say \"cat <file>\""
			return cmd
		}
	}
	cmd.shape = linuxRun
	return cmd
}

// ---------------------------------------------------------------------
// Session state: the family's home root and working folder.

// linuxHomeDir is the session's project root for this
// family — the confinement boundary and bare cd's target —
// and linuxWorkDir is the folder commands currently work
// in. Both are absolute, symlink-resolved, and captured
// together the first time a command needs them (from
// creationRoot: the open project's root, else the folder
// the chat started in — the run family's plumbing).
// linuxHomeDir never moves afterwards; cd moves only
// linuxWorkDir, always inside linuxHomeDir.
var (
	linuxHomeDir string
	linuxWorkDir string
)

// linuxSessionDirs returns the family's home and working
// folders, capturing them on first use.
func linuxSessionDirs() (home, cur, refusal string) {
	if linuxWorkDir != "" {
		return linuxHomeDir, linuxWorkDir, ""
	}
	base := creationRoot()
	if base == "" {
		return "", "", "I don't have a project folder to work in — start me inside your project and ask again."
	}
	resolved, err := filepath.EvalSymlinks(base)
	if err != nil {
		abs, aerr := filepath.Abs(base)
		if aerr != nil {
			return "", "", fmt.Sprintf("I couldn't find the folder I'd work in (%s).", base)
		}
		resolved = abs
	}
	linuxHomeDir, linuxWorkDir = resolved, resolved
	return linuxHomeDir, linuxWorkDir, ""
}

// linuxDisplay renders an absolute folder inside home the
// way the chat talks about it: home-relative with a
// trailing slash ("jake/", "jake/sub/"), or the home
// folder's own name ("gollemer/") for home itself.
func linuxDisplay(home, abs string) string {
	rel, err := filepath.Rel(home, abs)
	if err != nil || rel == "." {
		return filepath.Base(home) + "/"
	}
	return filepath.ToSlash(rel) + "/"
}

// linuxResolve turns a typed path into an absolute path
// inside home, resolved against cur, with symlinks
// followed — the same containment fenceResolve keeps for
// creations. A path that lands outside home is refused by
// name; the caller stats the result for existence.
func linuxResolve(home, cur, typed string) (abs, refusal string) {
	cand := typed
	if !filepath.IsAbs(cand) {
		cand = filepath.Join(cur, cand)
	}
	cand = filepath.Clean(cand)
	if resolved, err := filepath.EvalSymlinks(cand); err == nil {
		cand = resolved
	}
	rel, err := filepath.Rel(home, cand)
	switch {
	case err != nil:
		return "", fmt.Sprintf("%q isn't inside the project folder — these commands only work inside %s.", typed, linuxDisplay(home, home))
	case rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)):
		if filepath.IsAbs(typed) {
			return "", fmt.Sprintf("%q isn't inside the project folder — these commands only work inside %s.", typed, linuxDisplay(home, home))
		}
		return "", fmt.Sprintf("%q climbs out of the project folder — these commands only work inside %s.", typed, linuxDisplay(home, home))
	}
	return cand, ""
}

// linuxAboveHome reports whether a resolved path is home
// itself or one of its ancestors — the cd .. case, which
// stops at the root with a note instead of the escape
// refusal a sideways path gets. (Judged by Rel so the
// filesystem root itself counts as an ancestor.)
func linuxAboveHome(home, cand string) bool {
	rel, err := filepath.Rel(cand, home)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

// ---------------------------------------------------------------------
// The claim.

// tryLinuxCommand is the chat-loop entry, seated directly
// after run-the-program.
func tryLinuxCommand(line string, r *LineReader, profile *UserProfile) bool {
	return runLinuxCommand(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	}, profile, r != nil && r.term != nil)
}

// runLinuxCommand carries one line for this claim. It
// returns false for every line outside the first-word
// boundary, so the rest of the chain sees them untouched.
// interactive says whether the session is a real terminal
// (it decides what clear does).
func runLinuxCommand(line string, ask editAsk, profile *UserProfile, interactive bool) bool {
	cmd := parseLinuxLine(line)
	switch cmd.shape {
	case linuxNone:
		return false
	case linuxRefuseWord:
		linuxRefuseAnswer(cmd.word)
		return true
	case linuxRefuseMeta:
		tag := cTag("gollemer [terminal]>")
		fmt.Printf("%s That line has a %q in it — I run one basic command at a time, with no shell behind it, so no ; & | > < ` or $ tricks. Say just the command you want, like \"ls\" or \"cat main.go\".\n", tag, cmd.meta)
		recordBrain("runner", false, fmt.Sprintf("linux refused: metachar %q in %q", cmd.meta, line))
		return true
	case linuxBadArgs:
		tag := cTag("gollemer [terminal]>")
		fmt.Printf("%s %s.\n", tag, cmd.note)
		recordBrain("runner", false, "linux refused: bad args: "+line)
		return true
	}
	linuxRunCmd(cmd, ask, profile, interactive)
	return true
}

// linuxRefuseAnswer answers a refusal word with guidance.
// Nothing runs, so there is no gate.
func linuxRefuseAnswer(word string) {
	tag := cTag("gollemer [terminal]>")
	switch word {
	case "rm":
		fmt.Printf("%s I don't run rm — Gollemer doesn't delete files from the chat. Deleting is permanent, and I only do things I can show you first and stand behind; if a file has to go, that's a job for your file manager or your own terminal.\n", tag)
	case "mkdir":
		fmt.Printf("%s I don't run mkdir — but making a folder inside the project is something I do: say \"create a folder <name>\" and I'll show it to you and create it behind the usual [y/n].\n", tag)
	default:
		fmt.Printf("%s I don't run %s from the chat. The basic terminal commands I do run — each behind the usual [y/n] — are: pwd, ls, cd, cat, echo, clear, date, whoami.\n", tag, word)
	}
	recordBrain("runner", false, "linux refused: "+word)
}

// linuxGate is the established run gate, the same shape as
// run-the-program's: the profile's 'do all' directive
// auto-yeses with the same announcement; otherwise the
// answer is read through the session's one ask source.
func linuxGate(ask editAsk, profile *UserProfile, cmdText string) bool {
	prompt := "Run it? " + cmdText
	if profile != nil && profile.ShouldAutoBuild() {
		fmt.Printf("%s [auto-yes: your 'do all' directive]\n", prompt)
		return true
	}
	ans, err := ask(prompt + " [y/n] ")
	if err != nil {
		return false
	}
	a := strings.ToLower(strings.TrimSpace(ans))
	return a == "y" || a == "yes"
}

// linuxGateText is the command as the gate shows it: the
// word and its arguments as typed.
func linuxGateText(cmd linuxCmd) string {
	if len(cmd.args) == 0 {
		return cmd.word
	}
	return cmd.word + " " + strings.Join(cmd.args, " ")
}

// linuxRunCmd validates one well-formed command (refusals
// are ungated), gates it, then does it natively.
func linuxRunCmd(cmd linuxCmd, ask editAsk, profile *UserProfile, interactive bool) {
	tag := cTag("gollemer [terminal]>")
	home, cur, refusal := linuxSessionDirs()
	if refusal != "" {
		fmt.Printf("%s %s\n", tag, refusal)
		recordBrain("runner", false, "linux refused: no root")
		return
	}
	// Validation first, gate second — the run family's
	// order. Each validator returns the refusal text for a
	// line that must not run; "" means go ahead.
	var target string // resolved path for ls / cat / cd
	switch cmd.word {
	case "ls", "cat":
		typed := "."
		if len(cmd.args) == 1 {
			typed = cmd.args[0]
		}
		abs, ref := linuxResolve(home, cur, typed)
		if ref != "" {
			fmt.Printf("%s %s\n", tag, ref)
			recordBrain("runner", false, "linux refused: "+ref)
			return
		}
		target = abs
		if r := linuxValidateTarget(cmd.word, typed, abs, cur); r != "" {
			fmt.Printf("%s %s\n", tag, r)
			recordBrain("runner", false, "linux refused: "+r)
			return
		}
	case "cd":
		abs, note := linuxCdTarget(home, cur, cmd.args)
		if note != "" {
			fmt.Printf("%s %s\n", tag, note)
			recordBrain("runner", true, "linux cd note: "+note)
			return
		}
		target = abs
	}
	if !linuxGate(ask, profile, linuxGateText(cmd)) {
		fmt.Printf("%s OK — not run.\n", tag)
		recordBrain("runner", true, "linux declined: "+linuxGateText(cmd))
		return
	}
	switch cmd.word {
	case "pwd":
		fmt.Println(cur)
		recordBrain("runner", true, "linux pwd")
	case "ls":
		linuxDoLs(home, target)
		recordBrain("runner", true, "linux ls")
	case "cat":
		linuxDoCat(cmd.args[0], target)
		recordBrain("runner", true, "linux cat")
	case "cd":
		linuxDoCd(home, target)
	case "echo":
		// Arguments print as typed, joined by spaces —
		// flags included. There is no flag parsing in v1,
		// and no expansion of anything.
		fmt.Println(strings.Join(cmd.args, " "))
		recordBrain("runner", true, "linux echo")
	case "clear":
		// The honest minimal thing for a line-based chat:
		// on a real terminal, the same clear sequence the
		// line reader itself uses for Ctrl-L; into a pipe,
		// nothing at all (a real clear's escapes would just
		// be bytes in a file).
		if interactive {
			fmt.Print("\x1b[H\x1b[2J")
		}
		recordBrain("runner", true, "linux clear")
	case "date":
		fmt.Println(time.Now().Format("Mon Jan _2 15:04:05 MST 2006"))
		recordBrain("runner", true, "linux date")
	case "whoami":
		if name := linuxWhoami(); name != "" {
			fmt.Println(name)
			recordBrain("runner", true, "linux whoami")
		} else {
			fmt.Printf("%s I couldn't find your user name on this machine.\n", tag)
			recordBrain("runner", false, "linux whoami: unknown")
		}
	}
}

// linuxValidateTarget checks an ls/cat target after
// resolution: existence, and the right kind of thing.
func linuxValidateTarget(word, typed, abs, cur string) string {
	st, err := os.Stat(abs)
	if err != nil {
		if word == "ls" {
			return fmt.Sprintf("I can't ls %s — there's no such file or folder here.", typed)
		}
		return fmt.Sprintf("I can't cat %s — there's no such file here.", typed)
	}
	if word == "cat" && st.IsDir() {
		return fmt.Sprintf("%s is a folder — cat reads files. Say \"ls %s\" to see what's inside it.", typed, typed)
	}
	if word == "cat" && !st.Mode().IsRegular() {
		return fmt.Sprintf("%s isn't a regular file, so I won't print it — cat here is for text files.", typed)
	}
	return ""
}

// linuxCdTarget resolves a cd line to the folder it would
// move to, or returns the note that answers it instead
// (already there, the stops-at-the-root note, an escape, a
// missing folder, a file). Bare cd targets home.
func linuxCdTarget(home, cur string, args []string) (target, note string) {
	if len(args) == 0 {
		if cur == home {
			return "", fmt.Sprintf("You're already in %s — the folder the chat started in.", linuxDisplay(home, home))
		}
		return home, ""
	}
	typed := args[0]
	cand := typed
	if !filepath.IsAbs(cand) {
		cand = filepath.Join(cur, cand)
	}
	cand = filepath.Clean(cand)
	if resolved, err := filepath.EvalSymlinks(cand); err == nil {
		cand = resolved
	}
	rel, err := filepath.Rel(home, cand)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		if linuxAboveHome(home, cand) {
			return "", fmt.Sprintf("You're at the top of the project (%s) — cd .. stops here, so the chat stays inside the project folder. Say \"ls\" to see what's here.", linuxDisplay(home, home))
		}
		if filepath.IsAbs(typed) {
			return "", fmt.Sprintf("%q isn't inside the project folder — cd only moves inside %s.", typed, linuxDisplay(home, home))
		}
		return "", fmt.Sprintf("%q climbs out of the project folder — cd only moves inside %s.", typed, linuxDisplay(home, home))
	}
	st, err := os.Stat(cand)
	if err != nil {
		return "", fmt.Sprintf("There's no folder %s here — say \"ls\" to see the folders in %s.", typed, linuxDisplay(home, cur))
	}
	if !st.IsDir() {
		return "", fmt.Sprintf("%s is a file, not a folder — cd goes into folders.", typed)
	}
	if cand == cur {
		return "", fmt.Sprintf("You're already in %s.", linuxDisplay(home, cur))
	}
	return cand, ""
}

// linuxDoCd performs a validated cd: the working folder
// moves, and the session's current project follows through
// setCurrentProject, so the analyzer, creations, edits,
// and run-the-program all aim at the new folder. A program
// the run family already started is not touched — it holds
// its own process and binary; top and stop keep working.
//
// A folder with no Go code in it cannot become the working
// folder: there is no project there for the session to
// re-aim at (the same fail-closed rule every other project
// switch in the chat keeps — analyzeForTour, handleGoAnalyze,
// ensureProject). The answer says so and names the way to
// still look at the folder's files from here.
func linuxDoCd(home, target string) {
	tag := cTag("gollemer [terminal]>")
	p, err := analyze.Analyze(target)
	if err != nil {
		if strings.Contains(err.Error(), "no Go packages") {
			rel, rerr := filepath.Rel(linuxWorkDir, target)
			typed := linuxDisplay(home, target)
			if rerr == nil {
				typed = filepath.ToSlash(rel)
			}
			fmt.Printf("%s I can't move into %s — it has no Go code in it, and I work in folders I can read as Go. You're still in %s; \"ls %s\" and \"cat\" can still look at its files from here.\n", tag, linuxDisplay(home, target), linuxDisplay(home, linuxWorkDir), typed)
			recordBrain("runner", false, "linux cd refused: no Go code in "+linuxDisplay(home, target))
			return
		}
		fmt.Printf("%s I couldn't re-read %s as the working folder (%v) — I haven't moved.\n", tag, linuxDisplay(home, target), err)
		recordBrain("runner", false, "linux cd failed: "+err.Error())
		return
	}
	linuxWorkDir = target
	setCurrentProject(target, p)
	if target == home {
		fmt.Printf("%s Now working in %s — back at the folder the chat started in. Creations, edits, and runs happen here now.\n", tag, linuxDisplay(home, target))
	} else {
		fmt.Printf("%s Now working in %s — creations, edits, and runs happen here now. Say \"cd\" on its own to go back to %s.\n", tag, linuxDisplay(home, target), linuxDisplay(home, home))
	}
	recordBrain("runner", true, "linux cd: "+linuxDisplay(home, target))
}

// linuxDoLs prints a folder's entries: names sorted, one
// per line, folders with a trailing slash. Entries whose
// names start with a dot are skipped, like a real ls.
// A file target prints its own name.
func linuxDoLs(home, target string) {
	if st, err := os.Stat(target); err == nil && !st.IsDir() {
		fmt.Println(filepath.Base(target))
		return
	}
	entries, err := os.ReadDir(target)
	if err != nil {
		fmt.Printf("%s I couldn't read %s (%v).\n", cTag("gollemer [terminal]>"), linuxDisplay(home, target), err)
		return
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") {
			continue
		}
		if e.IsDir() {
			name += "/"
		}
		names = append(names, name)
	}
	sort.Strings(names)
	if len(names) == 0 {
		fmt.Printf("%s Nothing in %s — it's an empty folder.\n", cTag("gollemer [terminal]>"), linuxDisplay(home, target))
		return
	}
	const maxShown = 200
	shown := names
	if len(shown) > maxShown {
		shown = shown[:maxShown]
	}
	for _, n := range shown {
		fmt.Println(n)
	}
	if len(names) > maxShown {
		fmt.Printf("[… %d more not shown]\n", len(names)-maxShown)
	}
}

// linuxCatCap is how much of a file cat will print: enough
// for any beginner source file, small enough to keep a
// transcript readable. A larger file is cut at the cap and
// the cut is announced.
const linuxCatCap = 32 * 1024

// linuxDoCat prints a text file's contents. Binary content
// (a NUL byte in what would be shown) is refused plainly
// instead of dumped into the chat.
func linuxDoCat(typed, abs string) {
	tag := cTag("gollemer [terminal]>")
	f, err := os.Open(abs)
	if err != nil {
		fmt.Printf("%s I couldn't open %s (%v).\n", tag, typed, err)
		return
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, linuxCatCap+1))
	if err != nil {
		fmt.Printf("%s I couldn't read %s (%v).\n", tag, typed, err)
		return
	}
	truncated := len(data) > linuxCatCap
	if truncated {
		data = data[:linuxCatCap]
	}
	if bytes.IndexByte(data, 0) >= 0 {
		fmt.Printf("%s %s doesn't look like a text file — it has binary bytes in it, so I won't print it. cat here is for text.\n", tag, typed)
		return
	}
	fmt.Print(string(data))
	if !strings.HasSuffix(string(data), "\n") {
		fmt.Println()
	}
	if truncated {
		var size int64 = -1
		if st, serr := f.Stat(); serr == nil {
			size = st.Size()
		}
		if size >= 0 {
			fmt.Printf("[… stopped there — %s is %d bytes and I show the first %d; open the file to read the rest]\n", typed, size, linuxCatCap)
		} else {
			fmt.Printf("[… stopped there — %s is longer than the %d bytes I show; open the file to read the rest]\n", typed, linuxCatCap)
		}
	}
}

// linuxWhoami resolves the OS user name the way the
// machine reports it, with the environment as a fallback.
func linuxWhoami() string {
	if u, err := user.Current(); err == nil && u.Username != "" {
		return u.Username
	}
	if s := os.Getenv("USER"); s != "" {
		return s
	}
	return os.Getenv("LOGNAME")
}
