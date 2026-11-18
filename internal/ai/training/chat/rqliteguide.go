package chat

// Running rqlite: the guidance answer (2026-10-31). The
// concept book's rqlite rows write code that talks to a
// rqlite node over HTTP — but the node itself is a server
// program (rqlited) the user installs and starts, and a
// beginner's first rqlite question is exactly that: "how do
// I run rqlite". This claim answers it deterministically,
// beside the package lookup: the shape is tight (a how/what
// question whose subject is rqlite and whose verb is
// install/run/start/use/get/set up), so code asks and rqlite
// ROW asks ("add a rqlite query to …") keep their routes.
//
// The answer is printed guidance only — commands for the
// user to run himself, in his own terminal. The chat never
// installs, downloads, or starts anything: no shell ever,
// and this claim is no exception. The commands and facts
// are rqlite's own documented quick start (rqlite.io/docs/
// quick-start, read at build time): the Linux install
// script, the pre-built releases, the Docker image, the
// Homebrew package, and `rqlited -node-id=1 data/` — the
// folder named there is the node's storage, and the
// database file lives inside it, owned by the node. A
// program never creates that file; it creates tables.

import (
	"strings"
)

// parseRqliteGuide reads one line as a "how do I run
// rqlite" question. True only for the tight shapes in the
// header — anything else (above all the rqlite book rows'
// own sentences) is not this claim's.
func parseRqliteGuide(line string) bool {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSuffix(s, ".")
	low := strings.ToLower(strings.TrimSpace(s))

	// "what is rqlite" — the explanation opener, answered by
	// the same text (it starts by saying what rqlite is).
	for _, p := range []string{"what is rqlite", "what's rqlite", "whats rqlite"} {
		if low == p {
			return true
		}
	}

	rest := ""
	for _, p := range []string{"how do i ", "how can i ", "how do you ", "how to "} {
		if strings.HasPrefix(low, p) {
			rest = low[len(p):]
			break
		}
	}
	if rest == "" {
		// The bare imperative: "install rqlite".
		if strings.HasPrefix(low, "install rqlite") {
			rest = low
		} else {
			return false
		}
	}
	verb := false
	for _, v := range []string{"install", "run", "start", "use", "get", "set up", "setup", "download"} {
		if rest == v || strings.HasPrefix(rest, v+" ") {
			rest = strings.TrimSpace(strings.TrimPrefix(rest, v))
			verb = true
			break
		}
	}
	if !verb {
		return false
	}
	// What remains must be about rqlite itself, briefly:
	// "rqlite", "rqlite running", "rqlite on windows".
	fields := strings.Fields(rest)
	if len(fields) == 0 || fields[0] != "rqlite" || len(fields) > 4 {
		return false
	}
	return true
}

// rqliteGuideText is the answer, in the chat's plain-words
// voice. Every command in it is the user's to run, in his
// own terminal — the text says so.
const rqliteGuideText = `rqlite is not a library you add to your program — it's a database server (the program is called rqlited) that your program talks to over HTTP. Installing and starting it is something you do yourself, in your own terminal — I can't install or run programs for you. From rqlite's own quick start, the ways in are:

  - Linux, the easy way:  curl -fsSL https://rqlite.io/install.sh | sh
  - Any system: download a pre-built release from https://github.com/rqlite/rqlite/releases and unpack it
  - Docker:  docker run -p 4001:4001 rqlite/rqlite
  - macOS:  brew install rqlite

Then start a single node with:

  rqlited -node-id=1 data/

The folder you name (data/ there) is the node's storage — the database file lives inside it, owned by the node. Your program never creates that file; it creates tables and rows through the API instead ("add a rqlite create table to ..." and its family). Once the node is running it listens at http://localhost:4001 — the address the rqlite rows in my concept book talk to.`

// tryRqliteGuide claims a "how do I run rqlite" question
// and prints the guidance answer.
func tryRqliteGuide(line string) bool {
	if !parseRqliteGuide(line) {
		return false
	}
	printPaced(rqliteGuideText + "\n")
	recordBrain("analyst", true, "rqlite guide")
	return true
}
