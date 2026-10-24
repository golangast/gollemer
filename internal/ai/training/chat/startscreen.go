package chat

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/pkg/termui"
)

// startscreen.go — the unified chat's whole launch face since the
// 2026-10-24 readability pass: ONE termui panel. The name in the
// border, a three-line gopher, one line of what this is, three
// starter lines, and a status-chip row (this folder + the cast
// size — the clock chip's job). Everything the old startup used to
// dump — the phrase catalog, the teach-me pitch, profile trivia —
// now lives behind "help" or a single line here. Print less.

// startScreenText renders the start panel (pure, so tests can
// pin its shape).
func startScreenText(projectRoot string) string {
	folder := filepath.Base(projectRoot)
	if folder == "" || folder == "." {
		folder = "this folder"
	}
	var lines []string
	lines = append(lines, "  ( o o )   a small Go companion — reads your code,")
	lines = append(lines, "   \\___/    explains it in plain English")
	lines = append(lines, "")
	lines = append(lines, "  "+termui.Accent.Apply(termui.SubMark)+" start — a six-stop guided tour of this project")
	lines = append(lines, "  "+termui.Accent.Apply(termui.SubMark)+" ask in plain words — \"what does this program do\"")
	lines = append(lines, "  "+termui.Accent.Apply(termui.SubMark)+" help — everything you can say · /quit — leave")
	if teachOfferDue() {
		lines = append(lines, "  "+cHead("✦")+" new here? say "+cSay("\"teach me\"")+" — two minutes, learn by doing")
	}
	lines = append(lines, "")
	lines = append(lines, "  "+termui.Badge(folder, termui.Accent)+"  "+
		termui.Badge(fmt.Sprintf("%d brains ready", len(brainRegistry)), termui.Good))
	return termui.Panel("GOLLEMER", strings.Join(lines, "\n"), termui.Bright)
}

// printStartScreen shows the panel once at launch.
func printStartScreen(projectRoot string) {
	fmt.Println(startScreenText(projectRoot))
}
