//go:build !linux

package termui

// terminalWidth: no portable stdlib way to ask; 0 means unknown
// and TerminalWidth falls back to its default.
func terminalWidth() int { return 0 }
