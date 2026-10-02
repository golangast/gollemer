//go:build !linux

package chat

import "os"

// Raw terminal editing is Linux-only; everywhere else the LineReader
// falls back to a plain scanner.
const rawSupported = false

type keyCode int

const (
	keyEOF keyCode = iota
	keyChar
	keyEnter
	keyTab
	keyBackspace
	keyDelete
	keyLeft
	keyRight
	keyUp
	keyDown
	keyHome
	keyEnd
	keyCtrlA
	keyCtrlC
	keyCtrlD
	keyCtrlE
	keyCtrlK
	keyCtrlL
	keyCtrlU
	keyCtrlW
	keyIgnore
)

func enableRawMode(f *os.File) (func(), bool) { return nil, false }
func readKey(f *os.File) (keyCode, rune)      { return keyEOF, 0 }
func termWidth(f *os.File) int                { return 80 }
