package ui

import (
	"fmt"
	"os"
	"time"
)

// Spin runs a loading animation in a goroutine
func (m *Mascot) Spin(frames []string, label string, stop chan bool) {
	i := 0
	for {
		select {
		case <-stop:
			fmt.Print("\r\033[K") // Clear the line when finished
			os.Stdout.Sync()
			return
		default:
			// \r brings us back to the start of the line for the next frame
			fmt.Printf("\r%s/ʕ%sʔ/ > %s...%s", m.Color, frames[i%len(frames)], label, ColorReset)
			os.Stdout.Sync()
			i++
			time.Sleep(150 * time.Millisecond)
		}
	}
}
