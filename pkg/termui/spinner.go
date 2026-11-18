package termui

import (
	"fmt"
	"os"
	"sync"
	"time"
)

// spinner.go — a one-line loading indicator for the waits that are
// real (model loading, project analysis). On a terminal it cycles
// braille frames in place via \r; off a terminal it prints exactly
// one plain line per phase — never \r spam into pipes or logs. It
// is a printed line in a scrolling transcript, not a screen mode.

var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// Spinner is a running indicator; Stop ends it.
type Spinner struct {
	done chan struct{}
	once sync.Once
	wg   sync.WaitGroup
}

// Spin starts an indicator with the given label and returns its
// handle. Styling off (pipe, NO_COLOR): one plain "label" line is
// printed immediately and Stop is a no-op beyond that.
func Spin(label string) *Spinner {
	s := &Spinner{done: make(chan struct{})}
	if !Enabled {
		fmt.Fprintln(os.Stdout, label)
		return s
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		i := 0
		for {
			select {
			case <-s.done:
				return
			default:
			}
			fmt.Fprintf(os.Stdout, "\r%s %s", Accent.Apply(spinnerFrames[i%len(spinnerFrames)]), label)
			i++
			select {
			case <-s.done:
				return
			case <-time.After(120 * time.Millisecond):
			}
		}
	}()
	return s
}

// Stop clears the spinner line and, when final is non-empty,
// prints it as the phase's finished line ("Ready."). Off a
// terminal it prints final as a plain line when given.
func (s *Spinner) Stop(final string) {
	s.once.Do(func() { close(s.done) })
	s.wg.Wait()
	if !Enabled {
		if final != "" {
			fmt.Fprintln(os.Stdout, final)
		}
		return
	}
	fmt.Fprint(os.Stdout, "\r\x1b[K")
	if final != "" {
		fmt.Fprintln(os.Stdout, final)
	}
}
