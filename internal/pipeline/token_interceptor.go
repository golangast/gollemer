// Package pipeline provides the streaming token interceptor that separates
// hardware control tokens from natural-language prose in the model's output.
package pipeline

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/hardware"
)

// TokenInterceptor reads from a token stream channel sequentially.
// The first three tokens are expected to be hardware control tokens
// (<INTENT_*>, <ACT_*>, <DEV_*>).  Once all three are collected it fires
// the hardware handler asynchronously, then passes the remaining prose tokens
// to the speaker so TTS begins without waiting for the hardware call.
type TokenInterceptor struct {
	// OnProse is called with each regular prose token, in order.
	// Default behaviour is to print it to stdout.
	OnProse func(token string)
}

// NewTokenInterceptor returns an interceptor with a default stdout prose handler.
func NewTokenInterceptor() *TokenInterceptor {
	return &TokenInterceptor{
		OnProse: func(token string) {
			fmt.Printf("%s ", token)
		},
	}
}

// Process consumes the tokenStream channel until it is closed.
// It collects the leading <INTENT_*> / <ACT_*> / <DEV_*> tokens and, as
// soon as all three are known, spawns a goroutine to execute the mapped
// hardware routine while continuing to forward prose tokens to OnProse.
func (ti *TokenInterceptor) Process(tokenStream <-chan string) {
	var currentIntent string
	var currentAction string
	var currentDevice string
	hardwareFired := false

	fmt.Print("Gollemer: ")

	for token := range tokenStream {
		switch {
		case strings.HasPrefix(token, "<INTENT_"):
			currentIntent = token

		case strings.HasPrefix(token, "<ACT_"):
			currentAction = token

		case strings.HasPrefix(token, "<DEV_"):
			currentDevice = token
			// All three control tokens are gathered — fire hardware async.
			if !hardwareFired && currentIntent != "" && currentAction != "" && currentDevice != "" {
				hardwareFired = true
				go ti.executeHardwareTrigger(currentIntent, currentAction, currentDevice)
			}

		default:
			// Regular prose — forward to the speaker / TTS layer.
			if ti.OnProse != nil {
				ti.OnProse(token)
			}
		}
	}
	fmt.Println() // newline after the full response
}

// executeHardwareTrigger maps the bracket tokens back to an IntentPayload and
// calls the hardware controller asynchronously.
func (ti *TokenInterceptor) executeHardwareTrigger(intentTok, actionTok, deviceTok string) {
	// Strip angle brackets: <INTENT_MICROPHONE_MUTE> → MICROPHONE_MUTE
	intent := strings.TrimSuffix(strings.TrimPrefix(intentTok, "<INTENT_"), ">")
	action := strings.TrimSuffix(strings.TrimPrefix(actionTok, "<ACT_"), ">")
	device := strings.TrimSuffix(strings.TrimPrefix(deviceTok, "<DEV_"), ">")

	fmt.Printf("\n[ASYNC HARDWARE TRIGGER] Driving %s via %s based on %s\n", device, action, intentTok)

	payload := hardware.IntentPayload{
		Intent: intent,
		Roles: map[string]string{
			"action": strings.ToLower(action),
			"device": strings.ToLower(device),
		},
	}
	if err := hardware.HandleIntent(payload); err != nil {
		fmt.Printf("[HARDWARE ERROR] %s → %v\n", intent, err)
	}
}

// TokenStreamFromString converts a string (model output) into a token channel
// so existing code paths can be tested without a live model.
func TokenStreamFromString(output string) <-chan string {
	ch := make(chan string, 64)
	go func() {
		defer close(ch)
		for _, tok := range strings.Fields(output) {
			ch <- tok
		}
	}()
	return ch
}

// LoadMultitaskDataset parses the ### INPUT / ### TARGET plaintext format
// and returns (input, target) pairs for training.
func LoadMultitaskDataset(path string, content string) [][2]string {
	var pairs [][2]string
	blocks := strings.Split(content, "\n\n")
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		inputMarker := "### INPUT\n"
		targetMarker := "### TARGET\n"
		iIdx := strings.Index(block, inputMarker)
		tIdx := strings.Index(block, targetMarker)
		if iIdx == -1 || tIdx == -1 {
			continue
		}
		inputStart := iIdx + len(inputMarker)
		inputEnd := tIdx
		targetStart := tIdx + len(targetMarker)

		input := strings.TrimSpace(block[inputStart:inputEnd])
		target := strings.TrimSpace(block[targetStart:])
		if input != "" && target != "" {
			pairs = append(pairs, [2]string{input, target})
		}
	}
	return pairs
}
