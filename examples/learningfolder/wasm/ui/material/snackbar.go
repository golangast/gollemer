//go:build js && wasm


package material

import (
	"syscall/js"
	"time"
)

type SnackBar struct {
	Message  string
	Action   string
	Duration time.Duration
}

func ShowSnackBar(message string, action string, duration time.Duration) {
	document := js.Global().Get("document")

	bar := document.Call("createElement", "div")
	bar.Set("className", "mat-snackbar")

	txt := document.Call("createElement", "span")
	txt.Set("innerText", message)
	bar.Call("appendChild", txt)

	if action != "" {
		btn := document.Call("createElement", "button")
		btn.Set("className", "mat-snackbar-action")
		btn.Set("innerText", action)
		btn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			bar.Call("remove")
			return nil
		}))
		bar.Call("appendChild", btn)
	}

	document.Get("body").Call("appendChild", bar)

	time.AfterFunc(duration, func() {
		bar.Call("remove")
	})
}
