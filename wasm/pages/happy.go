package pages

import (
	"syscall/js"

	"github.com/golangast/gollemer/wasm/ui/material"
)

func RenderHappy() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "happy-page")
	container.Set("style", "padding: 4rem 2rem; max-width: 800px; margin: 0 auto; min-height: 80vh;")

	heading := document.Call("createElement", "h1")
	heading.Set("innerText", "happy")
	heading.Set("className", "mat-h1")
	container.Call("appendChild", heading)

	sub := document.Call("createElement", "p")
	sub.Set("innerText", "Welcome to the happy page.")
	sub.Set("className", "mat-body-1")
	container.Call("appendChild", sub)

	// Add a button to use the material package and avoid 'unused import' error
	btn := material.NewButton("Action", "primary", func() {
		js.Global().Call("alert", "Action triggered on %!s(MISSING) page!")
	})
	container.Call("appendChild", btn.Render())

	return container
}
