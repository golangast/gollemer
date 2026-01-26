package pages

import (
	"syscall/js"

	"github.com/golangast/gollemer/wasm/ui/material"
)

func RenderNewpage() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "newpage-page")
	container.Set("style", "padding: 4rem 2rem; max-width: 800px; margin: 0 auto; min-height: 80vh;")

	heading := document.Call("createElement", "h1")
	heading.Set("innerText", "NewPage")
	heading.Set("className", "mat-h1")
	container.Call("appendChild", heading)

	sub := document.Call("createElement", "p")
	sub.Set("innerText", "This is the NewPage page of our application.")
	sub.Set("className", "mat-body-1")
	container.Call("appendChild", sub)

	return container
}
