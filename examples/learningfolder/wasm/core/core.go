//go:build js && wasm


package core

import "syscall/js"

type Router struct {
	Routes map[string]func() js.Value
	Root   js.Value // The div where content is injected
}

func (r *Router) Start() {
	// Listen for hash changes in the URL
	onHashChange := js.FuncOf(func(this js.Value, args []js.Value) any {
		r.Navigate()
		return nil
	})
	js.Global().Set("onhashchange", onHashChange)

	// Initial load
	r.Navigate()
}

func (r *Router) Navigate() {
	hash := js.Global().Get("location").Get("hash").String()
	if hash == "" {
		hash = "#home"
	}

	// Clear current view
	r.Root.Set("innerHTML", "")

	// Find and render new view
	if renderFunc, ok := r.Routes[hash]; ok {
		r.Root.Call("appendChild", renderFunc())
	} else {
		// 404 Not Found
		errorDiv := js.Global().Get("document").Call("createElement", "div")
		errorDiv.Set("innerText", "404 - Page Not Found")
		r.Root.Call("appendChild", errorDiv)
	}
}
