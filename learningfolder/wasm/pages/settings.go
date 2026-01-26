package pages

import "syscall/js"

func RenderSettings() js.Value {
	div := js.Global().Get("document").Call("createElement", "div")
	div.Set("innerHTML", "<h1>Settings</h1><p>Manage your preferences here.</p>")
	return div
}
