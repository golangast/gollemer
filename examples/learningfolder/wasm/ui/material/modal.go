//go:build js && wasm


package material

import "syscall/js"

type Modal struct {
	Title     string
	Content   js.Value
	OnClose   func()
	container js.Value
}

func NewModal(title string, content js.Value, onClose func()) *Modal {
	return &Modal{Title: title, Content: content, OnClose: onClose}
}

func (m *Modal) Render() js.Value {
	document := js.Global().Get("document")

	// Overlay
	overlay := document.Call("createElement", "div")
	overlay.Set("className", "mat-modal-overlay")
	m.container = overlay

	// Modal Box
	modal := document.Call("createElement", "div")
	modal.Set("className", "mat-modal")

	// Header
	header := document.Call("createElement", "div")
	header.Set("className", "mat-modal-header")

	title := document.Call("createElement", "h3")
	title.Set("innerText", m.Title)
	header.Call("appendChild", title)

	closeBtn := document.Call("createElement", "button")
	closeBtn.Set("className", "mat-modal-close")
	closeBtn.Set("innerHTML", "&times;")
	closeBtn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		m.Close()
		return nil
	}))
	header.Call("appendChild", closeBtn)
	modal.Call("appendChild", header)

	// Body
	body := document.Call("createElement", "div")
	body.Set("className", "mat-modal-body")
	body.Call("appendChild", m.Content)
	modal.Call("appendChild", body)

	overlay.Call("appendChild", modal)

	// Append to body automatically
	document.Get("body").Call("appendChild", overlay)

	return overlay
}

func (m *Modal) Close() {
	if !m.container.IsUndefined() && !m.container.IsNull() {
		m.container.Call("remove")
	}
	if m.OnClose != nil {
		m.OnClose()
	}
}
