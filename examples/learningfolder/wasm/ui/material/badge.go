//go:build js && wasm


package material

import "syscall/js"

type Badge struct {
	Text  string
	Color string // "primary", "success", "warning", "danger"
}

func NewBadge(text, color string) *Badge {
	return &Badge{Text: text, Color: color}
}

func (b *Badge) Render() js.Value {
	document := js.Global().Get("document")
	span := document.Call("createElement", "span")
	span.Set("className", "mat-badge "+b.Color)
	span.Set("innerText", b.Text)
	return span
}
