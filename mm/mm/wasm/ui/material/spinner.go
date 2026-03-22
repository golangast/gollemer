package material

import "syscall/js"

type Spinner struct {
	Size string // "sm", "md", "lg"
}

func NewSpinner(size string) *Spinner {
	return &Spinner{Size: size}
}

func (s *Spinner) Render() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("className", "mat-spinner "+s.Size)

	// Create the spinning circle inside
	inner := document.Call("createElement", "div")
	inner.Set("className", "mat-spinner-inner")
	div.Call("appendChild", inner)

	return div
}
