package material

import "syscall/js"

type Skeleton struct {
	Type   string // "text", "circle", "rect"
	Width  string
	Height string
}

func NewSkeleton(sType, w, h string) *Skeleton {
	return &Skeleton{Type: sType, Width: w, Height: h}
}

func (s *Skeleton) Render() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("className", "mat-skeleton "+s.Type)
	div.Get("style").Call("setProperty", "width", s.Width)
	div.Get("style").Call("setProperty", "height", s.Height)
	return div
}
