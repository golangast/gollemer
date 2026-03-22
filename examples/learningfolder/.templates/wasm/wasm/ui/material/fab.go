package material

import "syscall/js"

type FAB struct {
	Icon    string
	Color   string
	OnClick func()
}

func NewFAB(icon, color string, onClick func()) *FAB {
	return &FAB{Icon: icon, Color: color, OnClick: onClick}
}

func (f *FAB) Render() js.Value {
	document := js.Global().Get("document")
	btn := document.Call("createElement", "button")
	btn.Set("className", "mat-fab "+f.Color)
	btn.Set("innerText", f.Icon)

	if f.OnClick != nil {
		btn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			f.OnClick()
			return nil
		}))
	}

	return btn
}
