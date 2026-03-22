package material

import "syscall/js"

type Button struct {
	Text    string
	Variant string // "primary", "secondary", "outline", "ghost"
	OnClick func()
}

func NewButton(text string, variant string, onClick func()) *Button {
	return &Button{Text: text, Variant: variant, OnClick: onClick}
}

func (b *Button) Render() js.Value {
	document := js.Global().Get("document")
	btn := document.Call("createElement", "button")
	btn.Set("className", "mat-button "+b.Variant)
	btn.Set("innerText", b.Text)

	if b.OnClick != nil {
		btn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			b.OnClick()
			return nil
		}))
	}

	return btn
}
