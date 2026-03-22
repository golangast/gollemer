package material

import "syscall/js"

type Alert struct {
	Message string
	Type    string // "info", "success", "warning", "danger"
	Dismiss bool
}

func NewAlert(message, alertType string, dismiss bool) *Alert {
	return &Alert{Message: message, Type: alertType, Dismiss: dismiss}
}

func (a *Alert) Render() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("className", "mat-alert "+a.Type)

	text := document.Call("createElement", "span")
	text.Set("innerText", a.Message)
	div.Call("appendChild", text)

	if a.Dismiss {
		closeBtn := document.Call("createElement", "button")
		closeBtn.Set("className", "mat-alert-close")
		closeBtn.Set("innerHTML", "&times;")
		closeBtn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			div.Call("remove")
			return nil
		}))
		div.Call("appendChild", closeBtn)
	}

	return div
}
