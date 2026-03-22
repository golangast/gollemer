package material

import "syscall/js"

type Switch struct {
	Label    string
	Checked  bool
	OnChange func(bool)
}

func NewSwitch(label string, checked bool, onChange func(bool)) *Switch {
	return &Switch{Label: label, Checked: checked, OnChange: onChange}
}

func (s *Switch) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "label")
	container.Set("className", "mat-switch")

	input := document.Call("createElement", "input")
	input.Set("type", "checkbox")
	input.Set("checked", s.Checked)

	input.Call("addEventListener", "change", js.FuncOf(func(this js.Value, args []js.Value) any {
		s.Checked = this.Get("checked").Bool()
		if s.OnChange != nil {
			s.OnChange(s.Checked)
		}
		return nil
	}))

	slider := document.Call("createElement", "span")
	slider.Set("className", "mat-switch-slider")

	label := document.Call("createElement", "span")
	label.Set("className", "mat-switch-label")
	label.Set("innerText", s.Label)

	container.Call("appendChild", label)
	container.Call("appendChild", input)
	container.Call("appendChild", slider)

	return container
}
