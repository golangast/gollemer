//go:build js && wasm


package material

import "syscall/js"

type Checkbox struct {
	Label    string
	Checked  bool
	OnChange func(bool)
}

func NewCheckbox(label string, checked bool, onChange func(bool)) *Checkbox {
	return &Checkbox{Label: label, Checked: checked, OnChange: onChange}
}

func (c *Checkbox) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "label")
	container.Set("className", "mat-checkbox-container")

	input := document.Call("createElement", "input")
	input.Set("type", "checkbox")
	input.Set("checked", c.Checked)

	input.Call("addEventListener", "change", js.FuncOf(func(this js.Value, args []js.Value) any {
		c.Checked = this.Get("checked").Bool()
		if c.OnChange != nil {
			c.OnChange(c.Checked)
		}
		return nil
	}))

	checkmark := document.Call("createElement", "span")
	checkmark.Set("className", "mat-checkbox-checkmark")

	label := document.Call("createElement", "span")
	label.Set("innerText", c.Label)

	container.Call("appendChild", input)
	container.Call("appendChild", checkmark)
	container.Call("appendChild", label)

	return container
}
