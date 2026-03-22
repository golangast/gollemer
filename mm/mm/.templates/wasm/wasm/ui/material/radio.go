package material

import "syscall/js"

type RadioOption struct {
	Label string
	Value string
}

type RadioGroup struct {
	Name     string
	Options  []RadioOption
	Selected string
	OnChange func(string)
}

func NewRadioGroup(name string, options []RadioOption, onChange func(string)) *RadioGroup {
	return &RadioGroup{Name: name, Options: options, OnChange: onChange}
}

func (r *RadioGroup) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-radio-group")

	for _, opt := range r.Options {
		label := document.Call("createElement", "label")
		label.Set("className", "mat-radio-item")

		input := document.Call("createElement", "input")
		input.Set("type", "radio")
		input.Set("name", r.Name)
		input.Set("value", opt.Value)
		if opt.Value == r.Selected {
			input.Set("checked", true)
		}

		input.Call("addEventListener", "change", js.FuncOf(func(this js.Value, args []js.Value) any {
			val := this.Get("value").String()
			r.Selected = val
			if r.OnChange != nil {
				r.OnChange(val)
			}
			return nil
		}))

		indicator := document.Call("createElement", "span")
		indicator.Set("className", "mat-radio-indicator")

		text := document.Call("createElement", "span")
		text.Set("innerText", opt.Label)

		label.Call("appendChild", input)
		label.Call("appendChild", indicator)
		label.Call("appendChild", text)
		container.Call("appendChild", label)
	}

	return container
}
