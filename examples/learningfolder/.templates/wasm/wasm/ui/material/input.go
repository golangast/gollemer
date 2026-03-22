package material

import "syscall/js"

type Input struct {
	Label       string
	Placeholder string
	Value       string
	Type        string // "text", "password", "email", etc.
	OnChange    func(string)
}

func NewInput(label, placeholder string, inputType string) *Input {
	return &Input{Label: label, Placeholder: placeholder, Type: inputType}
}

func (i *Input) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-input-container")

	if i.Label != "" {
		label := document.Call("createElement", "label")
		label.Set("innerText", i.Label)
		container.Call("appendChild", label)
	}

	input := document.Call("createElement", "input")
	input.Set("type", i.Type)
	input.Set("placeholder", i.Placeholder)
	input.Set("className", "mat-input")
	input.Set("value", i.Value)

	input.Call("addEventListener", "input", js.FuncOf(func(this js.Value, args []js.Value) any {
		val := this.Get("value").String()
		i.Value = val
		if i.OnChange != nil {
			i.OnChange(val)
		}
		return nil
	}))

	container.Call("appendChild", input)
	return container
}
