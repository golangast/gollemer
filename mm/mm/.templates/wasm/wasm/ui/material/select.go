package material

import "syscall/js"

type SelectOption struct {
	Label string
	Value string
}

type Select struct {
	Label    string
	Options  []SelectOption
	Selected string
	OnChange func(string)
}

func NewSelect(label string, options []SelectOption, onChange func(string)) *Select {
	return &Select{Label: label, Options: options, OnChange: onChange}
}

func (s *Select) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-select-container")

	if s.Label != "" {
		label := document.Call("createElement", "label")
		label.Set("innerText", s.Label)
		container.Call("appendChild", label)
	}

	selectEl := document.Call("createElement", "select")
	selectEl.Set("className", "mat-select")

	for _, opt := range s.Options {
		option := document.Call("createElement", "option")
		option.Set("value", opt.Value)
		option.Set("innerText", opt.Label)
		if opt.Value == s.Selected {
			option.Set("selected", true)
		}
		selectEl.Call("appendChild", option)
	}

	selectEl.Call("addEventListener", "change", js.FuncOf(func(this js.Value, args []js.Value) any {
		val := this.Get("value").String()
		s.Selected = val
		if s.OnChange != nil {
			s.OnChange(val)
		}
		return nil
	}))

	container.Call("appendChild", selectEl)
	return container
}
