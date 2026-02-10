package material

import (
	"fmt"
	"syscall/js"
)

type Slider struct {
	Label    string
	Min      float64
	Max      float64
	Step     float64
	Value    float64
	OnChange func(float64)
}

func NewSlider(label string, min, max, step, value float64, onChange func(float64)) *Slider {
	return &Slider{Label: label, Min: min, Max: max, Step: step, Value: value, OnChange: onChange}
}

func (s *Slider) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-slider-container")

	valDisplay := document.Call("createElement", "span")
	valDisplay.Set("className", "mat-slider-value")
	valDisplay.Set("innerText", fmt.Sprintf("%.1f", s.Value))

	if s.Label != "" {
		label := document.Call("createElement", "label")
		label.Set("innerText", s.Label+": ")
		label.Call("appendChild", valDisplay)
		container.Call("appendChild", label)
	}

	input := document.Call("createElement", "input")
	input.Set("type", "range")
	input.Set("min", s.Min)
	input.Set("max", s.Max)
	input.Set("step", s.Step)
	input.Set("value", s.Value)
	input.Set("className", "mat-slider")

	input.Call("addEventListener", "input", js.FuncOf(func(this js.Value, args []js.Value) any {
		val := this.Get("valueAsNumber").Float()
		s.Value = val
		valDisplay.Set("innerText", fmt.Sprintf("%.1f", val))
		if s.OnChange != nil {
			s.OnChange(val)
		}
		return nil
	}))

	container.Call("appendChild", input)
	return container
}
