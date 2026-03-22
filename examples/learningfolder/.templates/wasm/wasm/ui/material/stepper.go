package material

import (
	"strconv"
	"syscall/js"
)

type StepperStep struct {
	Label   string
	Content js.Value
}

type Stepper struct {
	Steps         []StepperStep
	SelectedIndex int
	contentArea   js.Value
}

func NewStepper(steps []StepperStep) *Stepper {
	return &Stepper{Steps: steps, SelectedIndex: 0}
}

func (s *Stepper) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-stepper")

	header := document.Call("createElement", "div")
	header.Set("className", "mat-stepper-header")

	s.contentArea = document.Call("createElement", "div")
	s.contentArea.Set("className", "mat-stepper-content")

	for i, step := range s.Steps {
		idx := i
		stepItem := document.Call("createElement", "div")
		stepItem.Set("className", "mat-step-item")
		if i == s.SelectedIndex {
			stepItem.Get("classList").Call("add", "active")
		}

		circle := document.Call("createElement", "div")
		circle.Set("className", "mat-step-circle")
		circle.Set("innerText", strconv.Itoa(i+1))

		label := document.Call("createElement", "span")
		label.Set("className", "mat-step-label")
		label.Set("innerText", step.Label)

		stepItem.Call("appendChild", circle)
		stepItem.Call("appendChild", label)

		stepItem.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			s.Select(idx, header)
			return nil
		}))

		header.Call("appendChild", stepItem)
		if i < len(s.Steps)-1 {
			line := document.Call("createElement", "div")
			line.Set("className", "mat-step-line")
			header.Call("appendChild", line)
		}
	}

	container.Call("appendChild", header)
	container.Call("appendChild", s.contentArea)
	s.updateContent()

	return container
}

func (s *Stepper) Select(index int, header js.Value) {
	s.SelectedIndex = index
	items := header.Call("querySelectorAll", ".mat-step-item")
	for i := 0; i < items.Get("length").Int(); i++ {
		items.Call("item", i).Get("classList").Call("remove", "active")
	}
	items.Call("item", index).Get("classList").Call("add", "active")
	s.updateContent()
}

func (s *Stepper) updateContent() {
	if !s.contentArea.IsUndefined() && !s.contentArea.IsNull() {
		s.contentArea.Set("innerHTML", "")
		s.contentArea.Call("appendChild", s.Steps[s.SelectedIndex].Content)
	}
}
