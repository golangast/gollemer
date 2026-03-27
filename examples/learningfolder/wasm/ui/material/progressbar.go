//go:build js && wasm


package material

import (
	"fmt"
	"syscall/js"
)

type ProgressBar struct {
	Value float64 // 0 to 100
	Mode  string  // "determinate", "indeterminate"
}

func NewProgressBar(value float64, mode string) *ProgressBar {
	return &ProgressBar{Value: value, Mode: mode}
}

func (p *ProgressBar) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-progress-bar "+p.Mode)

	fill := document.Call("createElement", "div")
	fill.Set("className", "mat-progress-bar-fill")

	if p.Mode == "determinate" {
		fill.Get("style").Call("setProperty", "width", fmt.Sprintf("%.2f%%", p.Value))
	}

	container.Call("appendChild", fill)
	return container
}
