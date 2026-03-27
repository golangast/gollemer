//go:build js && wasm


package material

import (
	"fmt"
	"syscall/js"
)

type Tooltip struct {
	Text    string
	Trigger js.Value
}

func NewTooltip(trigger js.Value, text string) *Tooltip {
	t := &Tooltip{Text: text, Trigger: trigger}

	var tip js.Value

	trigger.Call("addEventListener", "mouseenter", js.FuncOf(func(this js.Value, args []js.Value) any {
		document := js.Global().Get("document")
		tip = document.Call("createElement", "div")
		tip.Set("className", "mat-tooltip")
		tip.Set("innerText", t.Text)

		rect := trigger.Call("getBoundingClientRect")
		top := rect.Get("top").Float() - 40
		left := rect.Get("left").Float() + (rect.Get("width").Float() / 2)

		tip.Get("style").Call("setProperty", "position", "fixed")
		tip.Get("style").Call("setProperty", "top", fmt.Sprintf("%.2fpx", top))
		tip.Get("style").Call("setProperty", "left", fmt.Sprintf("%.2fpx", left))
		tip.Get("style").Call("setProperty", "transform", "translateX(-50%)")

		document.Get("body").Call("appendChild", tip)
		js.Global().Call("setTimeout", js.FuncOf(func(this js.Value, args []js.Value) any {
			tip.Get("classList").Call("add", "show")
			return nil
		}), 10)
		return nil
	}))

	trigger.Call("addEventListener", "mouseleave", js.FuncOf(func(this js.Value, args []js.Value) any {
		if !tip.IsUndefined() && !tip.IsNull() {
			tip.Call("remove")
		}
		return nil
	}))

	return t
}
