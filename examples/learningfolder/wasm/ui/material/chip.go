//go:build js && wasm


package material

import "syscall/js"

type Chip struct {
	Label     string
	Removable bool
	OnRemove  func()
	Color     string // "primary", "success", etc.
}

func NewChip(label string, color string) *Chip {
	return &Chip{Label: label, Color: color}
}

func (c *Chip) Render() js.Value {
	document := js.Global().Get("document")
	chip := document.Call("createElement", "div")
	chip.Set("className", "mat-chip "+c.Color)

	text := document.Call("createElement", "span")
	text.Set("innerText", c.Label)
	chip.Call("appendChild", text)

	if c.Removable {
		btn := document.Call("createElement", "span")
		btn.Set("className", "mat-chip-remove")
		btn.Set("innerHTML", "&times;")
		btn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			chip.Call("remove")
			if c.OnRemove != nil {
				c.OnRemove()
			}
			return nil
		}))
		chip.Call("appendChild", btn)
	}

	return chip
}
