package material

import "syscall/js"

type BottomSheet struct {
	Content   js.Value
	container js.Value
}

func NewBottomSheet(content js.Value) *BottomSheet {
	return &BottomSheet{Content: content}
}

func (b *BottomSheet) Render() js.Value {
	document := js.Global().Get("document")
	overlay := document.Call("createElement", "div")
	overlay.Set("className", "mat-bottom-sheet-overlay")
	b.container = overlay

	sheet := document.Call("createElement", "div")
	sheet.Set("className", "mat-bottom-sheet")
	sheet.Call("appendChild", b.Content)

	overlay.Call("appendChild", sheet)
	document.Get("body").Call("appendChild", overlay)

	overlay.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		if args[0].Get("target").Equal(overlay) {
			b.Close()
		}
		return nil
	}))

	js.Global().Call("setTimeout", js.FuncOf(func(this js.Value, args []js.Value) any {
		sheet.Get("classList").Call("add", "open")
		overlay.Get("classList").Call("add", "open")
		return nil
	}), 10)

	return overlay
}

func (b *BottomSheet) Close() {
	sheet := b.container.Call("querySelector", ".mat-bottom-sheet")
	sheet.Get("classList").Call("remove", "open")
	b.container.Get("classList").Call("remove", "open")
	js.Global().Call("setTimeout", js.FuncOf(func(this js.Value, args []js.Value) any {
		b.container.Call("remove")
		return nil
	}), 300)
}
