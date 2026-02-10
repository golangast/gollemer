package material

import "syscall/js"

type Drawer struct {
	Content js.Value
	isOpen  bool
	el      js.Value
	overlay js.Value
}

func NewDrawer(content js.Value) *Drawer {
	return &Drawer{Content: content}
}

func (d *Drawer) Render() js.Value {
	document := js.Global().Get("document")

	overlay := document.Call("createElement", "div")
	overlay.Set("className", "mat-drawer-overlay")
	d.overlay = overlay

	drawer := document.Call("createElement", "div")
	drawer.Set("className", "mat-drawer")
	d.el = drawer

	drawer.Call("appendChild", d.Content)
	overlay.Call("appendChild", drawer)

	overlay.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		if args[0].Get("target").Equal(overlay) {
			d.Close()
		}
		return nil
	}))

	return overlay
}

func (d *Drawer) Open() {
	document := js.Global().Get("document")
	document.Get("body").Call("appendChild", d.Render())
	// Delay to trigger animation
	js.Global().Call("setTimeout", js.FuncOf(func(this js.Value, args []js.Value) any {
		d.el.Get("classList").Call("add", "open")
		d.overlay.Get("classList").Call("add", "open")
		return nil
	}), 10)
}

func (d *Drawer) Close() {
	d.el.Get("classList").Call("remove", "open")
	d.overlay.Get("classList").Call("remove", "open")
	js.Global().Call("setTimeout", js.FuncOf(func(this js.Value, args []js.Value) any {
		d.overlay.Call("remove")
		return nil
	}), 300)
}
