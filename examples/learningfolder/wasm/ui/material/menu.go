package material

import (
	"fmt"
	"syscall/js"
)

type MenuItem struct {
	Label   string
	Icon    string
	OnClick func()
}

type Menu struct {
	Trigger js.Value
	Items   []MenuItem
	el      js.Value
	isOpen  bool
}

func NewMenu(trigger js.Value, items []MenuItem) *Menu {
	m := &Menu{Trigger: trigger, Items: items}
	trigger.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		m.Toggle()
		return nil
	}))
	return m
}

func (m *Menu) Render() js.Value {
	document := js.Global().Get("document")
	menu := document.Call("createElement", "div")
	menu.Set("className", "mat-menu")
	m.el = menu

	for _, item := range m.Items {
		option := document.Call("createElement", "div")
		option.Set("className", "mat-menu-item")
		option.Set("innerText", item.Label)

		option.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			if item.OnClick != nil {
				item.OnClick()
			}
			m.Close()
			return nil
		}))
		menu.Call("appendChild", option)
	}

	return menu
}

func (m *Menu) Toggle() {
	if m.isOpen {
		m.Close()
	} else {
		m.Open()
	}
}

func (m *Menu) Open() {
	m.isOpen = true
	document := js.Global().Get("document")

	// Re-render menu
	menu := m.Render()

	// Position relative to trigger
	rect := m.Trigger.Call("getBoundingClientRect")
	top := rect.Get("bottom").Float()
	left := rect.Get("left").Float()

	menu.Get("style").Call("setProperty", "position", "fixed")
	menu.Get("style").Call("setProperty", "top", fmt.Sprintf("%.2fpx", top))
	menu.Get("style").Call("setProperty", "left", fmt.Sprintf("%.2fpx", left))

	document.Get("body").Call("appendChild", menu)
}

func (m *Menu) Close() {
	m.isOpen = false
	if !m.el.IsUndefined() && !m.el.IsNull() {
		m.el.Call("remove")
	}
}
