package material

import "syscall/js"

type ListItem struct {
	Label    string
	Icon     string
	SubLabel string
	Action   js.Value
}

type List struct {
	Items []ListItem
}

func NewList(items []ListItem) *List {
	return &List{Items: items}
}

func (l *List) Render() js.Value {
	document := js.Global().Get("document")
	ul := document.Call("createElement", "ul")
	ul.Set("className", "mat-list")

	for _, item := range l.Items {
		li := document.Call("createElement", "li")
		li.Set("className", "mat-list-item")

		if item.Icon != "" {
			icon := document.Call("createElement", "span")
			icon.Set("className", "mat-list-icon")
			icon.Set("innerText", item.Icon)
			li.Call("appendChild", icon)
		}

		content := document.Call("createElement", "div")
		content.Set("className", "mat-list-content")

		label := document.Call("createElement", "span")
		label.Set("className", "mat-list-label")
		label.Set("innerText", item.Label)
		content.Call("appendChild", label)

		if item.SubLabel != "" {
			sub := document.Call("createElement", "span")
			sub.Set("className", "mat-list-sublabel")
			sub.Set("innerText", item.SubLabel)
			content.Call("appendChild", sub)
		}
		li.Call("appendChild", content)

		if !item.Action.IsUndefined() && !item.Action.IsNull() {
			li.Call("appendChild", item.Action)
		}

		ul.Call("appendChild", li)
	}

	return ul
}
