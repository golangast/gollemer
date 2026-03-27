//go:build js && wasm


package material

import "syscall/js"

type Breadcrumb struct {
	Label string
	Link  string
}

type Breadcrumbs struct {
	Items []Breadcrumb
}

func NewBreadcrumbs(items []Breadcrumb) *Breadcrumbs {
	return &Breadcrumbs{Items: items}
}

func (b *Breadcrumbs) Render() js.Value {
	document := js.Global().Get("document")
	nav := document.Call("createElement", "nav")
	nav.Set("className", "mat-breadcrumbs")

	for i, item := range b.Items {
		if i > 0 {
			separator := document.Call("createElement", "span")
			separator.Set("className", "mat-breadcrumb-separator")
			separator.Set("innerText", "/")
			nav.Call("appendChild", separator)
		}

		span := document.Call("createElement", "a")
		span.Set("innerText", item.Label)
		span.Set("href", item.Link)
		span.Set("className", "mat-breadcrumb-item")
		if i == len(b.Items)-1 {
			span.Get("classList").Call("add", "active")
		}
		nav.Call("appendChild", span)
	}

	return nav
}
