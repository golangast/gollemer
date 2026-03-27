//go:build js && wasm


package material

import "syscall/js"

type AccordionItem struct {
	Title   string
	Content js.Value
}

type Accordion struct {
	Items []AccordionItem
}

func NewAccordion(items []AccordionItem) *Accordion {
	return &Accordion{Items: items}
}

func (a *Accordion) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-accordion")

	for _, item := range a.Items {
		section := document.Call("createElement", "div")
		section.Set("className", "mat-accordion-item")

		header := document.Call("createElement", "div")
		header.Set("className", "mat-accordion-header")
		header.Set("innerText", item.Title)

		body := document.Call("createElement", "div")
		body.Set("className", "mat-accordion-body")
		body.Call("appendChild", item.Content)

		header.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			isOpen := body.Get("classList").Call("contains", "open").Bool()
			if isOpen {
				body.Get("classList").Call("remove", "open")
				header.Get("classList").Call("remove", "active")
			} else {
				body.Get("classList").Call("add", "open")
				header.Get("classList").Call("add", "active")
			}
			return nil
		}))

		section.Call("appendChild", header)
		section.Call("appendChild", body)
		container.Call("appendChild", section)
	}

	return container
}
