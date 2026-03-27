//go:build js && wasm


package material

import "syscall/js"

type Card struct {
	Title    string
	Subtitle string
	Content  string
	Actions  []js.Value
}

func NewCard(title, subtitle, content string) *Card {
	return &Card{Title: title, Subtitle: subtitle, Content: content}
}

func (c *Card) Render() js.Value {
	document := js.Global().Get("document")
	card := document.Call("createElement", "div")
	card.Set("className", "mat-card")

	if c.Title != "" {
		header := document.Call("createElement", "div")
		header.Set("className", "mat-card-header")

		title := document.Call("createElement", "h3")
		title.Set("innerText", c.Title)
		header.Call("appendChild", title)

		if c.Subtitle != "" {
			sub := document.Call("createElement", "p")
			sub.Set("className", "mat-card-subtitle")
			sub.Set("innerText", c.Subtitle)
			header.Call("appendChild", sub)
		}
		card.Call("appendChild", header)
	}

	body := document.Call("createElement", "div")
	body.Set("className", "mat-card-content")
	body.Set("innerHTML", c.Content)
	card.Call("appendChild", body)

	if len(c.Actions) > 0 {
		footer := document.Call("createElement", "div")
		footer.Set("className", "mat-card-actions")
		for _, action := range c.Actions {
			footer.Call("appendChild", action)
		}
		card.Call("appendChild", footer)
	}

	return card
}
