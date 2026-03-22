package pages

import (
	"fmt"
	"syscall/js"

	"mm/wasm/lib"
	"mm/wasm/ui/material"
)

func RenderSprig() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "container")

	section := document.Call("createElement", "section")
	section.Set("style", "max-width: 800px; margin: 0 auto;")

	header := document.Call("createElement", "h1")
	header.Set("innerText", "Sprig Engine Playground")
	section.Call("appendChild", header)

	description := document.Call("createElement", "p")
	description.Set("className", "text-muted")
	description.Set("innerText", "Test your Go templates with the full power of Sprig functions directly in the browser.")
	section.Call("appendChild", description)

	// Input Card
	card := document.Call("createElement", "div")
	card.Set("className", "mat-card")
	card.Set("style", "margin-top: 3rem; padding: 2.5rem;")

	// Input area
	input := material.NewInput("Template String", "{{ \"hello world\" | title }}", "textarea")
	inputEl := input.Render()
	card.Call("appendChild", inputEl)

	// Action Row
	actionRow := document.Call("createElement", "div")
	actionRow.Set("style", "display: flex; justify-content: space-between; align-items: center; margin-top: 2rem;")

	errorEl := document.Call("createElement", "div")
	errorEl.Set("style", "color: #ef4444; font-family: monospace; font-size: 0.85rem; display: none;")
	actionRow.Call("appendChild", errorEl)

	// Output area (we need it defined before the button callback)
	resultCard := material.NewCard("Signal Output", "Awaiting process...", "")
	resultEl := resultCard.Render()

	btn := material.NewButton("Process Template", "primary", func() {
		textarea := inputEl.Call("querySelector", "textarea")
		if textarea.IsUndefined() || textarea.IsNull() {
			textarea = inputEl.Call("querySelector", "input")
		}

		tmplStr := textarea.Get("value").String()
		result, err := lib.RenderWithSprig(tmplStr, nil)

		contentEl := resultEl.Call("querySelector", ".mat-card-content")
		if contentEl.IsUndefined() || contentEl.IsNull() {
			// Fallback if component structure changed
			contentEl = resultEl
		}

		if err != nil {
			errorEl.Set("innerText", fmt.Sprintf("Syntax Error: %v", err))
			errorEl.Get("style").Set("display", "block")
			contentEl.Set("innerText", "Process Failed")
		} else {
			errorEl.Get("style").Set("display", "none")
			contentEl.Set("innerText", result)
		}
	})
	actionRow.Call("appendChild", btn.Render())
	card.Call("appendChild", actionRow)

	section.Call("appendChild", card)

	// Result Display
	resultSection := document.Call("createElement", "div")
	resultSection.Set("style", "margin-top: 2rem;")
	resultSection.Call("appendChild", resultEl)
	section.Call("appendChild", resultSection)

	// Examples
	exTitle := document.Call("createElement", "h3")
	exTitle.Set("innerText", "Quick-Start Shortcuts")
	exTitle.Set("style", "margin-top: 5rem; font-size: 1rem; text-transform: uppercase; color: var(--text-muted);")
	section.Call("appendChild", exTitle)

	examples := []struct {
		name string
		tmpl string
	}{
		{"String Pipelining", `{{ "neural-mesh" | replace "-" " " | title }}`},
		{"Temporal Formatting", `System Year: {{ now | date "2006" }}`},
		{"Functional Math", `Quantum Sum: {{ add 1024 256 }}`},
	}

	exGrid := document.Call("createElement", "div")
	exGrid.Set("className", "mat-grid")
	exGrid.Set("style", "margin-top: 1.5rem; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));")

	for _, ex := range examples {
		exCard := material.NewCard(ex.name, ex.tmpl, "")
		el := exCard.Render()
		el.Set("style", "cursor: pointer; padding: 1.5rem;")
		
		el.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			textarea := inputEl.Call("querySelector", "textarea")
			if textarea.IsUndefined() || textarea.IsNull() {
				textarea = inputEl.Call("querySelector", "input")
			}
			textarea.Set("value", ex.tmpl)
			return nil
		}))
		exGrid.Call("appendChild", el)
	}
	section.Call("appendChild", exGrid)

	container.Call("appendChild", section)
	return container
}
