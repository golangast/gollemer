package pages

import (
	"fmt"
	"syscall/js"

	"github.com/golangast/gollemer/learningfolder/wasm/lib"
	"github.com/golangast/gollemer/learningfolder/wasm/ui/material"
)

func RenderSprig() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("style", "padding: 2rem; max-width: 800px; margin: 0 auto;")

	header := document.Call("createElement", "h1")
	header.Set("innerText", "Sprig Functions Playground")
	container.Call("appendChild", header)

	description := document.Call("createElement", "p")
	description.Set("innerText", "Enter a Go template string using Sprig functions (e.g., {{ \"hello\" | upper }}) to see the result.")
	container.Call("appendChild", description)

	// Input area
	input := material.NewInput("Template", "{{ \"hello\" | upper }} {{ now | date \"2006\" }}", "textarea")
	inputEl := input.Render()
	container.Call("appendChild", inputEl)

	// Output area
	outputCard := material.NewCard("Result", "Output will appear here", "")
	outputEl := outputCard.Render()
	container.Call("appendChild", outputEl)

	// Error area
	errorEl := document.Call("createElement", "div")
	errorEl.Set("style", "color: #ff5252; margin-top: 1rem; font-family: monospace; display: none;")
	container.Call("appendChild", errorEl)

	// Render button
	btn := material.NewButton("Render Template", "primary", func() {
		// We'd need to find the textarea value.
		// Since material.Input doesn't expose the element easily, we might need a better way or just find it in DOM
		textarea := inputEl.Call("querySelector", "textarea")
		if textarea.IsUndefined() || textarea.IsNull() {
			// Fallback for different input implementation
			textarea = inputEl.Call("querySelector", "input")
		}

		tmplStr := textarea.Get("value").String()
		result, err := lib.RenderWithSprig(tmplStr, nil)

		if err != nil {
			errorEl.Set("innerText", fmt.Sprintf("Error: %v", err))
			errorEl.Get("style").Set("display", "block")
			outputEl.Call("querySelector", ".mat-card-content").Set("innerText", "")
		} else {
			errorEl.Get("style").Set("display", "none")
			outputEl.Call("querySelector", ".mat-card-content").Set("innerText", result)
		}
	})
	container.Call("appendChild", btn.Render())

	// Examples
	examplesHeader := document.Call("createElement", "h3")
	examplesHeader.Set("innerText", "Examples")
	examplesHeader.Set("style", "margin-top: 3rem;")
	container.Call("appendChild", examplesHeader)

	examples := []struct {
		name string
		tmpl string
	}{
		{"String manipulation", `{{ "hello world" | title }} -> {{ "hello world" | upper }}`},
		{"Date matching", `Current year: {{ now | date "2006" }}`},
		{"List functions", `{{ range $idx, $val := splitList "," "a,b,c" }}Item {{ $idx }}: {{ $val }} {{ end }}`},
		{"Math", `10 + 5 = {{ add 10 5 }}`},
	}

	for _, ex := range examples {
		exCard := material.NewCard(ex.name, ex.tmpl, "")
		exEl := exCard.Render()
		exEl.Set("style", exEl.Get("style").String()+"cursor: pointer; margin-bottom: 0.5rem;")

		// Copy to input on click
		exEl.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			textarea := inputEl.Call("querySelector", "textarea")
			if textarea.IsUndefined() || textarea.IsNull() {
				textarea = inputEl.Call("querySelector", "input")
			}
			textarea.Set("value", ex.tmpl)
			return nil
		}))

		container.Call("appendChild", exEl)
	}

	return container
}
