package lib

import (
	"bytes"
	"html/template"
	"syscall/js"

	"github.com/Masterminds/sprig/v3"
)

// RegisterSprig exports a template rendering function to JS that uses Sprig functions.
func RegisterSprig() {
	renderFunc := js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) < 1 {
			return js.Global().Get("Error").New("template string required")
		}
		tmplStr := args[0].String()

		var data any
		if len(args) > 1 {
			// In a real app we'd convert JS object to Go map/struct
			// For now, let's just support basic data or nil
		}

		result, err := RenderWithSprig(tmplStr, data)
		if err != nil {
			return js.Global().Get("Error").New(err.Error())
		}
		return result
	})

	js.Global().Set("renderWithSprig", renderFunc)
}

// RenderWithSprig takes a template string and data, renders it with Sprig functions.
func RenderWithSprig(tmplStr string, data any) (string, error) {
	tmpl, err := template.New("base").Funcs(sprig.HtmlFuncMap()).Parse(tmplStr)
	if err != nil {
		return "", err
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", err
	}

	return buf.String(), nil
}
