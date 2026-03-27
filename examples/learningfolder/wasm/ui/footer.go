//go:build js && wasm


package ui

import "syscall/js"

type Footer struct {
	Copyright string
}

func NewFooter(copyright string) *Footer {
	return &Footer{Copyright: copyright}
}

func (f *Footer) Render() js.Value {
	document := js.Global().Get("document")
	footer := document.Call("createElement", "footer")
	footer.Set("className", "mat-footer")

	wrapper := document.Call("createElement", "div")
	wrapper.Set("className", "mat-footer-wrapper")

	// Links column 1
	col1 := document.Call("createElement", "div")
	col1.Set("className", "mat-footer-col")
	col1.Set("innerHTML", "<h4>Platform</h4><a href='#home'>Home</a><a href='#components'>Components</a><a href='#help'>Help Center</a>")
	wrapper.Call("appendChild", col1)

	// Links column 2
	col2 := document.Call("createElement", "div")
	col2.Set("className", "mat-footer-col")
	col2.Set("innerHTML", "<h4>Resources</h4><a href='#'>Docs</a><a href='#'>Github</a>")
	wrapper.Call("appendChild", col2)

	// Copyright bar
	bottom := document.Call("createElement", "div")
	bottom.Set("className", "mat-footer-bottom")
	bottom.Set("innerText", f.Copyright)

	footer.Call("appendChild", wrapper)
	footer.Call("appendChild", bottom)
	return footer
}
