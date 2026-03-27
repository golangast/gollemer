//go:build js && wasm


package ui

import "syscall/js"

type Header struct {
	Brand string
}

func NewHeader(brand string) *Header {
	return &Header{Brand: brand}
}

func (h *Header) Render() js.Value {
	document := js.Global().Get("document")
	header := document.Call("createElement", "header")
	header.Set("className", "mat-header")

	wrapper := document.Call("createElement", "div")
	wrapper.Set("className", "mat-header-wrapper")

	// 1. Logo
	logo := document.Call("createElement", "div")
	logo.Set("className", "mat-header-logo")
	logo.Set("innerText", h.Brand)
	logo.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		js.Global().Get("location").Set("hash", "#home")
		return nil
	}))
	wrapper.Call("appendChild", logo)

	// 2. Navigation
	nav := document.Call("createElement", "nav")
	nav.Set("className", "mat-header-nav")

	links := []struct{ Text, Hash string }{
		{"Home", "#home"},
		{"Components", "#components"},
		{"Settings", "#settings"},
		{"Help", "#help"},
		{"Sprig", "#sprig"},
		{"Contact", "#contact"},
	}

	for _, link := range links {
		a := document.Call("createElement", "a")
		a.Set("innerText", link.Text)
		a.Set("href", link.Hash)
		a.Set("className", "mat-header-nav-link")
		nav.Call("appendChild", a)
	}
	wrapper.Call("appendChild", nav)

	// 3. Avatar
	actions := document.Call("createElement", "div")
	actions.Set("className", "mat-header-actions")

	avatar := document.Call("createElement", "div")
	avatar.Set("className", "mat-header-avatar")
	avatar.Set("innerText", "GA")
	actions.Call("appendChild", avatar)

	wrapper.Call("appendChild", actions)
	header.Call("appendChild", wrapper)
	return header
}
