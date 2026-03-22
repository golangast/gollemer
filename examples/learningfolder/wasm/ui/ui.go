package ui

import "syscall/js"

type Navbar struct {
	IsOpen bool
	el     js.Value
}

func NewNavbar() *Navbar {
	return &Navbar{IsOpen: false}
}

func (n *Navbar) Render() js.Value {
	document := js.Global().Get("document")

	// 1. Top Level Container
	nav := document.Call("createElement", "nav")
	nav.Set("className", "nav-container")

	// 2. Brand/Logo
	logo := document.Call("createElement", "div")
	logo.Set("className", "nav-brand")
	logo.Set("innerText", "GopherApp")
	logo.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		js.Global().Get("location").Set("hash", "#home")
		return nil
	}))
	nav.Call("appendChild", logo)

	// 3. Links Wrapper
	links := document.Call("createElement", "div")
	links.Set("className", "nav-links")
	n.el = links // Save for toggling

	items := []struct {
		Text string
		Hash string
	}{
		{"Home", "#home"},
		{"Components", "#components"},
		{"Services", "#services"},
		{"About", "#about"},
		{"Contact", "#contact"},
	}

	for _, item := range items {
		a := document.Call("createElement", "a")
		a.Set("innerText", item.Text)
		a.Set("href", item.Hash)
		// Close menu when a link is clicked
		a.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			n.Close()
			return nil
		}))
		links.Call("appendChild", a)
	}

	// 4. Hamburger Button
	btn := document.Call("createElement", "button")
	btn.Set("id", "hamburger-btn")
	btn.Set("innerHTML", "&#9776;") // Hamburger icon
	btn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		n.Toggle()
		return nil
	}))

	nav.Call("appendChild", links)
	nav.Call("appendChild", btn)
	return nav
}

func (n *Navbar) Toggle() {
	n.IsOpen = !n.IsOpen
	n.updateUI()
}

func (n *Navbar) Close() {
	n.IsOpen = false
	n.updateUI()
}

func (n *Navbar) updateUI() {
	if n.IsOpen {
		n.el.Get("classList").Call("add", "open")
	} else {
		n.el.Get("classList").Call("remove", "open")
	}
}
