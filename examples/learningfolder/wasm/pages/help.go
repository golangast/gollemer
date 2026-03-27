//go:build js && wasm


package pages

import (
	"syscall/js"

	"github.com/golangast/gollemer/examples/learningfolder/wasm/ui/material"
)

func RenderHelp() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "container")

	// Page Header
	header := document.Call("createElement", "section")
	header.Set("style", "text-align: left; max-width: 800px;")
	header.Set("innerHTML", "<h1>Developer Portal</h1><p class='text-muted' style='font-size: 1.25rem;'>Master the art of WebAssembly orchestration with the Gollemer OS documentation.</p>")
	container.Call("appendChild", header)

	// Main content using Tabs
	tabs := material.NewTabs([]material.Tab{
		{Label: "Orientation", Content: renderOrientation()},
		{Label: "Blueprint", Content: renderDevelopmentGuide()},
	})
	
	contentSection := document.Call("createElement", "div")
	contentSection.Set("style", "margin-top: -2rem;")
	contentSection.Call("appendChild", tabs.Render())
	container.Call("appendChild", contentSection)

	return container
}

func renderOrientation() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 4rem; padding: 2rem 0;")

	// Architecture
	arch := document.Call("createElement", "div")
	arch.Call("appendChild", material.NewCard("The High-Level Architecture", "Pure Go + WASM",
		"Gollemer OS is a pure-Go application compiled to WebAssembly. It bypasses traditional JavaScript frameworks by interacting directly with the DOM through a reactive event-loop written in Go.").Render())
	div.Call("appendChild", arch)

	// Directives
	dir := document.Call("createElement", "div")
	dir.Call("appendChild", createDocHeader("Logical Hierarchy"))
	
	list := material.NewList([]material.ListItem{
		{Label: "wasm/core", SubLabel: "Navigation routing and global state residency."},
		{Label: "wasm/ui/material", SubLabel: "The atomic design system of atomic components."},
		{Label: "wasm/pages", SubLabel: "Synthesis of components into functional views."},
		{Label: "assets/style", SubLabel: "Universal design tokens and glassmorphism definitions."},
	})
	dir.Call("appendChild", list.Render())
	div.Call("appendChild", dir)

	return div
}

func renderDevelopmentGuide() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 3rem; padding: 2rem 0;")

	// Creating a Page
	step1 := document.Call("createElement", "div")
	step1.Call("appendChild", createDocHeader("1. View Synthesis"))
	step1.Call("appendChild", createCodeBlock(`func RenderMyPage() js.Value {
    container := document.Call("createElement", "div")
    container.Set("className", "container")
    
    card := material.NewCard("Title", "Meta", "Body content...")
    container.Call("appendChild", card.Render())
    
    return container
}`))
	div.Call("appendChild", step1)

	// Routing
	step2 := document.Call("createElement", "div")
	step2.Call("appendChild", createDocHeader("2. Route Registration"))
	step2.Call("appendChild", createCodeBlock(`router.Routes: map[string]func() js.Value{
    "#home": pages.RenderHome,
    "#mypage": pages.RenderMyPage,
}`))
	div.Call("appendChild", step2)

	return div
}

func createDocHeader(text string) js.Value {
	document := js.Global().Get("document")
	h2 := document.Call("createElement", "h2")
	h2.Set("innerText", text)
	h2.Set("style", "margin-bottom: 1.5rem; font-size: 1.5rem; letter-spacing: -0.02em;")
	return h2
}

func createCodeBlock(code string) js.Value {
	document := js.Global().Get("document")
	pre := document.Call("createElement", "pre")
	pre.Set("style", "background: #000; border: 1px solid var(--border); padding: 1.5rem; border-radius: 16px; color: #a5b4fc; font-family: 'Fira Code', monospace; font-size: 0.9rem; overflow-x: auto;")
	pre.Set("innerText", code)
	return pre
}
