package main

import (
	"syscall/js"

	"github.com/golangast/gollemer/jim/wasm/core"
	"github.com/golangast/gollemer/jim/wasm/lib"
	"github.com/golangast/gollemer/jim/wasm/pages"
	"github.com/golangast/gollemer/jim/wasm/ui"
)

func main() {
	js.Global().Get("console").Call("log", "Gollemer WASM started!")
	lib.RegisterSprig()
	document := js.Global().Get("document")
	body := document.Get("body")
	app := document.Call("getElementById", "app-root")

	// 1. Add Header
	header := ui.NewHeader("GopherApp")
	body.Call("prepend", header.Render())

	// 2. Create Content Area
	contentArea := document.Call("createElement", "main")
	app.Call("appendChild", contentArea)

	// 3. Add Footer
	footer := ui.NewFooter("© 2026 Gollemer Advanced AI. Built with Go WebAssembly.")
	body.Call("appendChild", footer.Render())

	// 4. Initialize Router
	router := &core.Router{
		Root: contentArea,
		Routes: map[string]func() js.Value{
			"#home":       pages.RenderHome,
			"#services":   func() js.Value { return renderPlaceholder("Services") },
			"#about":      func() js.Value { return renderPlaceholder("About Us") },
			"#contact":    pages.RenderContact,
			"#settings":   pages.RenderSettings,
			"#components": pages.RenderComponents,
			"#help":       pages.RenderHelp,
			"#sprig":      pages.RenderSprig,
		},
	}

	router.Start()

	select {}
}

func renderPlaceholder(title string) js.Value {
	div := js.Global().Get("document").Call("createElement", "div")
	div.Set("innerHTML", "<h1>"+title+"</h1><p>This page is currently under development.</p>")
	return div
}
