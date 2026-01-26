package pages

import (
	"syscall/js"

	"github.com/golangast/gollemer/learningfolder/wasm/ui/material"
)

func RenderHome() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "advanced-landing")

	// 1. Parallax Hero
	heroContent := document.Call("createElement", "div")
	heroContent.Set("innerHTML", "<h1>Gollemer OS</h1><p>The Future of WebAssembly Orchestration</p>")
	parallax := material.NewParallax("https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=1400", "500px", heroContent)
	container.Call("appendChild", parallax.Render())

	// 2. Carousel Showcase
	carousel := material.NewCarousel([]material.CarouselSlide{
		{ImageURL: "https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=1400", Caption: "Neural Processing"},
		{ImageURL: "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=1400", Caption: "Encrypted Mesh"},
	})
	container.Call("appendChild", carousel.Render())

	// 3. Grid of Stats with Tooltips
	grid := document.Call("createElement", "div")
	grid.Set("style", "padding: 4rem 2rem; display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem;")

	node1 := material.NewCard("Node Alpha", "99.9% Up", "Integrated node with active feedback.")
	node1El := node1.Render()
	material.NewTooltip(node1El, "Uptime verified by mesh consensus.")
	grid.Call("appendChild", node1El)

	// 4. Bottom Sheet Trigger
	bottomBtn := material.NewButton("Quick Settings", "primary", func() {
		sheetContent := document.Call("createElement", "div")
		sheetContent.Set("innerHTML", "<h2>Preferences</h2><p>Adjust system latency and thermal limits.</p>")
		material.NewBottomSheet(sheetContent).Render()
	})
	container.Call("appendChild", grid)
	container.Call("appendChild", bottomBtn.Render())

	// 5. FAB
	fab := material.NewFAB("+", "primary", func() {
		js.Global().Call("alert", "New Task Created")
	})
	container.Call("appendChild", fab.Render())

	return container
}
