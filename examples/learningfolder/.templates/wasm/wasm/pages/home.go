package pages

import (
	"syscall/js"

	"github.com/golangast/gollemer/examples/learningfolder/wasm/ui/material"
)

func RenderHome() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "advanced-landing")

	// 1. Hero
	heroContent := document.Call("createElement", "div")
	heroContent.Set("innerHTML", `
		<h1 style="margin-bottom: 0;">Gollemer OS</h1>
		<p style="font-size: 1.5rem; margin-top: 0.5rem; color: #a5b4fc; font-weight: 500;">Build WASM Apps with Simple Blocks</p>
		<p class="text-muted" style="max-width: 600px; margin: 1.5rem auto 3rem;">
			Forget complex JavaScript frameworks. We provide simple, high-performance building blocks written in Go 
			that you can snap together to build powerful decentralized applications.
		</p>
	`)
	parallax := material.NewParallax("assets/images/hero_bg.png", "80vh", heroContent)
	container.Call("appendChild", parallax.Render())

	mainContent := document.Call("createElement", "div")
	mainContent.Set("className", "container")

	// 2. The Simple Building Blocks (The Core Explanation)
	primerSection := document.Call("createElement", "section")
	primerSection.Set("innerHTML", `
		<div style="text-align: center; margin-bottom: 5rem;">
			<h2 style="font-size: 3rem; margin-bottom: 1rem;">Simple Building Blocks</h2>
			<p class="text-muted" style="font-size: 1.25rem;">Building a web UI is just like playing with Lego. Here's how we do it.</p>
		</div>
		<div class="mat-grid">
			<div class="mat-card">
				<div style="font-size: 2.5rem; margin-bottom: 1rem;">📦</div>
				<h3>Containers (Cards & Lists)</h3>
				<p>Think of these as boxes. You put your information inside a <b>Card</b> to make it stand out, or use a <b>List</b> to keep things tidy and organized.</p>
			</div>
			<div class="mat-card">
				<div style="font-size: 2.5rem; margin-bottom: 1rem;">🖱️</div>
				<h3>Actions (Buttons & Sliders)</h3>
				<p>Buttons let users "do" things. Sliders and Switches let them adjust settings. It's direct, simple interaction without the jargon.</p>
			</div>
			<div class="mat-card">
				<div style="font-size: 2.5rem; margin-bottom: 1rem;">🗺️</div>
				<h3>Moving Around (Tabs & Menus)</h3>
				<p>Need different pages? Use <b>Tabs</b> to switch views or <b>Breadcrumbs</b> to show users exactly where they are in your app.</p>
			</div>
			<div class="mat-card">
				<div style="font-size: 2.5rem; margin-bottom: 1rem;">🔔</div>
				<h3>Talking Back (Alerts & Toasts)</h3>
				<p>When something happens, tell your user. Use <b>Alerts</b> for big news and <b>Toasts</b> for small updates that don't get in the way.</p>
			</div>
		</div>
	`)
	mainContent.Call("appendChild", primerSection)

	// 3. Why It's Better
	whySection := document.Call("createElement", "section")
	whySection.Set("style", "background: var(--surface); border-radius: 40px; padding: 4rem; margin: 4rem 0;")
	whySection.Set("innerHTML", `
		<div class="mat-grid" style="align-items: center;">
			<div>
				<h2 style="font-size: 2.5rem; margin-bottom: 1.5rem;">Pure Go, No Magic</h2>
				<p class="text-muted" style="font-size: 1.1rem; margin-bottom: 2rem;">
					Every component you see is a 100% native Go struct. No hidden JavaScript layers, 
					no complex build tools. Just Go code compiling straight to the browser.
				</p>
				<ul style="list-style: none; padding: 0; display: flex; flex-direction: column; gap: 1rem;">
					<li style="display: flex; gap: 0.75rem; align-items: center;">✅ <b>Type Safe:</b> Your UI follows your data models.</li>
					<li style="display: flex; gap: 0.75rem; align-items: center;">✅ <b>Blazing Fast:</b> Native binary execution in the browser.</li>
					<li style="display: flex; gap: 0.75rem; align-items: center;">✅ <b>Modern Look:</b> Beautiful styles included out of the box.</li>
				</ul>
			</div>
			<div style="text-align: center;">
				<div style="background: #000; border: 1px solid var(--border); padding: 2rem; border-radius: 20px; text-align: left; font-family: monospace;">
					<span style="color: #6366f1;">btn</span> := material.<span style="color: #a5b4fc;">NewButton</span>(<span style="color: #f472b6;">"Click Me"</span>, <span style="color: #f472b6;">"primary"</span>, <span style="color: #8b5cf6;">func</span>() { ... })<br>
					<span style="color: #6366f1;">card</span> := material.<span style="color: #a5b4fc;">NewCard</span>(<span style="color: #f472b6;">"Intro"</span>, <span style="color: #f472b6;">"Hi!"</span>, <span style="color: #f472b6;">"I am a card."</span>)
				</div>
			</div>
		</div>
	`)
	mainContent.Call("appendChild", whySection)

	// 4. Final CTA
	cta := document.Call("createElement", "section")
	cta.Set("style", "text-align: center;")
	cta.Set("innerHTML", `
		<h2 style="font-size: 3rem; margin-bottom: 1rem;">Ready to build something?</h2>
		<p class="text-muted" style="margin-bottom: 3rem;">Explore every building block and see the code for yourself.</p>
	`)
	btn := material.NewButton("Explore All Components", "primary", func() {
		js.Global().Get("location").Set("hash", "#components")
	})
	cta.Call("appendChild", btn.Render())
	mainContent.Call("appendChild", cta)

	container.Call("appendChild", mainContent)

	return container
}
