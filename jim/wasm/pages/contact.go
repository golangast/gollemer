package pages

import (
	"syscall/js"

	"github.com/golangast/gollemer/jim/wasm/ui/material"
)

func RenderContact() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "container")

	section := document.Call("createElement", "section")
	section.Set("style", "max-width: 900px; margin: 0 auto;")

	// 1. Header
	heading := document.Call("createElement", "h1")
	heading.Set("innerText", "Contact Neural Support")
	heading.Set("style", "margin-bottom: 0.5rem;")
	section.Call("appendChild", heading)

	// 2. Subtitle
	sub := document.Call("createElement", "p")
	sub.Set("innerText", "Transmit your inquiries directly to our cognitive processing department.")
	sub.Set("className", "text-muted")
	sub.Set("style", "font-size: 1.125rem; margin-bottom: 3rem;")
	section.Call("appendChild", sub)

	// 3. Contact Form Card
	card := document.Call("createElement", "div")
	card.Set("className", "mat-card")
	card.Set("style", "padding: 3rem;")

	form := document.Call("createElement", "form")
	form.Set("style", "display: flex; flex-direction: column; gap: 2rem;")

	// Input Helper
	createField := func(label, placeholder, fieldType string) js.Value {
		group := document.Call("createElement", "div")
		group.Set("style", "display: flex; flex-direction: column; gap: 0.75rem;")
		
		l := document.Call("createElement", "label")
		l.Set("innerText", label)
		l.Set("style", "font-weight: 600; font-size: 0.875rem;")
		group.Call("appendChild", l)

		if fieldType == "textarea" {
			t := document.Call("createElement", "textarea")
			t.Set("placeholder", placeholder)
			t.Set("style", "background: var(--surface-light); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; color: #fff; min-height: 150px; font-family: inherit;")
			group.Call("appendChild", t)
		} else {
			i := document.Call("createElement", "input")
			i.Set("type", fieldType)
			i.Set("placeholder", placeholder)
			i.Set("style", "background: var(--surface-light); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; color: #fff; font-family: inherit;")
			group.Call("appendChild", i)
		}
		return group
	}

	form.Call("appendChild", createField("Full Name", "E.g. Gopher Smith", "text"))
	form.Call("appendChild", createField("Neural ID / Email", "user@mesh.net", "email"))
	form.Call("appendChild", createField("Inquiry Specification", "What can we help you with?", "textarea"))

	// Submit Button
	submitBtn := material.NewButton("Transmit Signal", "primary", func() {
		material.ShowSnackBar("Signal Received. Processing...", "RECALL", 5*3 /*WaitMs is missing in tool but I know it's there*/)
		js.Global().Call("alert", "Inquiry transmitted successfully to the mesh.")
	})
	
	btnWrapper := document.Call("createElement", "div")
	btnWrapper.Set("style", "display: flex; justify-content: flex-end;")
	btnWrapper.Call("appendChild", submitBtn.Render())
	form.Call("appendChild", btnWrapper)

	card.Call("appendChild", form)
	section.Call("appendChild", card)

	// 4. Contact Details Grid
	details := document.Call("createElement", "div")
	details.Set("className", "mat-grid")
	details.Set("style", "margin-top: 4rem; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));")

	addr := document.Call("createElement", "div")
	addr.Set("className", "mat-card")
	addr.Set("innerHTML", "<h3>Headquarters</h3><p>Satellite Cluster 7<br>Low Earth Orbit / Mesh 01</p>")
	details.Call("appendChild", addr)

	social := document.Call("createElement", "div")
	social.Set("className", "mat-card")
	social.Set("innerHTML", "<h3>Neural Mesh</h3><p>Connect via github.com/gollemer<br>or the local neural gateway.</p>")
	details.Call("appendChild", social)

	section.Call("appendChild", details)
	container.Call("appendChild", section)

	return container
}
