package pages

import (
	"syscall/js"

	"github.com/golangast/gollemer/learningfolder/wasm/ui/material"
)

func RenderContact() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "contact-page")
	container.Set("style", "padding: 4rem 2rem; max-width: 800px; margin: 0 auto; min-height: 80vh;")

	// 1. Header
	heading := document.Call("createElement", "h1")
	heading.Set("innerText", "Contact Us")
	heading.Set("className", "mat-h1")
	container.Call("appendChild", heading)

	// 2. Subtitle
	sub := document.Call("createElement", "p")
	sub.Set("innerText", "Get in touch with the Gollemer team for support or inquiries.")
	sub.Set("className", "mat-body-1")
	container.Call("appendChild", sub)

	// 3. Contact Form Card
	card := document.Call("createElement", "div")
	card.Set("className", "mat-card")
	card.Set("style", "margin-top: 2rem; padding: 2rem;")

	form := document.Call("createElement", "form")

	// Name Input
	nameLabel := document.Call("createElement", "label")
	nameLabel.Set("innerText", "Name")
	nameLabel.Set("className", "mat-label")
	form.Call("appendChild", nameLabel)

	nameInput := document.Call("createElement", "input")
	nameInput.Set("type", "text")
	nameInput.Set("placeholder", "Your Name")
	nameInput.Set("className", "mat-input")
	nameInput.Set("style", "width: 100%; margin-bottom: 1.5rem;")
	form.Call("appendChild", nameInput)

	// Email Input
	emailLabel := document.Call("createElement", "label")
	emailLabel.Set("innerText", "Email")
	emailLabel.Set("className", "mat-label")
	form.Call("appendChild", emailLabel)

	emailInput := document.Call("createElement", "input")
	emailInput.Set("type", "email")
	emailInput.Set("placeholder", "your@email.com")
	emailInput.Set("className", "mat-input")
	emailInput.Set("style", "width: 100%; margin-bottom: 1.5rem;")
	form.Call("appendChild", emailInput)

	// Message Input
	msgLabel := document.Call("createElement", "label")
	msgLabel.Set("innerText", "Message")
	msgLabel.Set("className", "mat-label")
	form.Call("appendChild", msgLabel)

	msgInput := document.Call("createElement", "textarea")
	msgInput.Set("placeholder", "How can we help?")
	msgInput.Set("className", "mat-input")
	msgInput.Set("style", "width: 100%; min-height: 150px; margin-bottom: 1.5rem;")
	form.Call("appendChild", msgInput)

	// Submit Button
	submitBtn := material.NewButton("Send Message", "primary", func() {
		js.Global().Call("alert", "Thank you for your message! We will get back to you soon.")
	})
	form.Call("appendChild", submitBtn.Render())

	card.Call("appendChild", form)
	container.Call("appendChild", card)

	// 4. Contact Details
	details := document.Call("createElement", "div")
	details.Set("style", "margin-top: 3rem; display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem;")

	addr := document.Call("createElement", "div")
	addr.Set("innerHTML", "<h3>Address</h3><p>123 WebAssembly Blvd<br>Binary City, WASM 01010</p>")
	details.Call("appendChild", addr)

	social := document.Call("createElement", "div")
	social.Set("innerHTML", "<h3>Social</h3><p>Follow us on GitHub and GopherNet.</p>")
	details.Call("appendChild", social)

	container.Call("appendChild", details)

	return container
}
