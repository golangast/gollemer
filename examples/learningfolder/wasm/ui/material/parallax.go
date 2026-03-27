//go:build js && wasm


package material

import (
	"fmt"
	"syscall/js"
)

type Parallax struct {
	ImageURL string
	Height   string
	Content  js.Value
}

func NewParallax(img, height string, content js.Value) *Parallax {
	return &Parallax{ImageURL: img, Height: height, Content: content}
}

func (p *Parallax) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-parallax-container")
	container.Get("style").Call("setProperty", "height", p.Height)

	bg := document.Call("createElement", "div")
	bg.Set("className", "mat-parallax-bg")

	// Use img element with high priority since parallax is typically hero content
	img := document.Call("createElement", "img")
	img.Set("src", p.ImageURL)
	img.Set("alt", "Parallax background")
	img.Set("className", "mat-parallax-image")
	img.Set("width", "1600") // Explicit dimensions for CLS prevention
	img.Set("height", "900") // 16:9 aspect ratio
	img.Set("fetchpriority", "high")
	img.Set("loading", "eager")
	bg.Call("appendChild", img)

	content := document.Call("createElement", "div")
	content.Set("className", "mat-parallax-content")
	if !p.Content.IsUndefined() && !p.Content.IsNull() {
		content.Call("appendChild", p.Content)
	}

	container.Call("appendChild", bg)
	container.Call("appendChild", content)

	// Parallax Scroll Effect
	js.Global().Call("addEventListener", "scroll", js.FuncOf(func(this js.Value, args []js.Value) any {
		rect := container.Call("getBoundingClientRect")
		top := rect.Get("top").Float()

		// Only calculate if visible
		if top < js.Global().Get("innerHeight").Float() && rect.Get("bottom").Float() > 0 {
			offset := (top) * 0.4
			bg.Get("style").Call("setProperty", "transform", fmt.Sprintf("translateY(%.2fpx)", offset))
		}
		return nil
	}))

	return container
}
