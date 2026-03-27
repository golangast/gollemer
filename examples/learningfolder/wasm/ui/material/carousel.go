//go:build js && wasm


package material

import (
	"strconv"
	"syscall/js"
)

type CarouselSlide struct {
	ImageURL string
	Caption  string
}

type Carousel struct {
	Slides []CarouselSlide
	Index  int
	el     js.Value
}

func NewCarousel(slides []CarouselSlide) *Carousel {
	return &Carousel{Slides: slides}
}

func (c *Carousel) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-carousel")
	c.el = container

	wrapper := document.Call("createElement", "div")
	wrapper.Set("className", "mat-carousel-wrapper")

	for i, slide := range c.Slides {
		s := document.Call("createElement", "div")
		s.Set("className", "mat-carousel-slide")

		// Create img element with proper loading attributes
		img := document.Call("createElement", "img")
		img.Set("src", slide.ImageURL)
		img.Set("alt", slide.Caption)
		img.Set("className", "mat-carousel-image")
		img.Set("width", "1600") // Explicit dimensions for CLS prevention
		img.Set("height", "900") // 16:9 aspect ratio

		// First slide gets high priority for LCP, others lazy load
		if i == 0 {
			img.Set("fetchpriority", "high")
			img.Set("loading", "eager")
		} else {
			img.Set("loading", "lazy")
		}
		s.Call("appendChild", img)

		if slide.Caption != "" {
			cap := document.Call("createElement", "div")
			cap.Set("className", "mat-carousel-caption")
			cap.Set("innerText", slide.Caption)
			s.Call("appendChild", cap)
		}
		wrapper.Call("appendChild", s)
	}

	container.Call("appendChild", wrapper)

	// Arrows Container
	arrows := document.Call("createElement", "div")
	arrows.Set("className", "mat-carousel-arrows")

	prev := document.Call("createElement", "button")
	prev.Set("innerText", "❮")
	prev.Set("className", "mat-carousel-prev")
	prev.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		c.Move(-1)
		return nil
	}))

	next := document.Call("createElement", "button")
	next.Set("innerText", "❯")
	next.Set("className", "mat-carousel-next")
	next.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
		c.Move(1)
		return nil
	}))

	container.Call("appendChild", prev)
	container.Call("appendChild", next)

	return container
}

func (c *Carousel) Move(dir int) {
	if len(c.Slides) == 0 {
		return
	}
	c.Index = (c.Index + dir + len(c.Slides)) % len(c.Slides)
	wrapper := c.el.Call("querySelector", ".mat-carousel-wrapper")
	offset := -c.Index * 100
	wrapper.Get("style").Call("setProperty", "transform", "translateX("+strconv.Itoa(offset)+"%)")
}
