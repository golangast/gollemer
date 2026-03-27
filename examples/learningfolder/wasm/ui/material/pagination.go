//go:build js && wasm


package material

import (
	"fmt"
	"syscall/js"
)

type Pagination struct {
	CurrentPage  int
	TotalPages   int
	OnPageChange func(int)
}

func NewPagination(current, total int, onChange func(int)) *Pagination {
	return &Pagination{CurrentPage: current, TotalPages: total, OnPageChange: onChange}
}

func (p *Pagination) Render() js.Value {
	document := js.Global().Get("document")
	nav := document.Call("createElement", "nav")
	nav.Set("className", "mat-pagination")

	for i := 1; i <= p.TotalPages; i++ {
		page := i
		btn := document.Call("createElement", "button")
		btn.Set("className", "mat-page-btn")
		if i == p.CurrentPage {
			btn.Get("classList").Call("add", "active")
		}
		btn.Set("innerText", fmt.Sprintf("%d", i))

		btn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			if p.OnPageChange != nil {
				p.OnPageChange(page)
			}
			return nil
		}))
		nav.Call("appendChild", btn)
	}

	return nav
}
