//go:build js && wasm


package material

import "syscall/js"

type Tab struct {
	Label   string
	Content js.Value
}

type Tabs struct {
	Tabs          []Tab
	SelectedIndex int
	contentArea   js.Value
}

func NewTabs(tabs []Tab) *Tabs {
	return &Tabs{Tabs: tabs, SelectedIndex: 0}
}

func (t *Tabs) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-tabs")

	header := document.Call("createElement", "div")
	header.Set("className", "mat-tabs-header")

	t.contentArea = document.Call("createElement", "div")
	t.contentArea.Set("className", "mat-tabs-content")

	for i, tab := range t.Tabs {
		idx := i
		tabBtn := document.Call("createElement", "div")
		tabBtn.Set("className", "mat-tab-label")
		if i == t.SelectedIndex {
			tabBtn.Get("classList").Call("add", "active")
		}
		tabBtn.Set("innerText", tab.Label)

		tabBtn.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			t.Select(idx, header)
			return nil
		}))

		header.Call("appendChild", tabBtn)
	}

	container.Call("appendChild", header)
	container.Call("appendChild", t.contentArea)

	// Initial selection
	t.updateContent()

	return container
}

func (t *Tabs) Select(index int, header js.Value) {
	t.SelectedIndex = index
	labels := header.Get("children")
	for i := 0; i < labels.Get("length").Int(); i++ {
		labels.Call("item", i).Get("classList").Call("remove", "active")
	}
	labels.Call("item", index).Get("classList").Call("add", "active")
	t.updateContent()
}

func (t *Tabs) updateContent() {
	if !t.contentArea.IsUndefined() && !t.contentArea.IsNull() {
		t.contentArea.Set("innerHTML", "")
		t.contentArea.Call("appendChild", t.Tabs[t.SelectedIndex].Content)
	}
}
