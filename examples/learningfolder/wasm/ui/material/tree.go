//go:build js && wasm


package material

import "syscall/js"

type TreeNode struct {
	Label      string
	Children   []TreeNode
	IsExpanded bool
}

type Tree struct {
	Nodes []TreeNode
}

func (t *Tree) Render() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "mat-tree")

	for _, node := range t.Nodes {
		container.Call("appendChild", renderNode(node))
	}

	return container
}

func renderNode(node TreeNode) js.Value {
	document := js.Global().Get("document")
	item := document.Call("createElement", "div")
	item.Set("className", "mat-tree-node")

	labelRow := document.Call("createElement", "div")
	labelRow.Set("className", "mat-tree-row")

	if len(node.Children) > 0 {
		toggle := document.Call("createElement", "span")
		toggle.Set("className", "mat-tree-toggle")
		toggle.Set("innerText", "▶")
		labelRow.Call("appendChild", toggle)
	}

	labelText := document.Call("createElement", "span")
	labelText.Set("innerText", node.Label)
	labelRow.Call("appendChild", labelText)
	item.Call("appendChild", labelRow)

	if len(node.Children) > 0 {
		childContainer := document.Call("createElement", "div")
		childContainer.Set("className", "mat-tree-children")
		for _, child := range node.Children {
			childContainer.Call("appendChild", renderNode(child))
		}
		item.Call("appendChild", childContainer)

		labelRow.Call("addEventListener", "click", js.FuncOf(func(this js.Value, args []js.Value) any {
			isOpen := childContainer.Get("classList").Call("toggle", "open").Bool()
			if isOpen {
				labelRow.Call("querySelector", ".mat-tree-toggle").Get("style").Call("setProperty", "transform", "rotate(90deg)")
			} else {
				labelRow.Call("querySelector", ".mat-tree-toggle").Get("style").Call("setProperty", "transform", "rotate(0deg)")
			}
			return nil
		}))
	}

	return item
}
