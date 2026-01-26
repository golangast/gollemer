package pages

import (
	"syscall/js"
	"time"

	"github.com/golangast/gollemer/learningfolder/wasm/ui/material"
)

func RenderHelp() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "help-page")

	// Page Header
	header := document.Call("createElement", "section")
	header.Set("style", "margin-bottom: 3rem; text-align: center; padding: 2rem 0;")
	header.Set("innerHTML", "<h1>Developer Documentation</h1><p>Comprehensive guide to the GopherApp Material Kit and WASM Architecture.</p>")
	container.Call("appendChild", header)

	// Main content using Tabs
	tabs := material.NewTabs([]material.Tab{
		{Label: "Getting Started", Content: renderGettingStarted()},
		{Label: "Component Catalog", Content: renderComponentCatalog()},
		{Label: "Development Guide", Content: renderAddingSection()},
	})
	container.Call("appendChild", tabs.Render())

	return container
}

func renderGettingStarted() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 2rem;")

	div.Call("appendChild", material.NewCard("Architecture Overview", "The WASM Core",
		"This application is built entirely in Go, compiled to WebAssembly. We use the syscall/js package to interact with the DOM directly, following a reactive, component-based pattern.").Render())

	// Step 1: Project Structure
	div.Call("appendChild", renderCategoryHeader("1. Project Structure"))
	div.Call("appendChild", material.NewList([]material.ListItem{
		{Label: "wasm/core", SubLabel: "Router and state persistence layer."},
		{Label: "wasm/ui/material", SubLabel: "Reusable UI components (The 'Material Kit')."},
		{Label: "wasm/pages", SubLabel: "Page components that compose UI elements into full views."},
		{Label: "assets/style", SubLabel: "CSS Variables and component-specific styles."},
	}).Render())

	// Step 2: The Router
	div.Call("appendChild", renderCategoryHeader("2. Defining Routes"))
	div.Call("appendChild", createCodeBlock(`// In wasm/wasm.go
router := &core.Router{
    Root: contentArea,
    Routes: map[string]func() js.Value{
        "#home":       pages.RenderHome,
        "#components": pages.RenderComponents,
    },
}
router.Start()`))

	// Step 3: Creating a Page
	div.Call("appendChild", renderCategoryHeader("3. Creating a Page"))
	div.Call("appendChild", createCodeBlock(`func RenderMyPage() js.Value {
    document := js.Global().Get("document")
    container := document.Call("createElement", "div")
    
    btn := material.NewButton("Click Me", "primary", func() {
        fmt.Println("Button clicked!")
    })
    
    container.Call("appendChild", btn.Render())
    return container
}`))

	return div
}

func renderComponentCatalog() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 2rem;")

	// --- Actions & Feedback ---
	div.Call("appendChild", renderCategoryHeader("Actions & Feedback"))

	div.Call("appendChild", renderShowcase("Button & FAB",
		`material.NewButton("Primary", "primary", nil)
material.NewFAB("+", "accent", nil)`,
		func() js.Value {
			box := createFlexRow()
			box.Call("appendChild", material.NewButton("Primary", "primary", nil).Render())
			box.Call("appendChild", material.NewButton("Accent", "accent", nil).Render())
			box.Call("appendChild", material.NewFAB("+", "primary", nil).Render())
			return box
		}()))

	div.Call("appendChild", renderShowcase("Badge & Chip",
		`material.NewBadge("Beta", "warning")
material.NewChip("Go WASM", "primary")`,
		func() js.Value {
			box := createFlexRow()
			box.Call("appendChild", material.NewBadge("NEW", "success").Render())
			box.Call("appendChild", material.NewBadge("BETA", "warning").Render())
			box.Call("appendChild", material.NewChip("Golang", "primary").Render())
			chip := material.NewChip("Removable", "accent")
			chip.Removable = true
			box.Call("appendChild", chip.Render())
			return box
		}()))

	div.Call("appendChild", renderShowcase("Spinners & Progress",
		`material.NewSpinner("md")
material.NewProgressBar(75.5, "determinate")`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%; display: flex; flex-direction: column; gap: 1rem;")
			row := createFlexRow()
			row.Call("appendChild", material.NewSpinner("sm").Render())
			row.Call("appendChild", material.NewSpinner("md").Render())
			row.Call("appendChild", material.NewSpinner("lg").Render())
			box.Call("appendChild", row)
			box.Call("appendChild", material.NewProgressBar(65.0, "determinate").Render())
			return box
		}()))

	// --- Form Inputs ---
	div.Call("appendChild", renderCategoryHeader("Form Inputs"))

	div.Call("appendChild", renderShowcase("Text Inputs & Select",
		`material.NewInput("Username", "Enter name...", "text")
material.NewSelect("Selection", options, nil)`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%; display: flex; flex-direction: column; gap: 1rem;")
			box.Call("appendChild", material.NewInput("Email", "user@example.com", "email").Render())
			box.Call("appendChild", material.NewSelect("Role", []material.SelectOption{
				{Label: "Admin", Value: "admin"},
				{Label: "Editor", Value: "editor"},
			}, nil).Render())
			return box
		}()))

	div.Call("appendChild", renderShowcase("Selection Controls",
		`material.NewSwitch("Enable", true, nil)
material.NewCheckbox("Agree", false, nil)
material.NewSlider("Volume", 0, 100, 1, 50, nil)`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%; display: flex; flex-direction: column; gap: 1rem;")
			row := createFlexRow()
			row.Call("appendChild", material.NewSwitch("Toggle", true, nil).Render())
			row.Call("appendChild", material.NewCheckbox("Check", false, nil).Render())
			box.Call("appendChild", row)
			box.Call("appendChild", material.NewSlider("Intensity", 0, 100, 1, 75, nil).Render())
			return box
		}()))

	// --- Navigation ---
	div.Call("appendChild", renderCategoryHeader("Navigation"))

	div.Call("appendChild", renderShowcase("Tabs & Breadcrumbs",
		`material.NewTabs(tabs)
material.NewBreadcrumbs(crumbs)`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%; display: flex; flex-direction: column; gap: 1.5rem;")
			box.Call("appendChild", material.NewBreadcrumbs([]material.Breadcrumb{
				{Label: "Home", Link: "#home"},
				{Label: "Help", Link: "#help"},
				{Label: "Catalog", Link: "#"},
			}).Render())
			box.Call("appendChild", material.NewTabs([]material.Tab{
				{Label: "One", Content: document.Call("createElement", "div")},
				{Label: "Two", Content: document.Call("createElement", "div")},
			}).Render())
			return box
		}()))

	div.Call("appendChild", renderShowcase("Pagination & Stepper",
		`material.NewPagination(1, 10, nil)
material.NewStepper(steps)`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%; display: flex; flex-direction: column; gap: 1.5rem;")
			box.Call("appendChild", material.NewPagination(3, 5, nil).Render())
			box.Call("appendChild", material.NewStepper([]material.StepperStep{
				{Label: "Step 1", Content: document.Call("createElement", "div")},
				{Label: "Step 2", Content: document.Call("createElement", "div")},
			}).Render())
			return box
		}()))

	// --- Content & Data ---
	div.Call("appendChild", renderCategoryHeader("Content & Data"))

	div.Call("appendChild", renderShowcase("Data Table",
		`material.NewDataTable(columns, rows)`,
		func() js.Value {
			cols := []material.DataTableColumn{
				{Header: "ID", Key: "id"},
				{Header: "Name", Key: "name"},
				{Header: "Status", Key: "status"},
			}
			rows := []map[string]interface{}{
				{"id": 1, "name": "Node A", "status": "Online"},
				{"id": 2, "name": "Node B", "status": "Pending"},
			}
			return material.NewDataTable(cols, rows).Render()
		}()))

	div.Call("appendChild", renderShowcase("Accordion & List",
		`material.NewAccordion(items)
material.NewList(listItems)`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%; display: flex; flex-direction: column; gap: 1rem;")
			box.Call("appendChild", material.NewAccordion([]material.AccordionItem{
				{Title: "Details", Content: material.NewList([]material.ListItem{
					{Label: "System", SubLabel: "Go 1.22"},
					{Label: "Arch", SubLabel: "WASM/JS"},
				}).Render()},
			}).Render())
			return box
		}()))

	// --- Overlays & Media ---
	div.Call("appendChild", renderCategoryHeader("Overlays & Media"))

	div.Call("appendChild", renderShowcase("Modal & Dialogs",
		`material.NewModal("Alert", content, nil).Render()
material.ShowSnackBar("Saved", "", time.Second)`,
		func() js.Value {
			box := createFlexRow()
			box.Call("appendChild", material.NewButton("Open Modal", "primary", func() {
				msg := document.Call("createElement", "p")
				msg.Set("innerText", "This is a modal content.")
				material.NewModal("System Message", msg, nil).Render()
			}).Render())
			box.Call("appendChild", material.NewButton("Show Toast", "accent", func() {
				material.ShowSnackBar("Operation Successful", "OK", 3*time.Second)
			}).Render())
			return box
		}()))

	div.Call("appendChild", renderShowcase("Carousel & Parallax",
		`material.NewCarousel(slides)
material.NewParallax(img, "600px")`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%;")
			box.Call("appendChild", material.NewCarousel([]material.CarouselSlide{
				{ImageURL: "https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=400", Caption: "WebAssembly"},
				{ImageURL: "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=400", Caption: "Circuit Board"},
			}).Render())
			return box
		}()))

	// --- Advanced & Utilities ---
	div.Call("appendChild", renderCategoryHeader("Advanced & Utilities"))

	div.Call("appendChild", renderShowcase("Alerts & Tooltips",
		`btn := material.NewButton("Hover Me", "primary", nil).Render()
material.NewTooltip(btn, "Useful info")`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%; display: flex; flex-direction: column; gap: 1rem;")
			box.Call("appendChild", material.NewAlert("Success Message", "success", true).Render())

			btn := material.NewButton("Hover for Tooltip", "primary", nil).Render()
			material.NewTooltip(btn, "This is a Material Tooltip!")
			box.Call("appendChild", btn)

			return box
		}()))

	div.Call("appendChild", renderShowcase("Tree View & Skeletons",
		`material.Tree{Nodes: nodes}
material.NewSkeleton("circle", "50px", "50px")`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("style", "width: 100%; display: flex; gap: 2rem; align-items: start;")

			tree := material.Tree{Nodes: []material.TreeNode{
				{Label: "src", Children: []material.TreeNode{
					{Label: "main.go"},
					{Label: "ui", Children: []material.TreeNode{{Label: "theme.go"}}},
				}},
			}}
			box.Call("appendChild", tree.Render())

			skelBox := document.Call("createElement", "div")
			skelBox.Set("style", "display: flex; flex-direction: column; gap: 0.5rem;")
			skelBox.Call("appendChild", material.NewSkeleton("circle", "40px", "40px").Render())
			skelBox.Call("appendChild", material.NewSkeleton("rect", "100px", "20px").Render())
			box.Call("appendChild", skelBox)

			return box
		}()))

	div.Call("appendChild", renderShowcase("Radio Groups",
		`material.NewRadioGroup("options", opts, nil)`,
		func() js.Value {
			opts := []material.RadioOption{
				{Label: "Fast", Value: "1"},
				{Label: "Secure", Value: "2"},
			}
			return material.NewRadioGroup("settings", opts, nil).Render()
		}()))

	return div
}

func renderAddingSection() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 1.5rem;")

	div.Call("appendChild", material.NewCard("Component Standard", "Creating New UI Elements",
		"All components should follow the struct initialization pattern and implement a Render() method returning a js.Value.").Render())

	steps := material.NewAccordion([]material.AccordionItem{
		{
			Title: "1. Create Component File",
			Content: createCodeBlock(`// wasm/ui/material/mycomponent.go
package material

type MyComponent struct {
    Title string
    el    js.Value
}`),
		},
		{
			Title: "2. Add Logic & Styling",
			Content: createCodeBlock(`func NewMyComponent(title string) *MyComponent {
    return &MyComponent{Title: title}
}

func (m *MyComponent) Render() js.Value {
    document := js.Global().Get("document")
    m.el = document.Call("createElement", "div")
    m.el.Set("className", "mat-my-component")
    m.el.Set("innerText", m.Title)
    return m.el
}`),
		},
		{
			Title:   "3. Register Styles",
			Content: document.Call("createElement", "p"),
		},
	})
	steps.Items[2].Content.Set("innerText", "Add your CSS to assets/style/material.css using variables for consistent theming.")

	div.Call("appendChild", steps.Render())
	return div
}

// Helper Utilities
func renderShowcase(title string, code string, demo js.Value) js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("style", "margin-bottom: 2rem; background: rgba(255,255,255,0.02); padding: 1.5rem; border-radius: 16px; border: 1px solid var(--glass-border);")

	h4 := document.Call("createElement", "h4")
	h4.Set("innerText", title)
	h4.Set("style", "margin-top: 0; margin-bottom: 1rem; color: var(--primary); font-size: 1.1rem;")
	container.Call("appendChild", h4)

	split := document.Call("createElement", "div")
	split.Set("className", "showcase-split")
	split.Set("style", "display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: start;")

	demoBox := document.Call("createElement", "div")
	demoBox.Set("style", "padding: 1.5rem; background: rgba(0,0,0,0.3); border-radius: 12px; min-height: 120px; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(255,255,255,0.05); overflow: hidden;")
	demoBox.Call("appendChild", demo)

	codeBox := createCodeBlock(code)

	split.Call("appendChild", demoBox)
	split.Call("appendChild", codeBox)
	container.Call("appendChild", split)

	return container
}

func renderCategoryHeader(title string) js.Value {
	document := js.Global().Get("document")
	h2 := document.Call("createElement", "h2")
	h2.Set("innerText", title)
	h2.Set("style", "border-left: 4px solid var(--primary); padding-left: 1rem; margin-top: 2rem; margin-bottom: 1.5rem; font-size: 1.4rem; color: #fff;")
	return h2
}

func createCodeBlock(code string) js.Value {
	document := js.Global().Get("document")
	pre := document.Call("createElement", "pre")
	pre.Set("style", "background: #0f172a; padding: 1.2rem; border-radius: 12px; color: #94a3b8; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.05); margin: 0; line-height: 1.5;")
	pre.Set("innerText", code)
	return pre
}

func createFlexRow() js.Value {
	document := js.Global().Get("document")
	row := document.Call("createElement", "div")
	row.Set("style", "display: flex; gap: 1rem; flex-wrap: wrap; align-items: center; justify-content: center;")
	return row
}
