package pages

import (
	"syscall/js"
	"time"

	"github.com/golangast/gollemer/learningfolder/wasm/ui/material"
)

func RenderComponents() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "components-demo-page")

	title := document.Call("createElement", "h1")
	title.Set("innerText", "Universal Component Kit")
	container.Call("appendChild", title)

	// Create Tabs for all categories
	tabs := material.NewTabs([]material.Tab{
		{Label: "Navigation", Content: renderNavigationCategory()},
		{Label: "Forms & Controls", Content: renderFormsCategory()},
		{Label: "Feedback & Overlays", Content: renderFeedbackCategory()},
		{Label: "Data & Media", Content: renderMediaCategory()},
	})
	container.Call("appendChild", tabs.Render())

	return container
}

func renderNavigationCategory() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 2rem;")

	// 1. Breadcrumbs
	bc := material.NewBreadcrumbs([]material.Breadcrumb{
		{Label: "Main", Link: "#home"},
		{Label: "Showcase", Link: "#components"},
		{Label: "Navigation", Link: "#"},
		{Label: "Help", Link: "#help"},
	})
	div.Call("appendChild", createSection("Breadcrumbs", bc.Render()))

	// 2. Stepper
	stepper := material.NewStepper([]material.StepperStep{
		{Label: "Auth", Content: document.Call("createElement", "div")},
		{Label: "Server Selection", Content: document.Call("createElement", "div")},
		{Label: "Confirmation", Content: document.Call("createElement", "div")},
	})
	stepper.Steps[0].Content.Set("innerText", "Step 1: Authenticate with neural link.")
	div.Call("appendChild", createSection("Stepper", stepper.Render()))

	// 3. Tree
	tree := material.Tree{Nodes: []material.TreeNode{
		{Label: "src", Children: []material.TreeNode{
			{Label: "main.go"},
			{Label: "types.go"},
		}},
		{Label: "assets", Children: []material.TreeNode{
			{Label: "style.css"},
		}},
	}}
	div.Call("appendChild", createSection("Tree View", tree.Render()))

	// 4. Pagination
	div.Call("appendChild", createSection("Pagination", material.NewPagination(3, 10, nil).Render()))

	return div
}

func renderFormsCategory() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 2rem;")

	// Buttons
	btnRow := document.Call("createElement", "div")
	btnRow.Set("style", "display: flex; gap: 1rem;")
	btnRow.Call("appendChild", material.NewButton("Primary", "primary", nil).Render())
	btnRow.Call("appendChild", material.NewButton("Outline", "outline", nil).Render())
	div.Call("appendChild", createSection("Buttons", btnRow))

	// Inputs & Select
	div.Call("appendChild", createSection("Input Field", material.NewInput("Username", "@gopher", "text").Render()))
	div.Call("appendChild", createSection("Select Dropdown", material.NewSelect("Platform", []material.SelectOption{
		{Label: "Darwin", Value: "os1"}, {Label: "Linux", Value: "os2"},
	}, nil).Render()))

	// Selection Controls
	controls := document.Call("createElement", "div")
	controls.Set("style", "display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;")

	col1 := document.Call("createElement", "div")
	col1.Call("appendChild", material.NewSwitch("Enable Turbo", true, nil).Render())
	col1.Call("appendChild", material.NewCheckbox("Agree to Node Privacy", false, nil).Render())
	controls.Call("appendChild", col1)

	col2 := document.Call("createElement", "div")
	radio := material.NewRadioGroup("env", []material.RadioOption{
		{Label: "Production", Value: "p"}, {Label: "Staging", Value: "s"},
	}, nil)
	col2.Call("appendChild", radio.Render())
	controls.Call("appendChild", col2)

	div.Call("appendChild", createSection("Selection Controls", controls))

	// Chips & Slider
	chips := document.Call("createElement", "div")
	chips.Call("appendChild", material.NewChip("GoLang", "primary").Render())
	chips.Call("appendChild", material.NewChip("WASM", "").Render())
	chips.Call("appendChild", material.NewChip("Neural", "primary").Render())
	div.Call("appendChild", createSection("Chips & Slider", chips))
	div.Call("appendChild", material.NewSlider("Power Output", 0, 100, 1, 75, nil).Render())

	return div
}

func renderFeedbackCategory() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 2rem;")

	// Alerts
	div.Call("appendChild", createSection("Alerts", material.NewAlert("New system update available.", "info", true).Render()))

	// Popups (Buttons to trigger)
	popupRow := document.Call("createElement", "div")
	popupRow.Set("style", "display: flex; gap: 1rem;")

	snackBtn := material.NewButton("Show Toast", "outline", func() { material.ShowSnackBar("Agent deployed", "OK", 3*time.Second) })
	sheetBtn := material.NewButton("Open Sheet", "primary", func() {
		content := document.Call("createElement", "div")
		content.Set("innerHTML", "<h3>Quick Config</h3><p>Choose neural mapping intensity.</p>")
		material.NewBottomSheet(content).Render()
	})

	popupRow.Call("appendChild", snackBtn.Render())
	popupRow.Call("appendChild", sheetBtn.Render())
	div.Call("appendChild", createSection("Toasts & Sheets", popupRow))

	// Indicators
	prog := document.Call("createElement", "div")
	prog.Call("appendChild", material.NewProgressBar(40, "determinate").Render())
	prog.Call("appendChild", material.NewSpinner("sm").Render())
	div.Call("appendChild", createSection("Progress & Spinners", prog))

	// Skeletons
	skel := document.Call("createElement", "div")
	skel.Call("appendChild", material.NewSkeleton("circle", "40px", "40px").Render())
	skel.Call("appendChild", material.NewSkeleton("text", "200px", "20px").Render())
	div.Call("appendChild", createSection("Skeleton Loaders", skel))

	return div
}

func renderMediaCategory() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 2rem;")

	// Accordion
	acc := material.NewAccordion([]material.AccordionItem{
		{Title: "Technical Specs", Content: material.NewBadge("Native Go", "success").Render()},
		{Title: "Security Audit", Content: document.Call("createElement", "p")},
	})
	div.Call("appendChild", createSection("Accordion", acc.Render()))

	// Carousel
	carousel := material.NewCarousel([]material.CarouselSlide{
		{ImageURL: "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=1200", Caption: "Hardware Layer"},
		{ImageURL: "https://images.unsplash.com/photo-1558494949-ef010cbdcc4b?q=80&w=1200", Caption: "Data Centers"},
	})
	div.Call("appendChild", createSection("Carousel", carousel.Render()))

	// Table
	cols := []material.DataTableColumn{{Header: "Metric", Key: "m"}, {Header: "Value", Key: "v"}}
	data := []map[string]interface{}{{"m": "TPS", "v": "10k"}, {"m": "Latency", "v": "0.1ms"}}
	div.Call("appendChild", createSection("Data Table", material.NewDataTable(cols, data).Render()))

	return div
}

func createSection(title string, content js.Value) js.Value {
	document := js.Global().Get("document")
	sec := document.Call("createElement", "section")
	sec.Set("style", "margin-bottom: 2rem; border-bottom: 1px solid var(--glass-border); padding-bottom: 1rem;")

	h2 := document.Call("createElement", "h3")
	h2.Set("innerText", title)
	h2.Set("style", "margin-bottom: 1rem; color: var(--text-muted); font-size: 0.9rem; text-transform: uppercase;")

	sec.Call("appendChild", h2)
	sec.Call("appendChild", content)
	return sec
}
