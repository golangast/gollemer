package pages

import (
	"syscall/js"
	"time"

	"github.com/golangast/gollemer/examples/learningfolder/wasm/ui/material"
)

func RenderComponents() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "container")

	// Page Title
	header := document.Call("createElement", "section")
	header.Set("style", "text-align: center; border-bottom: 1px solid var(--border); margin-bottom: 2rem;")
	header.Set("innerHTML", "<h1>Universal Component Library</h1><p class='text-muted'>Every WASM-native component available in the Gollemer OS kit, complete with implementation code.</p>")
	container.Call("appendChild", header)

	// exhaustive categorized tab system
	tabs := material.NewTabs([]material.Tab{
		{Label: "Actions", Content: renderActionsGroup()},
		{Label: "Navigation", Content: renderNavGroup()},
		{Label: "Data & Info", Content: renderDataGroup()},
		{Label: "Interactions", Content: renderInteractionGroup()},
		{Label: "State & Progress", Content: renderStateGroup()},
	})
	container.Call("appendChild", tabs.Render())

	return container
}

// --- Category: Actions ---
func renderActionsGroup() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 4rem; padding: 2rem 0;")

	div.Call("appendChild", renderShowcase("Button", "Standard interactive trigger.",
		`btn := material.NewButton("Launch", "primary", func() { ... })`,
		material.NewButton("Primary Action", "primary", nil).Render()))

	div.Call("appendChild", renderShowcase("FAB (Floating Action Button)", "Corner-fixed primary action.",
		`fab := material.NewFAB("+", "primary", nil)`,
		func() js.Value {
			f := material.NewFAB("+", "primary", nil).Render()
			f.Get("style").Set("position", "static")
			return f
		}()))

	div.Call("appendChild", renderShowcase("Switch", "Binary state toggle.",
		`sw := material.NewSwitch("Turbo", true, nil)`,
		material.NewSwitch("Neural Acceleration", true, nil).Render()))

	div.Call("appendChild", renderShowcase("Checkbox", "Individual option selection.",
		`cb := material.NewCheckbox("Agree", false, nil)`,
		material.NewCheckbox("Accept Mesh Terms", false, nil).Render()))

	div.Call("appendChild", renderShowcase("Radio Group", "Select one from a group.",
		`opts := []material.RadioOption{{Label: "Dev", Value: "d"}, {Label: "Prod", Value: "p"}}
rg := material.NewRadioGroup("env", opts, nil)`,
		material.NewRadioGroup("env_demo", []material.RadioOption{
			{Label: "Development", Value: "d"},
			{Label: "Production", Value: "p"},
		}, nil).Render()))

	div.Call("appendChild", renderShowcase("Slider", "Range-based value adjustment.",
		`sl := material.NewSlider("Volume", 0, 100, 1, 50, nil)`,
		material.NewSlider("Signal Intensity", 0, 100, 1, 75, nil).Render()))

	div.Call("appendChild", renderShowcase("Input Field", "Textual data entry.",
		`in := material.NewInput("Name", "Gopher", "text")`,
		material.NewInput("Access Key", "x-992-alpha", "password").Render()))

	div.Call("appendChild", renderShowcase("Select Menu", "Dropdown option selection.",
		`opts := []material.SelectOption{{Label: "Go", Value: "go"}, {Label: "JS", Value: "js"}}
sel := material.NewSelect("Runtime", opts, nil)`,
		material.NewSelect("Platform Hub", []material.SelectOption{
			{Label: "Berlin-Edge", Value: "be"},
			{Label: "Tokyo-Mesh", Value: "tm"},
		}, nil).Render()))

	div.Call("appendChild", renderShowcase("Menu", "Contextual actions list.",
		`trigger := material.NewButton("Open Menu", "outline", nil).Render()
m := material.NewMenu(trigger, []material.MenuItem{{Label: "Action 1"}, {Label: "Action 2"}})`,
		func() js.Value {
			trigger := material.NewButton("Open Node Menu", "outline", nil).Render()
			material.NewMenu(trigger, []material.MenuItem{
				{Label: "Sync Neural Link"},
				{Label: "Reset Cold Storage"},
				{Label: "Export Mesh Logs"},
			})
			return trigger
		}()))

	return div
}

// --- Category: Navigation ---
func renderNavGroup() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 4rem; padding: 2rem 0;")

	div.Call("appendChild", renderShowcase("Tabs", "Content segmentation.",
		`tabs := material.NewTabs([]material.Tab{{Label: "A", Content: divA}})`,
		material.NewTabs([]material.Tab{
			{Label: "Active Cluster", Content: document.Call("createElement", "div")},
			{Label: "Archive Data", Content: document.Call("createElement", "div")},
		}).Render()))

	div.Call("appendChild", renderShowcase("Breadcrumbs", "Hierarchical path tracking.",
		`bc := material.NewBreadcrumbs([]material.Breadcrumb{{Label: "Home", Link: "#home"}})`,
		material.NewBreadcrumbs([]material.Breadcrumb{
			{Label: "Root", Link: "#home"},
			{Label: "System", Link: "#"},
			{Label: "Navigation", Link: "#"},
		}).Render()))

	div.Call("appendChild", renderShowcase("Stepper", "Multi-step process flow.",
		`step := material.NewStepper([]material.StepperStep{{Label: "Auth", Content: div}})`,
		material.NewStepper([]material.StepperStep{
			{Label: "Neural Auth", Content: document.Call("createElement", "div")},
			{Label: "Node Selection", Content: document.Call("createElement", "div")},
			{Label: "Mesh Sync", Content: document.Call("createElement", "div")},
		}).Render()))

	div.Call("appendChild", renderShowcase("Tree View", "Directory and object hierarchies.",
		`tree := material.Tree{Nodes: []material.TreeNode{{Label: "src", Children: [...]}}}`,
		func() js.Value {
			t := material.Tree{Nodes: []material.TreeNode{
				{Label: "Gollemer_OS", Children: []material.TreeNode{
					{Label: "bin", Children: []material.TreeNode{{Label: "kernel.wasm"}}},
					{Label: "assets", Children: []material.TreeNode{{Label: "theme.css"}}},
				}},
			}}
			return t.Render()
		}()))

	div.Call("appendChild", renderShowcase("Pagination", "Data set segmentation.",
		`pag := material.NewPagination(1, 10, func(p int) { ... })`,
		material.NewPagination(3, 8, nil).Render()))

	div.Call("appendChild", renderShowcase("Drawer (Side Nav)", "Slide-out navigation menu.",
		`dr := material.NewDrawer(content)
dr.Open()`,
		material.NewButton("Trigger Side Drawer", "outline", func() {
			sc := document.Call("createElement", "div")
			sc.Set("innerHTML", "<h2>Mesh Explorer</h2><p>Select a cluster to begin synchronization.</p>")
			material.NewDrawer(sc).Open()
		}).Render()))

	return div
}

// --- Category: Data & Info ---
func renderDataGroup() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 4rem; padding: 2rem 0;")

	div.Call("appendChild", renderShowcase("Card", "Flexible information container.",
		`card := material.NewCard("Title", "Subtitle", "Body contents...")`,
		material.NewCard("Omega Cluster Status", "Hub: North Atlantic", "Current throughput is 12.4 TB/s with 99.98% delta verification accuracy.").Render()))

	div.Call("appendChild", renderShowcase("Data Table", "Structured record display.",
		`cols := []material.DataTableColumn{{Header: "Node", Key: "n"}}
dt := material.NewDataTable(cols, data)`,
		func() js.Value {
			cols := []material.DataTableColumn{{Header: "Cluster", Key: "c"}, {Header: "Load", Key: "l"}}
			data := []map[string]interface{}{{"c": "Alpha-7", "l": "42%"}, {"c": "Sigma-2", "l": "89%"}}
			return material.NewDataTable(cols, data).Render()
		}()))

	div.Call("appendChild", renderShowcase("Accordion", "Expandable content headers.",
		`acc := material.NewAccordion([]material.AccordionItem{{Title: "Info", Content: div}})`,
		material.NewAccordion([]material.AccordionItem{
			{Title: "Technical Specifications", Content: document.Call("createElement", "div")},
			{Title: "Security Audit Logs", Content: document.Call("createElement", "div")},
		}).Render()))

	div.Call("appendChild", renderShowcase("Carousel", "Sliding media showcase.",
		`c := material.NewCarousel([]material.CarouselSlide{{ImageURL: "...", Caption: "..."}})`,
		material.NewCarousel([]material.CarouselSlide{
			{ImageURL: "https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=400", Caption: "Mesh Visualization"},
			{ImageURL: "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=400", Caption: "Secure Link"},
		}).Render()))

	div.Call("appendChild", renderShowcase("List", "Simple vertical item stack.",
		`l := material.NewList([]material.ListItem{{Label: "CPU", SubLabel: "90%"}})`,
		material.NewList([]material.ListItem{
			{Label: "Storage Efficiency", SubLabel: "95% Compression"},
			{Label: "Neural Latency", SubLabel: "0.1ms Avg"},
		}).Render()))

	div.Call("appendChild", renderShowcase("Badge & Chip", "Status and tag markers.",
		`b := material.NewBadge("LIVE", "success")
c := material.NewChip("GoLang", "primary")`,
		func() js.Value {
			row := document.Call("createElement", "div")
			row.Set("style", "display: flex; gap: 1rem; align-items: center;")
			row.Call("appendChild", material.NewBadge("ACTIVE", "success").Render())
			row.Call("appendChild", material.NewChip("Compiled-WASM", "primary").Render())
			return row
		}()))

	return div
}

// --- Category: Interactions ---
func renderInteractionGroup() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 4rem; padding: 2rem 0;")

	div.Call("appendChild", renderShowcase("Modal", "Forced focus dialog window.",
		`m := material.NewModal("Danger", content, nil).Render()`,
		material.NewButton("Activate Protocol Modal", "primary", func() {
			p := document.Call("createElement", "p")
			p.Set("innerText", "You are about to synchronize the neural link across all active clusters.")
			material.NewModal("Mesh Synchronization", p, nil).Render()
		}).Render()))

	div.Call("appendChild", renderShowcase("SnackBar (Toast)", "Transient bottom notification.",
		`material.ShowSnackBar("Signal Lost", "RETRY", 3000)`,
		material.NewButton("Transmit Signal Toast", "outline", func() {
			material.ShowSnackBar("Node-01 Response: OK (Latency 0.2ms)", "DISMISS", 3*time.Second)
		}).Render()))

	div.Call("appendChild", renderShowcase("Bottom Sheet", "Mobile-optimized slide-up menu.",
		`material.NewBottomSheet(content).Render()`,
		material.NewButton("Reveal System Sheet", "primary", func() {
			sc := document.Call("createElement", "div")
			sc.Set("innerHTML", "<h2>Node Preferences</h2><p>Adjust neural weighting for current session.</p>")
			material.NewBottomSheet(sc).Render()
		}).Render()))

	div.Call("appendChild", renderShowcase("Alert", "Prominent inline status banner.",
		`a := material.NewAlert("System Update Ready", "info", true)`,
		material.NewAlert("A critical vulnerability in the neural link layer has been patched. Please restart your local mesh node.", "warning", true).Render()))

	div.Call("appendChild", renderShowcase("Tooltip", "Informational hover popup.",
		`material.NewTooltip(target, "Instructional text")`,
		func() js.Value {
			target := material.NewButton("Hover Over Neural Gateway", "outline", nil).Render()
			material.NewTooltip(target, "Endpoint ID: f472b6-mesh-tokyo")
			return target
		}()))

	return div
}

// --- Category: State & Progress ---
func renderStateGroup() js.Value {
	document := js.Global().Get("document")
	div := document.Call("createElement", "div")
	div.Set("style", "display: flex; flex-direction: column; gap: 4rem; padding: 2rem 0;")

	div.Call("appendChild", renderShowcase("ProgressBar", "Linear progress visualization.",
		`pb := material.NewProgressBar(75, "determinate")`,
		material.NewProgressBar(68, "determinate").Render()))

	div.Call("appendChild", renderShowcase("Spinner", "Indeterminate process cycle.",
		`s := material.NewSpinner("md")`,
		func() js.Value {
			row := document.Call("createElement", "div")
			row.Set("style", "display: flex; gap: 2rem; align-items: center;")
			row.Call("appendChild", material.NewSpinner("sm").Render())
			row.Call("appendChild", material.NewSpinner("md").Render())
			row.Call("appendChild", material.NewSpinner("lg").Render())
			return row
		}()))

	div.Call("appendChild", renderShowcase("Skeleton Loader", "Placeholder for loading content.",
		`sk := material.NewSkeleton("circle", "50px", "50px")`,
		func() js.Value {
			col := document.Call("createElement", "div")
			col.Set("style", "display: flex; flex-direction: column; gap: 1rem; width: 100%;")
			col.Call("appendChild", material.NewSkeleton("circle", "48px", "48px").Render())
			col.Call("appendChild", material.NewSkeleton("text", "100%", "24px").Render())
			col.Call("appendChild", material.NewSkeleton("text", "80%", "24px").Render())
			return col
		}()))

	div.Call("appendChild", renderShowcase("Parallax (Unit)", "Image-based depth container.",
		`p := material.NewParallax(img, "300px", content)`,
		func() js.Value {
			box := document.Call("createElement", "div")
			box.Set("innerHTML", "<h3>Depth Module</h3>")
			c := material.NewParallax("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=600", "200px", box)
			return c.Render()
		}()))

	return div
}

// --- Internal Page Builders ---

func renderShowcase(title, desc, code string, demo js.Value) js.Value {
	document := js.Global().Get("document")
	card := document.Call("createElement", "div")
	card.Set("className", "mat-card")
	card.Set("style", "padding: 0; overflow: hidden; border: 1px solid var(--border);")

	wrapper := document.Call("createElement", "div")
	wrapper.Set("style", "padding: 3rem;")

	h3 := document.Call("createElement", "h3")
	h3.Set("innerText", title)
	h3.Set("style", "margin-top: 0; font-size: 2rem; margin-bottom: 0.5rem;")
	wrapper.Call("appendChild", h3)

	p := document.Call("createElement", "p")
	p.Set("innerText", desc)
	p.Set("className", "text-muted")
	p.Set("style", "margin-bottom: 2.5rem; font-size: 1.1rem;")
	wrapper.Call("appendChild", p)

	// Live Demo Box
	demoContainer := document.Call("createElement", "div")
	demoContainer.Set("style", "margin-bottom: 2.5rem; padding: 4rem; background: #000; border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: center; min-height: 200px; position: relative;")
	demoContainer.Call("appendChild", demo)
	wrapper.Call("appendChild", demoContainer)

	// Code Box
	codeHeader := document.Call("createElement", "div")
	codeHeader.Set("innerText", "Go Integration")
	codeHeader.Set("style", "font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.15em; color: var(--primary); margin-bottom: 1.25rem; font-weight: 800;")
	wrapper.Call("appendChild", codeHeader)

	pre := document.Call("createElement", "pre")
	pre.Set("style", "background: #000; padding: 2rem; border-radius: 20px; color: #a5b4fc; font-family: 'Fira Code', monospace; font-size: 0.95rem; overflow-x: auto; margin: 0; border: 1px solid rgba(255,255,255,0.03); line-height: 1.6;")
	pre.Set("innerText", code)
	wrapper.Call("appendChild", pre)

	card.Call("appendChild", wrapper)
	return card
}
