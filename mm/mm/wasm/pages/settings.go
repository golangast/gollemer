package pages

import (
	"syscall/js"

	"mm/wasm/ui/material"
)

func RenderSettings() js.Value {
	document := js.Global().Get("document")
	container := document.Call("createElement", "div")
	container.Set("className", "container")

	section := document.Call("createElement", "section")
	section.Set("style", "max-width: 800px; margin: 0 auto;")

	// Header
	header := document.Call("createElement", "div")
	header.Set("style", "margin-bottom: 3rem;")
	header.Set("innerHTML", "<h1>System Settings</h1><p class='text-muted'>Configure your neural interface and mesh preferences.</p>")
	section.Call("appendChild", header)

	// Settings Card
	card := document.Call("createElement", "div")
	card.Set("className", "mat-card")
	card.Set("style", "padding: 0;")

	// Settings List
	list := document.Call("createElement", "div")
	list.Set("style", "display: flex; flex-direction: column;")

	createSetting := func(title, desc string, control js.Value) js.Value {
		row := document.Call("createElement", "div")
		row.Set("style", "padding: 2rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border);")
		
		info := document.Call("createElement", "div")
		info.Set("style", "display: flex; flex-direction: column; gap: 0.25rem;")
		
		t := document.Call("createElement", "div")
		t.Set("innerText", title)
		t.Set("style", "font-weight: 600; font-size: 1.1rem;")
		
		d := document.Call("createElement", "div")
		d.Set("innerText", desc)
		d.Set("className", "text-muted")
		d.Set("style", "font-size: 0.875rem;")
		
		info.Call("appendChild", t)
		info.Call("appendChild", d)
		
		row.Call("appendChild", info)
		row.Call("appendChild", control)
		return row
	}

	list.Call("appendChild", createSetting("Neural Acceleration", "Enhance processing speed at the cost of higher thermal output.", material.NewSwitch("turbo", true, nil).Render()))
	list.Call("appendChild", createSetting("Dark Pulse Mode", "Invert luminosity for deep-space environments.", material.NewSwitch("dark", true, nil).Render()))
	list.Call("appendChild", createSetting("Mesh Encryption", "Apply RSA-4096 to all outbound neural signals.", material.NewSwitch("crypto", true, nil).Render()))
	
	prioritySelect := material.NewSelect("Execution Priority", []material.SelectOption{
		{Label: "Low Latency", Value: "1"},
		{Label: "High Throughput", Value: "2"},
		{Label: "Balanced", Value: "3"},
	}, nil).Render()
	prioritySelect.Set("style", "width: 200px;")
	list.Call("appendChild", createSetting("Network Strategy", "Choose how data is prioritized across the mesh.", prioritySelect))

	card.Call("appendChild", list)
	section.Call("appendChild", card)

	// Action Bar
	actions := document.Call("createElement", "div")
	actions.Set("style", "margin-top: 3rem; display: flex; gap: 1rem; justify-content: flex-end;")
	
	saveBtn := material.NewButton("Apply Changes", "primary", func() {
		material.ShowSnackBar("Settings synchronized across mesh.", "REBOOT", 3000)
	})
	actions.Call("appendChild", material.NewButton("Factory Reset", "outline", nil).Render())
	actions.Call("appendChild", saveBtn.Render())
	
	section.Call("appendChild", actions)

	container.Call("appendChild", section)
	return container
}
