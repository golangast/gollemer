package ui

import (
	"fmt"
	"strings"
	"syscall/js"
)

// IOBViewer represents a UI component for displaying IOB (Inside-Outside-Beginning) tagging results.
type IOBViewer struct {
	Tokens []string
	Tags   []string
}

// NewIOBViewer creates a new IOBViewer.
func NewIOBViewer(tokens []string, tags []string) *IOBViewer {
	return &IOBViewer{Tokens: tokens, Tags: tags}
}

// Render generates a styled HTML div element to visualize IOB results.
// This implements a high-contrast, labeled view for NER/Slot filling results.
func (v *IOBViewer) Render() js.Value {
	div := js.Global().Get("document").Call("createElement", "div")
	div.Set("className", "iob-viewer-container")
	
	var htmlBuilder strings.Builder
	htmlBuilder.WriteString("<div class='iob-viewer' style='display:flex; flex-wrap:wrap; gap:8px; padding:16px; background:#fafafa; border-radius:8px;'>")

	// Consistent color mapping for tag types as requested
	tagColors := map[string]string{
		"B-": "background-color: #ffe082; border-bottom: 2px solid #fbc02d; padding: 2px 6px; border-radius: 4px; display: inline-flex; flex-direction: column; align-items: center;", // Beginning
		"I-": "background-color: #fff8e1; border-bottom: 2px solid #fff176; padding: 2px 6px; border-radius: 4px; display: inline-flex; flex-direction: column; align-items: center;", // Inside
		"O":  "background-color: #f5f5f5; padding: 2px 6px; border-radius: 4px; display: inline-flex; flex-direction: column; align-items: center; color: #757575;",                     // Outside
	}

	for i, token := range v.Tokens {
		tag := v.Tags[i]
		
		var prefix string
		if len(tag) >= 2 {
			prefix = tag[:2]
		} else {
			prefix = "O"
		}
		
		var style string
		if prefix == "O" || prefix == "O-" {
			style = tagColors["O"]
		} else {
			style = tagColors[prefix]
		}
		
		label := ""
		if prefix != "O" && len(tag) > 2 {
			label = strings.TrimPrefix(tag, prefix)
		} else if prefix == "O" {
			label = "Other"
		}

		// Build the token chunk with its tag label in CSS/HTML
		chunkHTML := fmt.Sprintf(
			"<span class='iob-chunk' style='%s' title='Tag: %s'>" +
				"<span class='iob-token' style='font-family: monospace; font-weight: bold;'>%s</span>" +
				"<span class='iob-tag' style='font-size: 0.7em; text-transform: uppercase; color: #424242;'>%s</span>" +
			"</span>", style, tag, token, label)
			
		htmlBuilder.WriteString(chunkHTML)
	}

	htmlBuilder.WriteString("</div>")
	div.Set("innerHTML", htmlBuilder.String())
	return div
}
