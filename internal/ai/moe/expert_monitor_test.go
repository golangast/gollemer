package moe

import (
	"strings"
	"testing"
)

func TestExpertMonitorHeatmapASCII(t *testing.T) {
	monitor := NewExpertMonitor(3)
	monitor.LogSelections([]int{0, 0, 1, 2})

	heatmap := monitor.HeatmapASCII("layer-0")
	if !strings.Contains(heatmap, "layer-0") {
		t.Fatalf("expected heatmap to include title, got %q", heatmap)
	}
	if !strings.Contains(heatmap, "E0") {
		t.Fatalf("expected heatmap to include expert labels, got %q", heatmap)
	}
}
