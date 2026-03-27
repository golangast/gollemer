package metrics

import (
	"net/http"
	"os"
	"path/filepath"
)

func Metrics(w http.ResponseWriter, r *http.Request) {
	cwd, _ := os.Getwd()
	metricsPath := filepath.Join(cwd, "logs", "latest_metric.json")
	
	data, err := os.ReadFile(metricsPath)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"error": "metrics not available"}`))
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Write(data)
}
