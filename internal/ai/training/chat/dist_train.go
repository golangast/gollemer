package chat

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
)

var (
	syncMutex sync.Mutex
)

// StartMaster starts an HTTP server that listens for incoming model weights from workers.
// It averages the received weights with its own weights.
func StartMaster(model *moe.IntentMoE, port string) {
	http.HandleFunc("/sync-weights", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Invalid method", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "Failed to read body", http.StatusInternalServerError)
			return
		}
		defer r.Body.Close()

		// Convert bytes back to float32 slice
		numWeights := len(body) / 4
		receivedWeights := make([]float32, numWeights)
		buf := bytes.NewReader(body)
		if err := binary.Read(buf, binary.LittleEndian, &receivedWeights); err != nil {
			log.Printf("⚠️  [Distributed] Master failed to decode weights: %v", err)
			http.Error(w, "Failed to decode weights", http.StatusBadRequest)
			return
		}

		// Average weights
		syncMutex.Lock()
		defer syncMutex.Unlock()

		params := model.Parameters()
		idx := 0
		for _, p := range params {
			for i := range p.Data {
				if idx < len(receivedWeights) {
					// Federated Averaging: Master and Worker contribute equally
					p.Data[i] = (p.Data[i] + receivedWeights[idx]) / 2.0
					idx++
				}
			}
		}

		log.Printf("🌐 [Distributed] Master successfully received and averaged %d weights from worker.", len(receivedWeights))
		w.WriteHeader(http.StatusOK)
	})

	addr := ":" + port
	log.Printf("🌐 [Distributed] Master listening for workers on %s", addr)
	go func() {
		if err := http.ListenAndServe(addr, nil); err != nil {
			log.Printf("⚠️  [Distributed] Master server error: %v", err)
		}
	}()
}

// SyncWithMaster sends the current model weights to the master node.
func SyncWithMaster(model *moe.IntentMoE, masterAddr string) {
	syncMutex.Lock()
	params := model.Parameters()
	var totalWeights int
	for _, p := range params {
		totalWeights += len(p.Data)
	}

	flatWeights := make([]float32, 0, totalWeights)
	for _, p := range params {
		flatWeights = append(flatWeights, p.Data...)
	}
	syncMutex.Unlock()

	// Convert float32 slice to byte slice
	buf := new(bytes.Buffer)
	if err := binary.Write(buf, binary.LittleEndian, flatWeights); err != nil {
		log.Printf("⚠️  [Distributed] Worker failed to encode weights: %v", err)
		return
	}

	url := fmt.Sprintf("http://%s/sync-weights", masterAddr)
	req, err := http.NewRequest("POST", url, buf)
	if err != nil {
		log.Printf("⚠️  [Distributed] Worker failed to create request: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("⚠️  [Distributed] Worker failed to sync with master %s: %v", masterAddr, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		log.Printf("🌐 [Distributed] Worker successfully synced %d weights with master.", totalWeights)
	} else {
		log.Printf("⚠️  [Distributed] Worker sync rejected by master, status: %d", resp.StatusCode)
	}
}
