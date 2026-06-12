package chat

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
)

var (
	syncMutex sync.Mutex
)

// resolveListenAddr normalises a user-supplied address into a valid
// "host:port" string suitable for net.Listen / http.ListenAndServe.
//
//	":8080"          → ":8080"          (already a listen address)
//	"0.0.0.0:8080"   → "0.0.0.0:8080"  (already a host:port)
//	"8080"           → ":8080"          (bare port – prepend colon)
func resolveListenAddr(addr string) string {
	if strings.Contains(addr, ":") {
		return addr // already host:port or :port — use as-is
	}
	return ":" + addr // bare port number
}

// StartMaster starts an HTTP server that listens for incoming model weights from workers.
// It averages the received weights with its own weights.
// addr may be a full listen address (":8080", "0.0.0.0:8080") or a bare port ("8080").
func StartMaster(model *moe.IntentMoE, addr string) error {
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

		workerIP := r.RemoteAddr
		log.Printf("🌐 [Distributed] Master successfully received and averaged %d weights from worker at %s.", len(receivedWeights), workerIP)
		w.WriteHeader(http.StatusOK)
	})

	listenAddr := resolveListenAddr(addr)
	
	// Helper to extract port for display
	displayPort := listenAddr
	if strings.HasPrefix(displayPort, ":") {
		displayPort = displayPort[1:]
	} else if parts := strings.Split(displayPort, ":"); len(parts) == 2 {
		displayPort = parts[1]
	}

	log.Printf("\n===================================================================")
	log.Printf("🌐 [Distributed] Master Node is Active & Listening!")
	log.Printf("📌 Bound Address : %s", listenAddr)
	log.Printf("🛠️  To connect a worker, run the same training command on the worker node")
	log.Printf("   with the following flags appended:")
	log.Printf("   -dist-mode=worker -dist-addr=<MASTER_IP>:%s", displayPort)
	log.Printf("   (Replace <MASTER_IP> with this machine's actual network IP if on another device)")
	log.Printf("===================================================================\n")
	
	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return fmt.Errorf("failed to bind master server to %s: %v", listenAddr, err)
	}

	go func() {
		if err := http.Serve(listener, nil); err != nil {
			log.Printf("⚠️  [Distributed] Master server error: %v", err)
		}
	}()
	
	return nil
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
