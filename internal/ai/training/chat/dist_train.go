package chat

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"net"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/moe"
)

var (
	syncMutex        sync.Mutex
	workerConn       net.Conn
	workerConnMutex  sync.Mutex
)

// Global or structural variable to track expected cluster size
const ExpectedWorkers = 1 // Since there's 1 master and 1 worker node in this scenario

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

// StartMaster starts a TCP server that listens for incoming model weights from workers.
// It blocks until the expected number of workers connect.
func StartMaster(model *moe.IntentMoE, addr string) error {
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

	runMasterSyncServer(listener, model)
	
	return nil
}

func runMasterSyncServer(listener net.Listener, model *moe.IntentMoE) {
	var wg sync.WaitGroup
	workersConnected := 0

	log.Printf("🌐 [Distributed] Waiting for %d workers to join the cluster...", ExpectedWorkers)

	for workersConnected < ExpectedWorkers {
		conn, err := listener.Accept()
		if err != nil {
			log.Printf("❌ [Distributed] Registration error: %v", err)
			continue
		}

		workersConnected++
		log.Printf("🌐 [Distributed] Worker %d/%d connected from %s", workersConnected, ExpectedWorkers, conn.RemoteAddr().String())
		
		// Handle individual worker streams/sync loops in separate goroutines
		wg.Add(1)
		go handleWorkerSync(conn, &wg, model)
	}

	// This is the crucial barrier: Do not let the master proceed to local training steps
	log.Printf("✅ [Distributed] All expected workers checked in. Synchronizing topologies...")
}

func handleWorkerSync(conn net.Conn, wg *sync.WaitGroup, model *moe.IntentMoE) {
	defer wg.Done()
	defer conn.Close()

	for {
		// Read number of weights
		var numWeights int32
		if err := binary.Read(conn, binary.LittleEndian, &numWeights); err != nil {
			if err != io.EOF {
				log.Printf("⚠️  [Distributed] Connection error with worker %s: %v", conn.RemoteAddr(), err)
			}
			return
		}

		// Read weights
		receivedWeights := make([]float32, numWeights)
		if err := binary.Read(conn, binary.LittleEndian, &receivedWeights); err != nil {
			log.Printf("⚠️  [Distributed] Failed to read weights from worker %s: %v", conn.RemoteAddr(), err)
			return
		}

		// Average weights
		syncMutex.Lock()
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
		syncMutex.Unlock()

		log.Printf("🌐 [Distributed] Master successfully received and averaged %d weights from worker at %s.", len(receivedWeights), conn.RemoteAddr())
		
		// Send ACK
		ack := []byte("OK")
		conn.Write(ack)
	}
}

// SyncWithMaster sends the current model weights to the master node.
func SyncWithMaster(model *moe.IntentMoE, masterAddr string) {
	workerConnMutex.Lock()
	if workerConn == nil {
		conn, err := net.Dial("tcp", masterAddr)
		if err != nil {
			log.Printf("⚠️  [Distributed] Worker failed to connect to master %s: %v", masterAddr, err)
			workerConnMutex.Unlock()
			return
		}
		workerConn = conn
		log.Printf("🌐 [Distributed] Worker connected to master at %s", masterAddr)
	}
	conn := workerConn
	workerConnMutex.Unlock()

	syncMutex.Lock()
	params := model.Parameters()
	var totalWeights int32
	for _, p := range params {
		totalWeights += int32(len(p.Data))
	}

	flatWeights := make([]float32, 0, totalWeights)
	for _, p := range params {
		flatWeights = append(flatWeights, p.Data...)
	}
	syncMutex.Unlock()

	// Write number of weights
	if err := binary.Write(conn, binary.LittleEndian, totalWeights); err != nil {
		log.Printf("⚠️  [Distributed] Worker failed to send total weights: %v", err)
		workerConn.Close()
		workerConnMutex.Lock()
		workerConn = nil
		workerConnMutex.Unlock()
		return
	}

	// Write weights
	if err := binary.Write(conn, binary.LittleEndian, flatWeights); err != nil {
		log.Printf("⚠️  [Distributed] Worker failed to send weights: %v", err)
		workerConn.Close()
		workerConnMutex.Lock()
		workerConn = nil
		workerConnMutex.Unlock()
		return
	}

	// Wait for ACK
	ack := make([]byte, 2)
	if _, err := io.ReadFull(conn, ack); err != nil {
		log.Printf("⚠️  [Distributed] Worker failed to receive ACK from master: %v", err)
		workerConn.Close()
		workerConnMutex.Lock()
		workerConn = nil
		workerConnMutex.Unlock()
		return
	}

	log.Printf("🌐 [Distributed] Worker successfully synced %d weights with master.", totalWeights)
}
