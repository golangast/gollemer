package moe

import (
	"encoding/binary"
	"encoding/gob"
	"fmt"
	"log"
	"os"
	"sync"
)

// CartridgeRequest represents a request to load or unload an expert cartridge.
type CartridgeRequest struct {
	Action    string // "load" or "unload"
	Path      string
	LayerIdx  int
	ExpertIdx int // For unloading
	RoleID    int
	Response  chan error
}

// CartridgeHeader defines the strict spec for .cartridge files.
type CartridgeHeader struct {
	Magic     [8]byte  // "GLMR_CRT"
	Version   uint32   // Engine version
	Namespace [32]byte // Intent namespace
	InputDim  uint32
	HiddenDim uint32
	OutputDim uint32
}

// CartridgeManager handles dynamic loading and unloading of expert layers.
type CartridgeManager struct {
	requests chan CartridgeRequest
	mu       sync.Mutex
	Loaded   map[string]Expert

	// LRU Cache & Centralized Context Pager
	lruOrder     []string
	MaxWarm      int
	ContextPager map[string][]float32
}

// Global slice pool to prevent GC trashing when swapping cartridges.
var slicePool = sync.Pool{
	New: func() interface{} {
		return make([]float32, 0)
	},
}

func getSlice(size int) []float32 {
	s := slicePool.Get().([]float32)
	if cap(s) < size {
		return make([]float32, size)
	}
	return s[:size]
}

func putSlice(s []float32) {
	if s == nil {
		return
	}
	// Clear references if necessary, but it's just float32
	slicePool.Put(s[:0])
}

func NewCartridgeManager() *CartridgeManager {
	cm := &CartridgeManager{
		requests:     make(chan CartridgeRequest, 10),
		Loaded:       make(map[string]Expert),
		MaxWarm:      3, // Default to keeping 3 cartridges warm in memory
		ContextPager: make(map[string][]float32),
	}
	go cm.loop()
	return cm
}

func (cm *CartridgeManager) loop() {
	for req := range cm.requests {
		switch req.Action {
		case "load":
			cm.mu.Lock()
			if _, exists := cm.Loaded[req.Path]; exists {
				cm.markUsed(req.Path)
				cm.mu.Unlock()
				if GlobalTelemetry != nil {
					GlobalTelemetry.RecordPoolHit()
					GlobalTelemetry.RecordTrace("cartridge", "pool_hit", map[string]interface{}{"path": req.Path})
				}
				req.Response <- nil
				continue
			}
			if GlobalTelemetry != nil {
				GlobalTelemetry.RecordPoolMiss()
				GlobalTelemetry.RecordTrace("cartridge", "pool_miss", map[string]interface{}{"path": req.Path})
			}
			cm.mu.Unlock()

			expert, err := cm.loadFromFile(req.Path)
			if err != nil {
				req.Response <- err
				continue
			}

			cm.mu.Lock()
			cm.Loaded[req.Path] = expert

			// Re-inflate context if we have paged it out previously
			if ctx, exists := cm.ContextPager[req.Path]; exists {
				expert.RestoreContext(ctx)
				log.Printf("🧠 Cartridge Manager: Re-inflated memory context for %s", req.Path)
			}

			cm.markUsed(req.Path)
			cm.evictLRU()
			cm.mu.Unlock()
			if GlobalTelemetry != nil {
				GlobalTelemetry.SetWarmCartridges(len(cm.Loaded))
				GlobalTelemetry.RecordTrace("cartridge", "loaded", map[string]interface{}{"path": req.Path, "warm": len(cm.Loaded)})
			}
			_ = EmitRuntimeTelemetry("logs/telemetry.json", nil)
			log.Printf("🎮 Cartridge Manager: Loaded cartridge %s into RAM.", req.Path)
			req.Response <- nil
		case "unload":
			cm.mu.Lock()
			if expert, exists := cm.Loaded[req.Path]; exists {
				// Page out context
				cm.ContextPager[req.Path] = expert.GetContext()

				cm.recycleExpert(expert)
				delete(cm.Loaded, req.Path)
				cm.removeFromLRU(req.Path)
				log.Printf("🎮 Cartridge Manager: UnLoaded cartridge %s from RAM. Context paged out.", req.Path)
			}
			cm.mu.Unlock()
			if GlobalTelemetry != nil {
				GlobalTelemetry.SetWarmCartridges(len(cm.Loaded))
				GlobalTelemetry.RecordTrace("cartridge", "unloaded", map[string]interface{}{"path": req.Path, "warm": len(cm.Loaded)})
			}
			_ = EmitRuntimeTelemetry("logs/telemetry.json", nil)
			req.Response <- nil
		}
	}
}

func (cm *CartridgeManager) markUsed(path string) {
	cm.removeFromLRU(path)
	cm.lruOrder = append(cm.lruOrder, path)
}

func (cm *CartridgeManager) removeFromLRU(path string) {
	for i, p := range cm.lruOrder {
		if p == path {
			cm.lruOrder = append(cm.lruOrder[:i], cm.lruOrder[i+1:]...)
			break
		}
	}
}

func (cm *CartridgeManager) evictLRU() {
	for len(cm.Loaded) > cm.MaxWarm && len(cm.lruOrder) > 0 {
		oldest := cm.lruOrder[0]
		if expert, exists := cm.Loaded[oldest]; exists {
			// Page out context before eviction
			cm.ContextPager[oldest] = expert.GetContext()

			cm.recycleExpert(expert)
			delete(cm.Loaded, oldest)
			log.Printf("🧹 Cartridge Manager: LRU Evicted %s. Context paged out to Supervisor buffer.", oldest)
		}
		cm.lruOrder = cm.lruOrder[1:]
	}
}

func (cm *CartridgeManager) recycleExpert(expert Expert) {
	// Reclaim memory to sync.Pool
	if ffe, ok := expert.(*FeedForwardExpert); ok {
		if ffe.Layer1 != nil && ffe.Layer1.Weights != nil {
			putSlice(ffe.Layer1.Weights.Data)
			if ffe.Layer1.Biases != nil {
				putSlice(ffe.Layer1.Biases.Data)
			}
		}
		if ffe.Layer2 != nil && ffe.Layer2.Weights != nil {
			putSlice(ffe.Layer2.Weights.Data)
			if ffe.Layer2.Biases != nil {
				putSlice(ffe.Layer2.Biases.Data)
			}
		}
	}
}

func (cm *CartridgeManager) loadFromFile(path string) (Expert, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open cartridge %s: %v", path, err)
	}
	defer file.Close()

	var header CartridgeHeader
	if err := binary.Read(file, binary.LittleEndian, &header); err != nil {
		return nil, fmt.Errorf("failed to read cartridge header %s: %v", path, err)
	}

	if string(header.Magic[:]) != "GLMR_CRT" {
		// Fallback to Gob decoder for backward compatibility
		file.Seek(0, 0)
		var expert Expert
		decoder := gob.NewDecoder(file) // Note: Requires importing "encoding/gob" inside function or globally
		if err := decoder.Decode(&expert); err != nil {
			return nil, fmt.Errorf("failed to decode gob cartridge %s: %v", path, err)
		}
		return expert, nil
	}

	// Zero-copy read into pooled buffers
	expert, err := NewFeedForwardExpert(int(header.InputDim), int(header.HiddenDim), int(header.OutputDim))
	if err != nil {
		return nil, err
	}

	// Allocate from pool
	expert.Layer1.Weights.Data = getSlice(len(expert.Layer1.Weights.Data))
	if err := binary.Read(file, binary.LittleEndian, expert.Layer1.Weights.Data); err != nil {
		return nil, err
	}

	if expert.Layer1.Biases != nil {
		expert.Layer1.Biases.Data = getSlice(len(expert.Layer1.Biases.Data))
		if err := binary.Read(file, binary.LittleEndian, expert.Layer1.Biases.Data); err != nil {
			return nil, err
		}
	}

	expert.Layer2.Weights.Data = getSlice(len(expert.Layer2.Weights.Data))
	if err := binary.Read(file, binary.LittleEndian, expert.Layer2.Weights.Data); err != nil {
		return nil, err
	}

	if expert.Layer2.Biases != nil {
		expert.Layer2.Biases.Data = getSlice(len(expert.Layer2.Biases.Data))
		if err := binary.Read(file, binary.LittleEndian, expert.Layer2.Biases.Data); err != nil {
			return nil, err
		}
	}

	return expert, nil
}

// LoadCartridge sends a request to load a cartridge.
func (cm *CartridgeManager) LoadCartridge(path string, layerIdx, roleID int) error {
	resp := make(chan error, 1)
	cm.requests <- CartridgeRequest{
		Action:   "load",
		Path:     path,
		LayerIdx: layerIdx,
		RoleID:   roleID,
		Response: resp,
	}
	return <-resp
}

// UnloadCartridge sends a request to unload a cartridge.
func (cm *CartridgeManager) UnloadCartridge(path string) error {
	resp := make(chan error, 1)
	cm.requests <- CartridgeRequest{
		Action:   "unload",
		Path:     path,
		Response: resp,
	}
	return <-resp
}

// PreloadCartridge sends a request to load a cartridge asynchronously.
// This enables zero-latency loading while the user is still typing.
func (cm *CartridgeManager) PreloadCartridge(path string, layerIdx, roleID int) {
	go func() {
		resp := make(chan error, 1)
		cm.requests <- CartridgeRequest{
			Action:   "load",
			Path:     path,
			LayerIdx: layerIdx,
			RoleID:   roleID,
			Response: resp,
		}
		// Wait for completion but ignore error for preload
		<-resp
	}()
}

// Lock manually locks the CartridgeManager.
func (cm *CartridgeManager) Lock() {
	cm.mu.Lock()
}

// Unlock manually unlocks the CartridgeManager.
func (cm *CartridgeManager) Unlock() {
	cm.mu.Unlock()
}
