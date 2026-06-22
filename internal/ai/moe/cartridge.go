package moe

import (
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

// CartridgeManager handles dynamic loading and unloading of expert layers.
type CartridgeManager struct {
	requests chan CartridgeRequest
	Mu       sync.Mutex
	Loaded   map[string]Expert
}

// NewCartridgeManager initializes a CartridgeManager.
func NewCartridgeManager() *CartridgeManager {
	cm := &CartridgeManager{
		requests: make(chan CartridgeRequest, 10),
		Loaded:   make(map[string]Expert),
	}
	go cm.loop()
	return cm
}

func (cm *CartridgeManager) loop() {
	for req := range cm.requests {
		switch req.Action {
		case "load":
			cm.Mu.Lock()
			if _, exists := cm.Loaded[req.Path]; exists {
				cm.Mu.Unlock()
				req.Response <- nil
				continue
			}
			cm.Mu.Unlock()
			
			expert, err := cm.loadFromFile(req.Path)
			if err != nil {
				req.Response <- err
				continue
			}
			
			cm.Mu.Lock()
			cm.Loaded[req.Path] = expert
			cm.Mu.Unlock()
			log.Printf("🎮 Cartridge Manager: Loaded cartridge %s into RAM.", req.Path)
			req.Response <- nil
		case "unload":
			cm.Mu.Lock()
			if _, exists := cm.Loaded[req.Path]; exists {
				delete(cm.Loaded, req.Path)
				log.Printf("🎮 Cartridge Manager: UnLoaded cartridge %s from RAM to save memory.", req.Path)
			}
			cm.Mu.Unlock()
			req.Response <- nil
		}
	}
}

func (cm *CartridgeManager) loadFromFile(path string) (Expert, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open cartridge %s: %v", path, err)
	}
	defer file.Close()
	
	var expert Expert
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&expert); err != nil {
		return nil, fmt.Errorf("failed to decode cartridge %s: %v", path, err)
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
