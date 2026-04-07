package main

import (
	"log"
)

// SamplePlugin implements the Plugin interface.
type SamplePlugin struct{}

func (p *SamplePlugin) Name() string {
	return "SamplePlugin"
}

func (p *SamplePlugin) Init(svc *Service) error {
	log.Printf("🔌 [SamplePlugin] Initializing... (Svc Port: %s)", svc.Port)
	return nil
}

func init() {
	// In a real multi-file project, you'd ensure this registers 
	// specifically when the package is loaded.
	RegisterPlugin(&SamplePlugin{})
}
