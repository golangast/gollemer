package moe

import (
	"sync/atomic"
	"testing"

	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func TestRuntimeTelemetryTraceBuffer(t *testing.T) {
	rt := NewRuntimeTelemetry()
	rt.RecordTrace("router", "dispatch", map[string]interface{}{"expert": 2})

	snap := rt.TraceSnapshot()
	if len(snap) != 1 {
		t.Fatalf("expected one trace event, got %d", len(snap))
	}
	if snap[0].Category != "router" {
		t.Fatalf("expected router category, got %q", snap[0].Category)
	}
	if snap[0].Message != "dispatch" {
		t.Fatalf("expected dispatch message, got %q", snap[0].Message)
	}
}

func TestRuntimeTelemetryLeakDetector(t *testing.T) {
	rt := NewRuntimeTelemetry()
	rt.StartLeakDetector(1, 2)
	defer rt.StopLeakDetector()
	if rt.LeakDetector == nil {
		t.Fatal("expected leak detector to be initialized")
	}
	if !rt.LeakDetector.Enabled {
		t.Fatal("expected leak detector to be enabled")
	}
}

func TestRuntimeTelemetryMathSandbox(t *testing.T) {
	rt := NewRuntimeTelemetry()
	result := rt.RunMathSandbox("sanity", 2, 2, 2)
	if result == nil {
		t.Fatal("expected sandbox result")
	}
	if result["match"] != true {
		t.Fatalf("expected SIMD and fallback to match, got %#v", result)
	}
}

func TestRuntimeTelemetrySupervisorTimeline(t *testing.T) {
	rt := NewRuntimeTelemetry()
	rt.RecordSupervisorAdjustment(500, "plateau", "lr", "0.001", "0.002")
	entries := rt.SupervisorTimeline()
	if len(entries) != 1 {
		t.Fatalf("expected one supervisor adjustment, got %d", len(entries))
	}
	if entries[0].Reason != "plateau" {
		t.Fatalf("expected plateau reason, got %q", entries[0].Reason)
	}
}

func TestRuntimeTelemetrySerializationMetrics(t *testing.T) {
	rt := NewRuntimeTelemetry()
	rt.RecordSerializationMetrics("checkpoint", map[string]interface{}{"alloc_delta": 128})
	stats := rt.SerializationSnapshot()
	if stats == nil || stats.Label != "checkpoint" {
		t.Fatalf("expected checkpoint snapshot, got %#v", stats)
	}
}

func TestSupervisor_PlateauGuardSkipsSurgery(t *testing.T) {
	s := NewSupervisor()
	s.BestPerplexity = 8.5
	s.PlateauCount = 4

	if !s.ShouldSkipExpertSurgery() {
		t.Fatal("expected plateaued training to skip autonomous expert surgery")
	}
}

func TestMoELayer_ResetExpertWeightsWarmupSuppressed(t *testing.T) {
	layer, err := NewMoELayer(8, 8, 2, 1, func(i int) (Expert, error) {
		return NewFeedForwardExpert(8, 16, 8)
	})
	if err != nil {
		t.Fatalf("failed to create layer: %v", err)
	}
	layer.CurrentPhase = 1
	layer.AccumulatedUtilization = []int{0, 0}

	layer.ResetExpertWeights(1)
	if got := atomic.LoadInt32(&layer.ResetCount); got != 0 {
		t.Fatalf("expected warmup reset to be suppressed, got reset count %d", got)
	}
}

func TestSupervisor_Interventions(t *testing.T) {
	// Initialize a small vocabulary
	vocab := mainvocab.NewVocabulary()
	vocab.AddToken("what")
	vocab.AddToken("who")
	vocab.AddToken("you")

	// Initialize model
	vocabSize := vocab.Size()
	embeddingDim := 64
	numExperts := 8
	model, err := NewHybridIntentMoE(vocabSize, embeddingDim, numExperts, 5, 5, vocabSize, 2, nil)
	if err != nil {
		t.Fatalf("Failed to create HybridIntentMoE: %v", err)
	}
	model.SentenceVocab = vocab

	// Initialize supervisor
	supervisor := NewSupervisor()

	// 1. Test SpawnSpecializedExpert
	// We want to spawn expert ID 25 on Layer 0.
	supervisor.SpawnSpecializedExpert(model, 0, "IDENTITY", 25)

	layer0 := model.Encoder.GetMoELayers()[0]
	if len(layer0.Experts) < 26 {
		t.Errorf("Expected at least 26 experts in layer 0 after spawning specialized expert, got %d", len(layer0.Experts))
	}

	roleName := layer0.ExpertRole[25]
	if roleName != "IDENTITY" {
		t.Errorf("Expected expert 25 role to be 'IDENTITY', got '%s'", roleName)
	}

	// 2. Test AdjustRoutingAffinity
	// Before adjusting, check weight alignment/value
	weightsBefore := layer0.GatingNetwork.Linear.Weights.Data[0*layer0.GatingNetwork.Linear.Weights.Shape[1]+25]

	supervisor.AdjustRoutingAffinity(model, "what", 25, 2.5)

	weightsAfter := layer0.GatingNetwork.Linear.Weights.Data[0*layer0.GatingNetwork.Linear.Weights.Shape[1]+25]
	if weightsBefore == weightsAfter {
		t.Errorf("Expected routing weights to change after AdjustRoutingAffinity, but they remained equal (value = %f)", weightsAfter)
	}

	// 3. Test ClearFailureLogs
	supervisor.FailureLogs["E16"] = 10
	supervisor.ClearFailureLogs(model)

	if supervisor.FailureLogs["E16"] != 0 {
		t.Errorf("Expected FailureLogs to be cleared, but E16 still has count %d", supervisor.FailureLogs["E16"])
	}
}
