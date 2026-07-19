package moe

import (
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func TestSemanticDriftTrackerReportsDriftAfterSuccessiveSnapshots(t *testing.T) {
	tracker := NewSemanticDriftTracker(nil, vocab.NewVocabulary())

	first := tensor.NewTensor([]int{2, 64}, make([]float32, 128), false)
	for i := range first.Data {
		first.Data[i] = float32(i%2) * 0.5
	}
	tracker.RecordEmbeddingState(first)

	second := tensor.NewTensor([]int{2, 64}, make([]float32, 128), false)
	for i := range second.Data {
		second.Data[i] = float32((i+1)%2) * 0.5
	}
	tracker.RecordEmbeddingState(second)

	drifts := tracker.GetTopSemanticShifts(3)
	if len(drifts) == 0 {
		t.Fatalf("expected semantic drift entries, got none")
	}
	if driftVal, ok := drifts[0]["drift"].(float32); !ok || driftVal <= 0 {
		t.Fatalf("expected a positive drift value, got %#v", drifts[0]["drift"])
	}
}

func TestRecordStepMapsBatchExpertIDsAcrossTokenIDs(t *testing.T) {
	obs := NewMoEObservability(4, 32, vocab.NewVocabulary(), nil)
	obs.RecordStep([]int{0, 1}, []int{10, 11, 12, 13}, 0.12)

	lexicon := obs.ExpertLexicon.GetTopKTokensPerExpert(10, vocab.NewVocabulary())
	usedExperts := 0
	for expertID, tokens := range lexicon {
		if len(tokens) > 0 {
			usedExperts++
			if expertID < 0 || expertID >= 4 {
				t.Fatalf("unexpected expert id %d in lexicon", expertID)
			}
		}
	}

	if usedExperts == 0 {
		t.Fatalf("expected routing data to populate expert lexicon, got %#v", lexicon)
	}
}

func TestMetricsAggregatorStoresHealthIndicatorsInCurrentMetrics(t *testing.T) {
	v := vocab.NewVocabulary()
	trainer := &Trainer{}
	trainer.InitializeObservability(4, 32, v, nil)

	agg := NewMetricsAggregator(trainer, v)
	metrics := agg.CollectMetrics()

	if _, ok := metrics["health_indicators"].(map[string]interface{}); !ok {
		t.Fatalf("expected health indicators in collected metrics, got %#v", metrics["health_indicators"])
	}

	current := agg.GetCurrentMetrics()
	if _, ok := current["health_indicators"].(map[string]interface{}); !ok {
		t.Fatalf("expected health indicators in current metrics history, got %#v", current["health_indicators"])
	}
}
