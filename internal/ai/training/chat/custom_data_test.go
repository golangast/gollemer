package chat

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadCustomSocialDatasetUsesSmallDemoCSV(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..")
	path := filepath.Join(root, "data", "training", "trainingdata", "small_social_demo.csv")
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected small demo csv at %s: %v", path, err)
	}

	pairs, err := loadCustomSocialPairs(path)
	if err != nil {
		t.Fatalf("loadCustomSocialPairs returned error: %v", err)
	}
	if len(pairs) != 6 {
		t.Fatalf("loadCustomSocialPairs() = %d pairs, want 6", len(pairs))
	}
	if pairs[0].Intent != "greeting" {
		t.Fatalf("first pair intent = %q, want %q", pairs[0].Intent, "greeting")
	}
}

func TestIsSmallDemoDataset(t *testing.T) {
	if !isSmallDemoDataset("/tmp/small_social_demo.csv") {
		t.Fatal("isSmallDemoDataset returned false for the tiny demo CSV")
	}
	if !isSmallDemoDataset("/tmp/small_social_demo.pb") {
		t.Fatal("isSmallDemoDataset returned false for the tiny demo protobuf")
	}
	if isSmallDemoDataset("/tmp/other.csv") {
		t.Fatal("isSmallDemoDataset returned true for a non-demo dataset")
	}
}

// TestLoadSmallSocialDatasetFromProtoMatchesCSV verifies that the protobuf
// ConversationDataset fixture (small_social_demo.pb) produces the same
// Q/A/Intent training pairs as the legacy CSV fixture it was generated from,
// via scripts/convert_small_demo_to_proto.go.
func TestLoadSmallSocialDatasetFromProtoMatchesCSV(t *testing.T) {
	root := filepath.Join("..", "..", "..", "..")
	csvPath := filepath.Join(root, "data", "training", "trainingdata", "small_social_demo.csv")
	pbPath := filepath.Join(root, "data", "training", "trainingdata", "small_social_demo.pb")

	if _, err := os.Stat(pbPath); err != nil {
		t.Fatalf("expected small demo protobuf at %s: %v", pbPath, err)
	}

	csvPairs, err := loadCustomSocialPairs(csvPath)
	if err != nil {
		t.Fatalf("loadCustomSocialPairs returned error: %v", err)
	}
	pbPairs, err := loadCustomSocialPairsAny(pbPath)
	if err != nil {
		t.Fatalf("loadCustomSocialPairsAny returned error: %v", err)
	}

	if len(pbPairs) != len(csvPairs) {
		t.Fatalf("proto loader returned %d pairs, want %d (from CSV)", len(pbPairs), len(csvPairs))
	}
	for i := range csvPairs {
		if pbPairs[i].Q != csvPairs[i].Q {
			t.Fatalf("pair %d: Q = %q, want %q", i, pbPairs[i].Q, csvPairs[i].Q)
		}
		if pbPairs[i].A != csvPairs[i].A {
			t.Fatalf("pair %d: A = %q, want %q", i, pbPairs[i].A, csvPairs[i].A)
		}
		if pbPairs[i].Intent != csvPairs[i].Intent {
			t.Fatalf("pair %d: Intent = %q, want %q", i, pbPairs[i].Intent, csvPairs[i].Intent)
		}
	}
}
