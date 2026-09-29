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
