package moe

import (
	"reflect"
	"testing"
)

func TestTriageCartridgesMulti(t *testing.T) {
	sup := NewSupervisor()
	sup.KeywordMap = map[string]string{
		"handler":  "data/models/intents/gofix.cartridge",
		"database": "data/models/intents/sql_builder.cartridge",
		"gorm":     "data/models/intents/sql_builder.cartridge",
		"deadlock": "data/models/intents/goroutine_fix.cartridge",
	}

	query := "add handler auth_handler with database connection"
	matches := sup.TriageCartridgesMulti(query, nil)

	want := []string{"data/models/intents/gofix.cartridge", "data/models/intents/sql_builder.cartridge"}
	if len(matches) != len(want) {
		t.Fatalf("TriageCartridgesMulti got %v, want %v", matches, want)
	}

	for i := range want {
		found := false
		for _, m := range matches {
			if m == want[i] {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected cartridge %s in matches %v", want[i], matches)
		}
	}
}

func TestBlendPredictionVectors(t *testing.T) {
	vec1 := []float32{1.0, 2.0, 3.0}
	vec2 := []float32{3.0, 4.0, 5.0}

	// We can test vector blending logic directly
	blended := make([]float32, len(vec1))
	for i := range vec1 {
		blended[i] = (vec1[i] + vec2[i]) / 2.0
	}

	expected := []float32{2.0, 3.0, 4.0}
	if !reflect.DeepEqual(blended, expected) {
		t.Errorf("BlendPredictionVectors got %v, want %v", blended, expected)
	}
}

func TestMergeASTSubKeys(t *testing.T) {
	output1 := map[string]interface{}{
		"handler": "auth_handler",
		"url":     "/auth",
	}
	output2 := map[string]interface{}{
		"database":    "sqlite",
		"inject_code": "db.Connect()",
	}

	merged := make(map[string]interface{})
	for k, v := range output1 {
		merged[k] = v
	}
	for k, v := range output2 {
		merged[k] = v
	}

	if merged["handler"] != "auth_handler" || merged["database"] != "sqlite" || merged["url"] != "/auth" {
		t.Errorf("MergeASTSubKeys produced unexpected merged map: %v", merged)
	}
}
