//go:build ignore

// convert_small_demo_to_proto reads the legacy small_social_demo.csv fixture
// (query,answer,intent,grammar) and writes an equivalent dataset.proto
// ConversationDataset protobuf file (small_social_demo.pb) so the small
// training path can load conversations from protobuf instead of CSV.
//
// Usage:
//
//	go run scripts/convert_small_demo_to_proto.go \
//	    -in data/training/trainingdata/small_social_demo.csv \
//	    -out data/training/trainingdata/small_social_demo.pb
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	datasetpb "github.com/golangast/gollemer/internal/ai/training/proto/dataset"
)

func main() {
	inPath := flag.String("in", "data/training/trainingdata/small_social_demo.csv", "input CSV path (query,answer,intent,grammar)")
	outPath := flag.String("out", "data/training/trainingdata/small_social_demo.pb", "output protobuf ConversationDataset path")
	flag.Parse()

	f, err := os.Open(*inPath)
	if err != nil {
		log.Fatalf("open %s: %v", *inPath, err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	rows, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("read %s: %v", *inPath, err)
	}
	if len(rows) < 2 {
		log.Fatalf("%s has no data rows", *inPath)
	}

	ds := &datasetpb.ConversationDataset{}
	for i, row := range rows {
		if i == 0 {
			continue // header
		}
		if len(row) < 2 {
			continue
		}
		q := strings.TrimSpace(row[0])
		a := strings.TrimSpace(row[1])
		if q == "" || a == "" {
			continue
		}
		conv := &datasetpb.Conversation{
			ConversationId: fmt.Sprintf("demo-%d", i),
			Turns: []*datasetpb.ConversationTurn{
				{TurnSequence: 1, Role: datasetpb.Role_ROLE_USER, Content: q},
				{TurnSequence: 2, Role: datasetpb.Role_ROLE_ASSISTANT, Content: a},
			},
		}
		ds.Conversations = append(ds.Conversations, conv)
	}

	if err := datasetpb.SaveConversationDatasetToProto(*outPath, ds); err != nil {
		log.Fatalf("save %s: %v", *outPath, err)
	}
	log.Printf("wrote %d conversations to %s", len(ds.Conversations), *outPath)
}
