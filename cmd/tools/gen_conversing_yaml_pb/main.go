// gen_conversing_yaml_pb converts data/training/trainingdata/conversing.yaml
// into the datasetpb.ConversationDataset protobuf at
// data/training/trainingdata/conversing.pb.
//
// The YAML format mirrors the conversing.csv columns:
//
//	conversations:
//	  - conversation_id: "conv_457"
//	    turns:
//	      - turn_sequence: 1
//	        role: "user"
//	        content: "..."
//	      - turn_sequence: 2
//	        role: "assistant"
//	        content: |
//	          [PREDICTIVE_REASONING]
//	          ...
//	          [RESPONSE] ...
//
// Usage:
//
//	make conversing-pb
//	# or directly:
//	go run ./cmd/tools/gen_conversing_yaml_pb [-in path] [-out path]
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	datasetpb "github.com/golangast/gollemer/internal/ai/training/proto/dataset"
	"gopkg.in/yaml.v3"
)

// yamlConversation mirrors the conversing.yaml structure.
type yamlConversation struct {
	ConversationID string     `yaml:"conversation_id"`
	Turns          []yamlTurn `yaml:"turns"`
}

type yamlTurn struct {
	TurnSequence   int32  `yaml:"turn_sequence"`
	Role           string `yaml:"role"`
	Content        string `yaml:"content"`
	ReasoningTrace string `yaml:"reasoning_trace,omitempty"`
}

type yamlRoot struct {
	Conversations []yamlConversation `yaml:"conversations"`
}

func roleFromString(s string) (datasetpb.Role, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "system":
		return datasetpb.Role_ROLE_SYSTEM, nil
	case "user":
		return datasetpb.Role_ROLE_USER, nil
	case "assistant":
		return datasetpb.Role_ROLE_ASSISTANT, nil
	default:
		return datasetpb.Role_ROLE_UNSPECIFIED, fmt.Errorf("unknown role %q", s)
	}
}

func main() {
	inPath := flag.String("in", "data/training/trainingdata/conversing.yaml", "input YAML path")
	outPath := flag.String("out", "data/training/trainingdata/conversing.pb", "output protobuf path")
	flag.Parse()

	data, err := os.ReadFile(*inPath)
	if err != nil {
		log.Fatalf("read %s: %v", *inPath, err)
	}

	var root yamlRoot
	if err := yaml.Unmarshal(data, &root); err != nil {
		log.Fatalf("parse yaml: %v", err)
	}
	if len(root.Conversations) == 0 {
		log.Fatalf("no conversations found in %s", *inPath)
	}

	ds := &datasetpb.ConversationDataset{}
	for _, conv := range root.Conversations {
		pbConv := &datasetpb.Conversation{ConversationId: conv.ConversationID}
		for _, turn := range conv.Turns {
			role, err := roleFromString(turn.Role)
			if err != nil {
				log.Fatalf("%s turn %d: %v", conv.ConversationID, turn.TurnSequence, err)
			}
			pbConv.Turns = append(pbConv.Turns, &datasetpb.ConversationTurn{
				TurnSequence:   turn.TurnSequence,
				Role:           role,
				Content:        turn.Content,
				ReasoningTrace: turn.ReasoningTrace,
			})
		}
		ds.Conversations = append(ds.Conversations, pbConv)
	}

	if dir := filepath.Dir(*outPath); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			log.Fatalf("mkdir %s: %v", dir, err)
		}
	}
	if err := datasetpb.SaveConversationDatasetToProto(*outPath, ds); err != nil {
		log.Fatalf("save %s: %v", *outPath, err)
	}
	fmt.Printf("✅ wrote %s (%d conversations) from %s\n", *outPath, len(ds.Conversations), *inPath)
}
