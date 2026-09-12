package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/training/makefile"
	datasetpb "github.com/golangast/gollemer/internal/ai/training/proto/dataset"
	"gopkg.in/yaml.v3"
)

// yamlConversation mirrors the Makefile-generated YAML structure.
type yamlConversation struct {
	ConversationID string     `yaml:"conversation_id"`
	Turns          []yamlTurn `yaml:"turns"`
}

type yamlTurn struct {
	TurnSequence int32  `yaml:"turn_sequence"`
	Role         string `yaml:"role"`
	Content      string `yaml:"content"`
}

type yamlRoot struct {
	Conversations []yamlConversation `yaml:"conversations"`
}

func main() {
	makefilePath := flag.String("makefile", "Makefile", "path to Makefile")
	outPath := flag.String("out", "", "output protobuf path")
	flag.Parse()

	if *outPath == "" {
		base := filepath.Base(*makefilePath)
		*outPath = strings.TrimSuffix(base, filepath.Ext(base)) + ".pb"
	}

	targets, err := makefile.ParseMakefile(makefile.ParseOptions{MakefilePath: *makefilePath})
	if err != nil {
		log.Fatalf("parse makefile: %v", err)
	}
	if len(targets) == 0 {
		log.Fatalf("no targets found in %s", *makefilePath)
	}
	log.Printf("📋 Found %d make targets in %s", len(targets), *makefilePath)

	yamlPath := *outPath
	if ext := filepath.Ext(yamlPath); ext == ".pb" {
		yamlPath = strings.TrimSuffix(yamlPath, ext) + ".yaml"
	}
	if err := makefile.SaveTrainingData(targets, yamlPath); err != nil {
		log.Fatalf("save training yaml: %v", err)
	}
	log.Printf("📝 Wrote training YAML to %s", yamlPath)

	data, err := os.ReadFile(yamlPath)
	if err != nil {
		log.Fatalf("read generated yaml: %v", err)
	}
	var root yamlRoot
	if err := yaml.Unmarshal(data, &root); err != nil {
		log.Fatalf("parse generated yaml: %v", err)
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
				TurnSequence: turn.TurnSequence,
				Role:         role,
				Content:      turn.Content,
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
	fmt.Printf("✅ wrote %s (%d conversations) from %s\n", *outPath, len(ds.Conversations), *makefilePath)
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
