package datasetpb

import (
	"fmt"
	"os"

	"google.golang.org/protobuf/proto"
)

// SaveConversationDatasetToProto serializes a ConversationDataset to a protobuf file.
func SaveConversationDatasetToProto(path string, ds *ConversationDataset) error {
	data, err := proto.Marshal(ds)
	if err != nil {
		return fmt.Errorf("marshal conversation dataset: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write protobuf file: %w", err)
	}
	return nil
}

// LoadConversationDatasetFromProto reads a protobuf file containing a ConversationDataset.
func LoadConversationDatasetFromProto(path string) (*ConversationDataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read protobuf file: %w", err)
	}
	var ds ConversationDataset
	if err := proto.Unmarshal(data, &ds); err != nil {
		return nil, fmt.Errorf("unmarshal conversation dataset: %w", err)
	}
	return &ds, nil
}
