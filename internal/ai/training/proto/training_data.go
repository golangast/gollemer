package trainingpb

import (
	"fmt"
	"os"

	"google.golang.org/protobuf/proto"
)

// SaveCommandExamplesToProto serializes a slice of CommandExample to a protobuf file.
func SaveCommandExamplesToProto(path string, examples []*CommandExample) error {
	set := &CommandExampleSet{Examples: examples}
	data, err := proto.Marshal(set)
	if err != nil {
		return fmt.Errorf("marshal command examples: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write protobuf file: %w", err)
	}
	return nil
}

// LoadCommandExamplesFromProto reads a protobuf file containing CommandExampleSet.
func LoadCommandExamplesFromProto(path string) ([]*CommandExample, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read protobuf file: %w", err)
	}
	var set CommandExampleSet
	if err := proto.Unmarshal(data, &set); err != nil {
		return nil, fmt.Errorf("unmarshal command examples: %w", err)
	}
	return set.Examples, nil
}

// SaveConversationsToProto serializes a slice of Conversation to a protobuf file.
func SaveConversationsToProto(path string, conversations []*Conversation) error {
	set := &ConversationSet{Conversations: conversations}
	data, err := proto.Marshal(set)
	if err != nil {
		return fmt.Errorf("marshal conversations: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("write protobuf file: %w", err)
	}
	return nil
}

// LoadConversationsFromProto reads a protobuf file containing ConversationSet.
func LoadConversationsFromProto(path string) ([]*Conversation, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read protobuf file: %w", err)
	}
	var set ConversationSet
	if err := proto.Unmarshal(data, &set); err != nil {
		return nil, fmt.Errorf("unmarshal conversations: %w", err)
	}
	return set.Conversations, nil
}
