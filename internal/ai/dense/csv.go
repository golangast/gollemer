package dense

import (
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"io"
	"os"
	"strings"

	trainingpb "github.com/golangast/gollemer/internal/ai/training/proto"
)

// LoadCommandExamplesFromCSV reads CommandExample records from a CSV file.
func LoadCommandExamplesFromCSV(path string) ([]CommandExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1
	reader.LazyQuotes = true

	var records [][]string
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			data, readErr := os.ReadFile(path)
			if readErr != nil {
				return nil, fmt.Errorf("read csv: %w", err)
			}
			backtickRecords, backtickErr := parseBacktickCSV(string(data))
			if backtickErr != nil {
				return nil, fmt.Errorf("parse csv: %w", err)
			}
			records = backtickRecords
			break
		}
		records = append(records, record)
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("empty csv file")
	}
	records = records[1:]

	var examples []CommandExample
	for i, rec := range records {
		if len(rec) < 4 {
			return nil, fmt.Errorf("record %d: csv record has %d fields, want 4: %v", i+2, len(rec), rec)
		}
		examples = append(examples, CommandExample{
			Type:      strings.TrimSpace(rec[0]),
			Prompt:    strings.TrimSpace(rec[1]),
			Response:  strings.TrimSpace(rec[2]),
			CodeAfter: strings.TrimSpace(rec[3]),
		})
	}
	return examples, nil
}

func parseBacktickCSV(content string) ([][]string, error) {
	var records [][]string
	var fields []string
	var current strings.Builder
	inBacktick := false

	flushField := func() {
		fields = append(fields, current.String())
		current.Reset()
	}
	flushRecord := func() {
		flushField()
		records = append(records, fields)
		fields = nil
	}

	for i := 0; i < len(content); i++ {
		c := content[i]
		switch {
		case c == '`':
			inBacktick = !inBacktick
		case c == ',' && !inBacktick:
			flushField()
		case c == '\n' && !inBacktick:
			flushRecord()
		case c == '\r' && !inBacktick:
			// skip carriage returns outside backticks
		default:
			current.WriteByte(c)
		}
	}
	if current.Len() > 0 || len(fields) > 0 {
		flushRecord()
	}
	if inBacktick {
		return nil, fmt.Errorf("unterminated backtick in csv")
	}
	return records, nil
}

// CommandDatasetFromCSV builds a Dataset from a CSV file of CommandExamples.
func CommandDatasetFromCSV(path string, seed int64) (*Dataset, error) {
	examples, err := LoadCommandExamplesFromCSV(path)
	if err != nil {
		return nil, err
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("no command examples found in %s", path)
	}

	samples := make([]Sample, len(examples))
	for i, c := range examples {
		samples[i] = Sample{
			Input: BagOfWords(c.Prompt, CommandVocab),
			Label: LabelForCommand(c.Type),
		}
	}
	return NewDataset(seed, samples...), nil
}

// CommandDatasetFromProto builds a Dataset from a protobuf file of CommandExamples.
func CommandDatasetFromProto(path string, seed int64) (*Dataset, error) {
	examples, err := LoadCommandExamplesFromProto(path)
	if err != nil {
		return nil, err
	}
	if len(examples) == 0 {
		return nil, fmt.Errorf("no command examples found in %s", path)
	}

	samples := make([]Sample, len(examples))
	for i, c := range examples {
		samples[i] = Sample{
			Input: BagOfWords(c.Prompt, CommandVocab),
			Label: LabelForCommand(c.Type),
		}
	}
	return NewDataset(seed, samples...), nil
}

// LoadCommandExamplesFromProto reads CommandExample records from a protobuf file.
func LoadCommandExamplesFromProto(path string) ([]CommandExample, error) {
	pbExamples, err := trainingpb.LoadCommandExamplesFromProto(path)
	if err != nil {
		return nil, err
	}
	examples := make([]CommandExample, len(pbExamples))
	for i, e := range pbExamples {
		examples[i] = CommandExample{
			Type:      e.Type,
			Prompt:    e.UserPrompt,
			Response:  e.AssistantResponse,
			CodeAfter: e.CodeAfter,
		}
	}
	return examples, nil
}

// SaveGob serializes the DenseModel to a gob file.
func (m *DenseModel) SaveGob(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create gob file: %w", err)
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	if err := enc.Encode(m); err != nil {
		return fmt.Errorf("encode model: %w", err)
	}
	return nil
}

// LoadGob deserializes a DenseModel from a gob file.
func LoadGob(path string) (*DenseModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open gob file: %w", err)
	}
	defer f.Close()

	var m DenseModel
	dec := gob.NewDecoder(f)
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("decode model: %w", err)
	}
	return &m, nil
}
