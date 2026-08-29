// Command dense_train trains the simplified dense MLP on the basic
// go_edit_agent update-command corpus (social chit-chat + code_update
// commands) and saves the model as a gob file for the dense_llm inference CLI.
//
// Usage:
//
//	go run ./cmd/tools/dense_train \
//	  -data=data/training/command_examples.pb \
//	  -model=data/models/dense/model.gob \
//	  -epochs=300 -batch=4 -lr=0.05
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/dense"
)

func main() {
	dataPath := flag.String("data", "data/training/command_examples.pb", "path to protobuf training data (type,prompt,response,code_after)")
	modelPath := flag.String("model", "data/models/dense/model.gob", "output path for gob model file")
	epochs := flag.Int("epochs", 300, "number of training epochs")
	batch := flag.Int("batch", 8, "minibatch size")
	lr := flag.Float64("lr", 0.05, "base learning rate (cosine decay to min-lr)")
	minLR := flag.Float64("min-lr", 0.001, "minimum learning rate at end of cosine decay")
	warmup := flag.Int("warmup", 10, "linear warmup steps")
	logEvery := flag.Int("log-every", 10, "log a line every N steps")
	flag.Parse()

	// Load the training corpus from protobuf (or CSV for backward compatibility).
	var ds *dense.Dataset
	var err error
	if strings.HasSuffix(*dataPath, ".pb") {
		ds, err = dense.CommandDatasetFromProto(*dataPath, 42)
	} else {
		ds, err = dense.CommandDatasetFromCSV(*dataPath, 42)
	}
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}

	model := dense.NewDenseModel(ds.FeatureSize(), []int{16}, ds.NumClasses())
	trainer := dense.NewTrainer(model, dense.Config{
		Epochs:      *epochs,
		BatchSize:   *batch,
		BaseLR:      float32(*lr),
		MinLR:       float32(*minLR),
		WarmupSteps: *warmup,
		LogEvery:    *logEvery,
	})

	if err := os.MkdirAll(filepath.Dir(*modelPath), 0755); err != nil {
		log.Fatalf("create model dir: %v", err)
	}

	fmt.Printf("🔨 Training dense MLP on command corpus from %s (%d samples, %d features -> [16] -> %d classes)\n",
		*dataPath, len(ds.Samples), ds.FeatureSize(), ds.NumClasses())
	loss, err := trainer.Train(ds)
	if err != nil {
		log.Fatalf("training failed: %v", err)
	}

	if err := model.SaveGob(*modelPath); err != nil {
		log.Fatalf("save gob model: %v", err)
	}
	fmt.Printf("💾 Model saved to %s (final loss %.6f)\n", *modelPath, loss)
}
