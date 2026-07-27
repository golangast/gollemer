package llm

import (
	"fmt"
	"os"
	"testing"
)

func TestFindWhisperBinaryAndModelEnv(t *testing.T) {
	os.Setenv("WHISPER_CLI_BIN", "/home/zendrulat/g/gollemer/build_whisper/whisper.cpp/build/bin/whisper-cli")
	os.Setenv("WHISPER_MODEL_PATH", "/home/zendrulat/g/gollemer/build_whisper/whisper.cpp/models/ggml-tiny.en.bin")
	bin, model, err := findWhisperBinaryAndModel()
	fmt.Printf("bin=%q model=%q err=%v\n", bin, model, err)
}
