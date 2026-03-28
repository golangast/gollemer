.PHONY: all build test bench train verify clean run

all: build

build:
	/usr/local/go/bin/go build -o bin/gollemer ./cmd/gollemer

# Run your SIMD benchmarks to ensure no performance regression
bench:
	/usr/local/go/bin/go test -bench=. -benchmem ./internal/ai/moe/...

# Train and then immediately verify
run:
	/usr/local/go/bin/go run cmd/train/main.go
	/usr/local/go/bin/go run cmd/verify/main.go

# Clean up binary checkpoints and artifacts
clean:
	rm -rf checkpoints/*.bin
	rm -f *.out *.test cpu.prof mem.prof
	rm -f bin/gollemer

# Launch the shell training script
train:
	bash scripts/train.sh
