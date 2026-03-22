#!/bin/bash

# Navigate to the multi_orchestrator directory
pushd cmd/multi_orchestrator > /dev/null

# Build the multi_orchestrator executable
echo "Building multi_orchestrator..."
go build -o multi_orchestrator main.go

# Run the multi_orchestrator without the -llm flag
echo "Running multi_orchestrator without -llm flag..."
./multi_orchestrator

# Return to the original directory
popd > /dev/null

echo "Script finished."