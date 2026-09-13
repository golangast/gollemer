#!/bin/bash
# Append goz to main.go, fix imports, and add -fuzzy flag
cat ./cmd/tools/goz/main.go > /tmp/goz.go
