#!/bin/bash
# deploy.sh - Automation script to compile, transfer, and run Gollemer

TARGET_IP="192.168.1.XX" # Replace with your Raspberry Pi's actual IP address
TARGET_USER="pi"

echo "Step 1: Cross-compiling Gollemer binary for ARM64 architecture..."
# Using standard env variables for cross compilation
env GOOS=linux GOARCH=arm64 go build -tags=pi -o gollemer_prod ./cmd/gollemer

echo "Step 2: Pushing production binary and trained model weights to the Pi..."
scp ./gollemer_prod ${TARGET_USER}@${TARGET_IP}:~/gollemer/
scp -r ./data/models/ ${TARGET_USER}@${TARGET_IP}:~/gollemer/data/

echo "Step 3: Launching Gollemer on the target Pi with Real-Time scheduling permissions..."
ssh ${TARGET_USER}@${TARGET_IP} "sudo chown root ~/gollemer/gollemer_prod && sudo chmod +x ~/gollemer/gollemer_prod && sudo chrt -f 90 ~/gollemer/gollemer_prod"

echo "⚡ Deployment Complete. Gollemer is now actively listening for live audio hardware inputs!"
