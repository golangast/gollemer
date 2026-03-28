#!/bin/bash
# Gollemer Ignition Script - v1.0

# 1. Clear the noise
clear

# 2. Run the Gollemer Audit & Trace
# For this script to work fully, the main.go should handle --mode and --trace flags
# Since we are in the middle of development, we'll simulate the startup pulse
go run main.go --mode=audit --trace=inputTensor

# 3. Display the latest Journal Entry
echo -e "\n--- [ LATEST JOURNAL ENTRY ] ---"
# Check if the journal exists before tailing
LOG_FILE="logs/daily_$(date +%Y-%m-%d).md"
if [ -f "$LOG_FILE" ]; then
    tail -n 15 "$LOG_FILE"
else
    echo "No journal entry for today yet. Let's build something great first!"
fi

# 4. Open the environment
echo -e "\nHi, welcome to gollemer!"
