#!/usr/bin/env bash
# Starts observability backend, injector, and dashboard detached
set -euo pipefail

# Ensure old processes are killed
pkill -f cmd/tools/observability_example/main.go || true
pkill -f scripts/dashboard_injector.sh || true
pkill -f cmd/tools/dashboard/main.go || true

# Start observability
nohup go run cmd/tools/observability_example/main.go > /tmp/observability.log 2>&1 &
echo $! > /tmp/observability.pid

# Start injector
nohup bash scripts/dashboard_injector.sh > /tmp/dashboard_injector.log 2>&1 &
echo $! > /tmp/dashboard_injector.pid

# Start dashboard frontend detached
nohup setsid go run cmd/tools/dashboard/main.go > /tmp/dashboard.log 2>&1 < /dev/null &
echo $! > /tmp/dashboard.pid

sleep 0.2

cat <<EOF
✅ Dashboard started: http://localhost:8765
   Observability: http://localhost:8080/docs/dashboard-enhanced.html
   PIDs - observability=$(cat /tmp/observability.pid 2>/dev/null || echo NA), injector=$(cat /tmp/dashboard_injector.pid 2>/dev/null || echo NA), dashboard=$(cat /tmp/dashboard.pid 2>/dev/null || echo NA)
Logs: /tmp/observability.log /tmp/dashboard_injector.log /tmp/dashboard.log
EOF
