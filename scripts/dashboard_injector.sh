#!/usr/bin/env bash
# Simple injector that posts synthetic metrics to the observability server
# Usage: METRICS_HOST=localhost:8080 bash scripts/dashboard_injector.sh

METRICS_HOST=${METRICS_HOST:-localhost:8080}
INTERVAL=${INTERVAL:-1}

echo "Starting dashboard injector -> http://${METRICS_HOST} (interval=${INTERVAL}s)"

while true; do
  payload=$(python3 - <<PY
import json, random
def rnd(a,b,d=4): return round(random.uniform(a,b),d)
vels = {f"layer_{i}": rnd(0.0,1.0,6) for i in range(4)}
maxv = max(vels.values()) if len(vels)>0 else 0.0
norm = {k: round((v/maxv) if maxv>0 else 0.0,6) for k,v in vels.items()}
payload = {
  "weight_velocity": {"velocities": vels, "normalized": norm, "max_velocity": round(maxv,6)},
  "category_loss": {
    "Core Verbs": {"average_loss": rnd(0.2,1.2), "sample_count": random.randint(10,200), "improvement": rnd(-0.2,0.2), "improvement_rate": round(random.uniform(-0.2,0.5),3)},
    "Technical Terms": {"average_loss": rnd(0.1,1.0), "sample_count": random.randint(5,100), "improvement": rnd(-0.1,0.3), "improvement_rate": round(random.uniform(-0.1,0.4),3)}
  }
}
print(json.dumps(payload))
PY
)

  # POST payload
  curl -s -X POST "http://${METRICS_HOST}/api/metrics/inject_metrics" -H 'Content-Type: application/json' -d "$payload" >/dev/null || true

  # Also inject a small expert lexicon occasionally
  if [ $((RANDOM%5)) -eq 0 ]; then
    lex=$(python3 - <<PY
import json, random
lex={str(i): [random.choice(["hello","world","fox","dog","model","token","routing","expert","test"]) for _ in range(3)] for i in range(4)}
print(json.dumps(lex))
PY
)
    curl -s -X POST "http://${METRICS_HOST}/api/metrics/inject" -H 'Content-Type: application/json' -d "$lex" >/dev/null || true
  fi

  sleep ${INTERVAL}
done
