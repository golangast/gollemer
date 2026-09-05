#!/usr/bin/env bash
# Simple watcher for the overfit run loss (Phase 1).
LOG="logs/train_overfit_run.log"
ALERT="logs/overfit_alerts.log"
TH1=0.01
TH2=0.002
notified1=0
notified2=0

mkdir -p logs
echo "$(date +'%Y/%m/%d %H:%M:%S') watcher started" >> "$ALERT"

while true; do
  if [ -f "$LOG" ]; then
    line=$(grep "Phase 1" "$LOG" | tail -n1)
    if [ -n "$line" ]; then
      loss=$(echo "$line" | sed -n 's/.*Loss: \([0-9.]*\.?[0-9]*\).*/\1/p')
      if [ -n "$loss" ]; then
        lvl=$(awk -v l="$loss" -v t1="$TH1" -v t2="$TH2" 'BEGIN{ if(l<=t2) print 2; else if(l<=t1) print 1; else print 0 }')
        ts=$(date +'%Y/%m/%d %H:%M:%S')
        if [ "$lvl" -eq 2 ] && [ "$notified2" -eq 0 ]; then
          echo "$ts ALERT: loss <= $TH2 (loss=$loss)" | tee -a "$ALERT"
          notified2=1; notified1=1
        elif [ "$lvl" -eq 1 ] && [ "$notified1" -eq 0 ]; then
          echo "$ts NOTICE: loss <= $TH1 (loss=$loss)" | tee -a "$ALERT"
          notified1=1
        fi
      fi
    fi
  fi
  sleep 10
done
#!/usr/bin/env bash
LOG=logs/train_overfit.log
ALERT_LOG=logs/monitor_overfit_alerts.log
THRESH1=0.01
THRESH2=0.002
COOLDOWN=300
LAST_ALERT_TIME=0

mkdir -p logs

echo "Starting overfit watcher: $(date)" >> "$ALERT_LOG"

while true; do
  # get latest loss
  latest=$(tail -n 500 "$LOG" 2>/dev/null | grep -E "Phase .*Loss:" | awk '{for(i=1;i<=NF;i++) if($i=="Loss:") print $(i+1)}' | tail -n1)
  if [ -z "$latest" ]; then
    sleep 5
    continue
  fi
  # convert to float
  latestf=$(printf "%f" "$latest" 2>/dev/null || echo "NaN")
  now=$(date +%s)
  if [ "$latestf" != "NaN" ]; then
    # check thresholds
    if (( $(echo "$latestf < $THRESH2" | bc -l) )); then
      if (( now - LAST_ALERT_TIME > COOLDOWN )); then
        echo "$(date) ALERT: loss <$THRESH2 : $latestf" | tee -a "$ALERT_LOG"
        LAST_ALERT_TIME=$now
      fi
    elif (( $(echo "$latestf < $THRESH1" | bc -l) )); then
      if (( now - LAST_ALERT_TIME > COOLDOWN )); then
        echo "$(date) NOTICE: loss <$THRESH1 : $latestf" | tee -a "$ALERT_LOG"
        LAST_ALERT_TIME=$now
      fi
    fi
  fi
  sleep 10
done
