#!/usr/bin/env bash
# Watch training log for loss stagnation and emit alerts to logs/monitor_alerts.log
LOG=logs/train_resume.log
ALERT_LOG=logs/monitor_alerts.log
STATE=/tmp/gollemer_loss_state.tmp
THRESHOLD=${1:-50} # stagnant epochs threshold
TAIL_LINES=2000

mkdir -p logs

# Initialize state
if [ ! -f "$STATE" ]; then
  echo "best_loss=1e9" > "$STATE"
  echo "stagnant=0" >> "$STATE"
fi

while true; do
  # extract recent Loss entries
  vals=$(tail -n $TAIL_LINES "$LOG" 2>/dev/null | grep -E "Phase .*Loss:" | awk '{for(i=1;i<=NF;i++){if($i=="Loss:") print $(i+1)}}')
  if [ -z "$vals" ]; then
    sleep 10
    continue
  fi
  # get latest loss
  latest=$(echo "$vals" | tail -n1)
  best=$(grep "^best_loss=" "$STATE" | cut -d'=' -f2)
  stagnant=$(grep "^stagnant=" "$STATE" | cut -d'=' -f2)

  # compare
  awk -v lat="$latest" -v best="$best" -v thr="$THRESHOLD" -v STATE="$STATE" -v ALERT="$ALERT_LOG" 'BEGIN{
    lat=lat+0; best=best+0
  }
  {
  }
  END{
    if(lat < best - 0.0001) {
      # improvement
      cmd = "sed -i 's/^best_loss=.*/best_loss=" lat "\n/; s/^stagnant=.*/stagnant=0\n/' " STATE
      system(cmd)
      print strftime("%Y/%m/%d %H:%M:%S"), "LOSS IMPROVE ->", lat >> ALERT
    } else {
      # stagnation increment
      newst = (stagnant+1)
      cmd = "sed -i 's/^stagnant=.*/stagnant=" newst "\n/' " STATE
      system(cmd)
      if(newst >= thr) {
        print strftime("%Y/%m/%d %H:%M:%S"), "ALERT: loss stagnant for", newst, "epochs (latest=", lat, ")" >> ALERT
        # reset counter so we don't spam
        cmd2 = "sed -i 's/^stagnant=.*/stagnant=0\n/' " STATE
        system(cmd2)
      }
    }
  }'

  sleep 10
done
