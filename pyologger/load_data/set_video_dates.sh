#!/usr/bin/env bash
set -euo pipefail

for f in *.mp4; do
  # last YYYY-MM-DD_HH-MM-SS in filename = start timestamp
  datetime_raw=$(
    printf '%s' "$f" |
      grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}' |
      tail -n1
  )

  if [[ -n "${datetime_raw:-}" ]]; then
    # SetFile wants: MM/DD/YYYY HH:MM:SS
    setfile_date=$(date -j -f "%Y-%m-%d_%H-%M-%S" "$datetime_raw" "+%m/%d/%Y %H:%M:%S")
    # touch -t wants: YYYYMMDDHHMM.SS
    touch_date=$(date -j -f "%Y-%m-%d_%H-%M-%S" "$datetime_raw" "+%Y%m%d%H%M.%S")

    echo "Setting file date to '$setfile_date' (touch '$touch_date') for '$f'"
    SetFile -d "$setfile_date" "$f"
    touch -t "$touch_date" "$f"
  else
    echo "❌ Could not extract datetime from: $f"
  fi
done