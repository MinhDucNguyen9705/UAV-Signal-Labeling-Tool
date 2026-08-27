#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$DIR"

PORT=8000
echo "Checking if port $PORT is currently occupied..."
PID=$(lsof -ti :$PORT 2>/dev/null || fuser $PORT/tcp 2>/dev/null || true)
if [ -n "$PID" ]; then
  echo "Stopping existing process on port $PORT (PID $PID)..."
  kill -9 $PID 2>/dev/null || true
  sleep 1
fi

echo "Starting RF Spectrogram Annotator (CVAT BDW Edition) on http://localhost:$PORT ..."
exec python3 -m uvicorn backend.app:app --host 0.0.0.0 --port $PORT --reload
