#!/bin/bash
set -e

echo "🚀 Starting Smart Money Trading System..."

# Default PORT if not set by Railway
export PORT=${PORT:-8000}

echo "📡 Starting API server on port $PORT..."
# Start gunicorn in background with logging
gunicorn api.server:app \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --access-logfile - \
    --error-logfile - &

GUNICORN_PID=$!
echo "✅ API server started (PID: $GUNICORN_PID)"

# Give gunicorn a moment to start
sleep 3

echo "⏰ Starting trading scheduler..."
# Start scheduler in foreground
python3 scripts/scheduler_daemon.py

# If scheduler exits, kill gunicorn
kill $GUNICORN_PID 2>/dev/null || true
