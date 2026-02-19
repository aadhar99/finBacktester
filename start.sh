#!/bin/bash
# Start both API server and scheduler

# Start API server in background
gunicorn api.server:app --bind 0.0.0.0:$PORT --workers 2 &

# Start scheduler in foreground
python3 scripts/scheduler_daemon.py
