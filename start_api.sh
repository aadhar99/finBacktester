#!/bin/bash
# Read PORT from environment (Railway sets this automatically)
exec gunicorn api.server:app --bind "0.0.0.0:${PORT:-8000}" --workers 2 --access-logfile - --error-logfile -
