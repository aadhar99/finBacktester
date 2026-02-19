#!/usr/bin/env python3
"""
Entry point for API server - handles PYTHONPATH setup
"""
import os
import sys

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run gunicorn
from gunicorn.app.wsgiapp import run

if __name__ == '__main__':
    # Get PORT from environment (Railway sets this)
    port = os.environ.get('PORT', '8000')
    
    # Run gunicorn with proper settings
    sys.argv = [
        'gunicorn',
        'api.server:app',
        '--bind', f'0.0.0.0:{port}',
        '--workers', '2',
        '--access-logfile', '-',
        '--error-logfile', '-'
    ]
    
    run()
