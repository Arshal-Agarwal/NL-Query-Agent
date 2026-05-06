#!/usr/bin/env bash
echo "Starting NL Financial Query Agent..."
echo "Open http://localhost:8000 in your browser"
cd "$(dirname "$0")"
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
