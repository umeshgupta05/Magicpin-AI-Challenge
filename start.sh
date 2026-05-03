#!/usr/bin/env sh
set -e

PORT_VALUE="${PORT:-8080}"
exec uvicorn bot:app --host 0.0.0.0 --port "$PORT_VALUE"
