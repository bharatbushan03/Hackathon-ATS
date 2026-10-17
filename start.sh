#!/bin/sh
# Container entrypoint for Azure App Service.
# FastAPI runs privately on 127.0.0.1:8001; Streamlit is the public app on $PORT.
set -e

PORT="${PORT:-8000}"
mkdir -p "${DATA_DIR:-/home/data}"

python -m uvicorn hackeval.api:app --host 127.0.0.1 --port 8001 &

exec python -m streamlit run streamlit_app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.maxUploadSize "${MAX_ZIP_MB:-200}" \
  --browser.gatherUsageStats false
