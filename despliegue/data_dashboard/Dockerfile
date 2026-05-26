# ──────────────────────────────────────────────────────────
# Dockerfile - Predictor ICFES Dashboard (Q1 + Q2 + Q3)
# ──────────────────────────────────────────────────────────

FROM python:3.12-slim

# System deps: gcc/g++ for numba/llvmlite, libgomp for TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY app.py .
COPY model_utils.py .
COPY model_utils_q2.py .
COPY model_utils_q3.py .

# Static assets
COPY assets/ ./assets/

# Pre-trained model artifacts
COPY model/ ./model/
COPY model_q2/ ./model_q2/
COPY model_q3/ ./model_q3/

EXPOSE 8050

CMD ["gunicorn", "app:server", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info"]
