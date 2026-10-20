# Multi-Level IoT Intrusion Detection System - Production Edge Container
FROM python:3.12-slim

# Install system dependencies (libpcap for Scapy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpcap-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition & install packages
COPY pyproject.toml .
RUN pip install --no-cache-dir scapy pandas numpy scikit-learn joblib pytest

# Copy source repository
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/
COPY data/ data/

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Health check command
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "from iot_ids.inference.predictor import IDSPredictor; p = IDSPredictor.from_artifact('models/final/deployable_artifact'); assert p.health_check()['pipeline']['status'] == 'HEALTHY'"

# Default entrypoint
ENTRYPOINT ["python", "-m", "iot_ids.cli"]
CMD ["validate", "--artifact-dir", "models/final/deployable_artifact", "--test-domain", "ToN-IoT"]
