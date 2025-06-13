# Multi-stage build for production-ready object detection API
FROM python:3.10-slim as builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim as production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libgthread-2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser requirements.txt .

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Health check function\n\
health_check() {\n\
    curl -f http://localhost:8000/health > /dev/null 2>&1\n\
}\n\
\n\
# Start the application\n\
exec uvicorn app:app \\\n\
    --host 0.0.0.0 \\\n\
    --port 8000 \\\n\
    --workers ${WORKERS:-1} \\\n\
    --loop uvloop \\\n\
    --http httptools \\\n\
    --access-log \\\n\
    --log-level ${LOG_LEVEL:-info}\n\
' > /app/start.sh && chmod +x /app/start.sh

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV WORKERS=1
ENV LOG_LEVEL=info
ENV MODEL_V1_PATH="/app/models/yolov9_v1.onnx"
ENV MODEL_V2_PATH="/app/models/yolov9_v2.onnx"
ENV S3_BUCKET_NAME="ml-active-learning"
ENV UNCERTAINTY_THRESHOLD="0.7"
ENV REFERENCE_DATA_PATH="/app/data/reference_features.json"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["/app/start.sh"]

# Labels for metadata
LABEL maintainer="MLOps Team"
LABEL version="1.0.0"
LABEL description="Production-ready YOLOv9 Object Detection API with MLOps features"
LABEL org.opencontainers.image.source="https://github.com/your-org/real-time-vision-api" 