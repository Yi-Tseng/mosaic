# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

FROM python:3.14-alpine

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY analyze_pipelines.py .
COPY pipeline_server.py .

# Expose the service port
EXPOSE 5000

# Health check - simple HTTP GET to /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"

# Run the server
CMD ["python3", "pipeline_server.py"]


ARG GIT_HASH=unknown
LABEL org.opencontainers.image.revision="${GIT_HASH}"
LABEL org.opencontainers.image.title="NCCL Profiler Pipeline Analyzer"
LABEL org.opencontainers.image.description="NCCL Profiler Pipeline Analyzer"
LABEL org.opencontainers.image.url="https://github.com/open-mosaic/mosaic"
LABEL org.opencontainers.image.source="https://github.com/open-mosaic/mosaic"
