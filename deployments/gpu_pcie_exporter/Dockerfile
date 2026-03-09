# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0

FROM python:3.14-alpine

# FIXME: Remove this once we have a fix for the zlib version issue in the python:3.14-alpine image
RUN apk add --no-cache zlib=1.3.2-r0

# Set working directory
WORKDIR /app

# Upgrade pip to ensure we can install packages
RUN pip install --upgrade pip

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify prometheus_client is installed
RUN python3 -c "import prometheus_client; print('prometheus_client installed successfully')"

# Copy the exporter script
COPY gpu_pcie_exporter.py .

# Make the script executable
RUN chmod +x gpu_pcie_exporter.py

# Expose the default port
EXPOSE 8000

# Run the exporter
CMD ["python3", "gpu_pcie_exporter.py"]

ARG GIT_HASH=unknown
LABEL org.opencontainers.image.revision="${GIT_HASH}"
LABEL org.opencontainers.image.title="GPU PCIe Exporter"
LABEL org.opencontainers.image.description="GPU PCIe Exporter"
LABEL org.opencontainers.image.url="https://github.com/open-mosaic/mosaic"
LABEL org.opencontainers.image.source="https://github.com/open-mosaic/mosaic"
