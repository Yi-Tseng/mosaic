<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Nvidia GPU Monitoring

This directory contains a Docker Compose configuration that provides additional metrics sources specifically designed to analyze Nvidia GPU activity.

## Overview

The `docker-compose.yml` file orchestrates multiple monitoring services that collect and expose metrics related to Nvidia GPU performance, device status, PCIe activity, and system-level metrics. These metrics sources can be scraped by monitoring systems (such as Prometheus) to provide comprehensive visibility into Nvidia GPU workloads and system behavior.

## Services

### nvidia-device-exporter
- **Image**: `utkuozdemir/nvidia_gpu_exporter:1.4.1`
- **Port**: `5053` (maps to container port `9835`)
- **Profile**: `nvidia-device-exporter` (use `--profile nvidia-device-exporter` to enable)
- **Purpose**: Exposes Nvidia device-specific metrics from Nvidia GPUs using nvidia-smi
- **GPU Access**: Requires access to Nvidia GPUs via Docker's GPU runtime
- **Note**: This service is only started when using the `nvidia-device-exporter` profile
- **Source**: https://github.com/utkuozdemir/nvidia_gpu_exporter

### gpu-pcie-exporter
- **Build**: Custom image from `../gpu_pcie_exporter`
- **Port**: `5052`
- **Purpose**: Provides information to map hostname and GPU ID to PCIe port number
- **GPU Access**: Requires access to Nvidia GPUs via Docker's GPU runtime
- **Dependencies**: Mounts `/usr/bin/nvidia-smi` from the host to access Nvidia GPU tools
- **Source**: [gpu_pcie_exporter](../gpu_pcie_exporter/README.md)

### node-exporter
- **Image**: `prom/node-exporter:latest`
- **Port**: `9100`
- **Profile**: `node-exporter` (use `--profile node-exporter` to enable)
- **Purpose**: Collects system-level metrics (CPU, memory, disk, network, etc.)
- **Note**: This service is only started when using the `node-exporter` profile
- **Source**: https://github.com/prometheus/node_exporter

### process-exporter
- **Image**: `ncabatoff/process-exporter:latest`
- **Port**: `9256`
- **Purpose**: Exports process-level metrics for detailed workload analysis
- **Configuration**: Uses `process-exporter.config.yml` for process filtering
- **Source:** https://github.com/ncabatoff/process-exporter

## Usage

To start services:

```bash
docker compose up -d
```

To start with the node-exporter (node-exporter profile):

```bash
docker compose --profile=node-exporter up -d
```

To start with the node-exporter and nvidia-device-exporter (node-exporter and nvidia-device-exporter profiles):

```bash
docker compose --profile=node-exporter --profile=nvidia-device-exporter up -d
```

To stop all services:

```bash
docker compose down
```

## Metrics Endpoints

Once running, metrics are available at:
- Nvidia Device Metrics: `http://localhost:5053/metrics` (if enabled)
- GPU PCIe Metrics: `http://localhost:5052/metrics`
- Node Metrics: `http://localhost:9100/metrics` (if enabled)
- Process Metrics: `http://localhost:9256/metrics`

## Requirements

- Docker and Docker Compose
- Nvidia GPUs with Nvidia drivers installed
- Docker GPU runtime support (nvidia-container-toolkit)
- Access to `/usr/bin/nvidia-smi` on the host system

## Integration

These metrics endpoints can be configured as scrape targets in Prometheus or other monitoring systems to collect and analyze Nvidia GPU activity over time.
