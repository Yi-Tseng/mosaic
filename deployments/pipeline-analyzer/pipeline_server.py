#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Pipeline Analysis Metrics Exporter for Prometheus

Provides Prometheus-compatible metrics endpoints that Grafana queries via Prometheus.
Only includes endpoints actually used by the Grafana dashboard.
"""

from flask import Flask
from flask_cors import CORS
import os
import sys

# Import the analysis functions from analyze_pipelines.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_pipelines import analyze_pipelines, parse_metrics_endpoint

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Get metrics endpoint from environment variable
METRICS_ENDPOINT = os.getenv("METRICS_ENDPOINT", "http://localhost:8889/metrics")


@app.route("/health")
def health():
    """Health check endpoint for Docker"""
    return "OK", 200


@app.route("/metrics/pipelines")
def get_pipeline_metrics():
    """
    Export pipeline analysis as Prometheus-compatible metrics.
    Used by Grafana's "Pipeline Analysis" table panel.

    Returns:
        nccl_pipeline_info: One row per communicator with pipeline assignment
    """
    try:
        comm_to_gpus, gpu_to_comms, pipeline_assignments = analyze_pipelines(METRICS_ENDPOINT)

        lines = []
        lines.append("# HELP nccl_pipeline_info Pipeline assignment information")
        lines.append("# TYPE nccl_pipeline_info gauge")

        for item in pipeline_assignments:
            pid = item.get("pipeline_id", -1)
            comm = item.get("communicator", "")
            gpu_count = item.get("gpu_count", 0)
            is_pipeline = 1 if item.get("is_pipeline", False) else 0
            comm_type = item.get("type", "unknown")

            # For inter-pipeline, add connects_from and connects_to labels plus GPU details
            if comm_type == "inter-pipeline" and "connects_pipelines" in item:
                pipelines = item["connects_pipelines"]
                pipeline_gpus = item.get("pipeline_gpus", {})

                if len(pipelines) >= 2:
                    p_from = pipelines[0]
                    p_to = pipelines[1]

                    # Get GPU details for source and destination
                    from_gpus = pipeline_gpus.get(p_from, [])
                    to_gpus = pipeline_gpus.get(p_to, [])

                    # Format: hostname:gpu_uuid (first 8 chars)
                    from_gpu_str = ""
                    to_gpu_str = ""

                    if from_gpus:
                        from_gpu = from_gpus[0]  # Take first GPU
                        from_gpu_str = f"{from_gpu['hostname']}:{from_gpu['gpu_uuid'][:8]}"

                    if to_gpus:
                        to_gpu = to_gpus[0]  # Take first GPU
                        to_gpu_str = f"{to_gpu['hostname']}:{to_gpu['gpu_uuid'][:8]}"

                    labels = f'pipeline_id="{pid}",communicator="{comm}",type="{comm_type}",is_pipeline="{is_pipeline}",connects_from="{p_from}",connects_to="{p_to}",from_gpu="{from_gpu_str}",to_gpu="{to_gpu_str}"'
                else:
                    labels = f'pipeline_id="{pid}",communicator="{comm}",type="{comm_type}",is_pipeline="{is_pipeline}",connects_from="",connects_to="",from_gpu="",to_gpu=""'
            else:
                # For pipeline stages, just use pipeline_id
                labels = f'pipeline_id="{pid}",communicator="{comm}",type="{comm_type}",is_pipeline="{is_pipeline}",connects_from="",connects_to="",from_gpu="",to_gpu=""'

            lines.append(f"nccl_pipeline_info{{{labels}}} {gpu_count}")

        return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}
    except Exception as e:
        return f"# Error: {str(e)}\n", 500, {"Content-Type": "text/plain"}


@app.route("/metrics/rank_mapping")
def get_rank_mapping_metrics():
    """
    Export rank-to-pipeline mapping as Prometheus metrics.
    Used by Grafana's "Rank Mapping Table" panel.

    Returns:
        nccl_rank_pipeline_mapping: One row per (communicator, GPU) pair with rank and pipeline info
    """
    try:
        # Parse raw metrics
        metrics = parse_metrics_endpoint(METRICS_ENDPOINT)

        # Get pipeline analysis
        comm_to_gpus, gpu_to_comms, pipeline_assignments = analyze_pipelines(METRICS_ENDPOINT)

        # Build comm -> pipeline_id mapping (numeric)
        comm_to_pipeline_id = {}
        comm_to_type = {}
        for item in pipeline_assignments:
            comm = item.get("communicator", "")
            pid = item.get("pipeline_id", -1)
            comm_type = item.get("type", "unknown")

            comm_to_pipeline_id[comm] = pid
            comm_to_type[comm] = comm_type

        lines = []
        lines.append("# HELP nccl_rank_pipeline_mapping Rank to pipeline mapping")
        lines.append("# TYPE nccl_rank_pipeline_mapping gauge")

        for metric in metrics:
            comm = metric.get("communicator", "")
            hostname = metric.get("hostname", "")
            gpu_uuid = metric.get("gpu_uuid", "")
            pci_bus_id = metric.get("gpu_pci_bus_id", "")
            rank = metric.get("rank", "")
            local_rank = metric.get("local_rank", "")

            if comm and hostname and gpu_uuid and rank:
                pid = comm_to_pipeline_id.get(comm, -1)
                comm_type = comm_to_type.get(comm, "unknown")

                labels = f'pipeline_id="{pid}",communicator="{comm}",global_rank="{rank}",local_rank="{local_rank}",hostname="{hostname}",gpu_uuid="{gpu_uuid}",gpu_pci_bus_id="{pci_bus_id}",type="{comm_type}"'
                lines.append(f"nccl_rank_pipeline_mapping{{{labels}}} 1")

        return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}
    except Exception as e:
        return f"# Error: {str(e)}\n", 500, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"Starting Pipeline Metrics Exporter on port {port}")
    print(f"Metrics endpoint: {METRICS_ENDPOINT}")
    print("Exposing:")
    print("  - /metrics/pipelines (nccl_pipeline_info)")
    print("  - /metrics/rank_mapping (nccl_rank_pipeline_mapping)")
    app.run(host="0.0.0.0", port=port, debug=False)
