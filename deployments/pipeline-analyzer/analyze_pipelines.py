#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Pipeline Analysis for NCCL Profiler

Analyzes Prometheus metrics to identify pipeline parallelism structure.
Determines which communicators form pipeline stages and which connect pipelines.

Logic:
- A GPU (hostname+uuid+pci) can only belong to ONE pipeline
- Communicators sharing GPUs are part of the same pipeline
- Communicators with ≥4 unique GPUs are pipeline stages
- Others are classified as pipeline-internal or inter-pipeline connectors
"""

import requests
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_metrics_endpoint(metrics_url: str) -> List[Dict]:
    """
    Parse metrics directly from /metrics endpoint.

    Args:
        metrics_url: URL to Prometheus metrics endpoint (e.g., http://localhost:8889/metrics)

    Returns:
        List of dicts with metric labels (communicator, hostname, gpu_uuid, rank, etc.)
    """
    try:
        response = requests.get(metrics_url)
        response.raise_for_status()

        results = []
        for line in response.text.split("\n"):
            if line.startswith("nccl_profiler_collective_count_count{"):
                # Parse metric labels
                start = line.index("{") + 1
                end = line.index("}")
                labels_str = line[start:end]

                labels = {}
                for label_pair in labels_str.split(","):
                    if "=" in label_pair:
                        key, value = label_pair.split("=", 1)
                        labels[key.strip()] = value.strip('"')

                if "communicator" in labels and "hostname" in labels and "gpu_uuid" in labels:
                    results.append(labels)

        return results
    except Exception as e:
        print(f"Error parsing metrics: {e}", file=sys.stderr)
        return []


def get_gpu_id(hostname: str, gpu_uuid: str, pci_bus_id: str) -> str:
    """
    Create unique GPU identifier.

    Args:
        hostname: Node hostname
        gpu_uuid: GPU UUID
        pci_bus_id: PCI bus ID

    Returns:
        Unique identifier string: "hostname|gpu_uuid|pci_bus_id"
    """
    return f"{hostname}|{gpu_uuid}|{pci_bus_id}"


def analyze_pipelines(metrics_url: str) -> Tuple[Dict, Dict, List]:
    """
    Analyze pipeline structure from metrics endpoint.

    Algorithm:
    1. Parse all metrics to build communicator ↔ GPU mappings
    2. Identify pipeline communicators (≥4 GPUs)
    3. Assign sequential pipeline IDs (0 to PP-1)
    4. Classify all communicators:
       - "pipeline": Main pipeline stage
       - "pipeline-internal": Smaller comm within one pipeline
       - "inter-pipeline": Connects multiple pipelines
       - "unassigned": Not part of any identified pipeline

    Args:
        metrics_url: URL to Prometheus metrics endpoint

    Returns:
        Tuple of:
        - comm_to_gpus: Dict[str, Set[str]] - communicator hash -> set of GPU IDs
        - gpu_to_comms: Dict[str, Set[str]] - GPU ID -> set of communicator hashes
        - pipeline_assignments: List[Dict] - list of pipeline assignment dicts with:
            {
                "pipeline_id": int,           # 0 to PP-1, or -1 for inter-pipeline
                "communicator": str,          # communicator hash
                "gpu_count": int,             # number of GPUs in this comm
                "total_pipeline_gpus": int,   # total GPUs in the pipeline
                "is_pipeline": bool,          # True if main pipeline stage
                "type": str,                  # "pipeline", "pipeline-internal", "inter-pipeline", "unassigned"
                "connects_pipelines": List[int],  # [p1, p2] for inter-pipeline only
                "pipeline_gpus": Dict         # GPU details by pipeline (for inter-pipeline)
            }
    """

    # Parse metrics from endpoint
    metrics = parse_metrics_endpoint(metrics_url)

    comm_to_gpus = defaultdict(set)
    gpu_to_comms = defaultdict(set)

    # Build mappings
    for metric in metrics:
        comm = metric.get("communicator", "")
        hostname = metric.get("hostname", "")
        gpu_uuid = metric.get("gpu_uuid", "")
        pci_bus_id = metric.get("gpu_pci_bus_id", "")

        if comm and hostname and gpu_uuid:
            gpu_id = get_gpu_id(hostname, gpu_uuid, pci_bus_id)
            comm_to_gpus[comm].add(gpu_id)
            gpu_to_comms[gpu_id].add(comm)

    # Step 1: Identify pipeline communicators (those with ≥4 unique GPUs)
    pipeline_comms = {comm for comm, gpus in comm_to_gpus.items() if len(gpus) >= 4}

    # Step 2: Assign each GPU to a pipeline based on which pipeline communicator uses it
    gpu_to_pipeline = {}
    pipeline_to_gpus = defaultdict(set)

    # Sort pipeline communicators by GPU count (largest first) to assign GPUs
    for comm in sorted(pipeline_comms, key=lambda c: len(comm_to_gpus[c]), reverse=True):
        for gpu_id in comm_to_gpus[comm]:
            if gpu_id not in gpu_to_pipeline:
                # This GPU hasn't been assigned to a pipeline yet
                gpu_to_pipeline[gpu_id] = comm
                pipeline_to_gpus[comm].update(comm_to_gpus[comm])

    # Step 3: Assign pipeline IDs to unique pipeline GPU sets
    unique_pipelines = list(pipeline_to_gpus.items())
    unique_pipelines.sort(key=lambda x: (len(x[1]), x[0]), reverse=True)

    comm_to_pipeline_id = {}
    pipeline_id = 0

    for comm, gpus in unique_pipelines:
        comm_to_pipeline_id[comm] = pipeline_id
        pipeline_id += 1

    # Step 4: Assign each communicator to a pipeline
    pipeline_assignments = []

    for comm in sorted(comm_to_gpus.keys()):
        comm_gpus = comm_to_gpus[comm]
        gpu_count = len(comm_gpus)

        if comm in pipeline_comms and comm in comm_to_pipeline_id:
            # This is a main pipeline communicator
            pid = comm_to_pipeline_id[comm]
            total_gpus = len(pipeline_to_gpus[comm])
            pipeline_assignments.append(
                {
                    "pipeline_id": pid,
                    "communicator": comm,
                    "gpu_count": gpu_count,
                    "total_pipeline_gpus": total_gpus,
                    "is_pipeline": True,
                    "type": "pipeline",
                }
            )
        else:
            # Check if this communicator's GPUs belong to a single pipeline or multiple
            pipeline_owners = set()
            for gpu_id in comm_gpus:
                if gpu_id in gpu_to_pipeline:
                    owner_comm = gpu_to_pipeline[gpu_id]
                    pipeline_owners.add(comm_to_pipeline_id[owner_comm])

            if len(pipeline_owners) == 1:
                # All GPUs belong to same pipeline - this is pipeline-internal
                pid = pipeline_owners.pop()
                total_gpus = gpu_count
                pipeline_assignments.append(
                    {
                        "pipeline_id": pid,
                        "communicator": comm,
                        "gpu_count": gpu_count,
                        "total_pipeline_gpus": total_gpus,
                        "is_pipeline": False,
                        "type": "pipeline-internal",
                    }
                )
            elif len(pipeline_owners) > 1:
                # GPUs span multiple pipelines - inter-pipeline connector
                # Get GPU details for each pipeline
                pipeline_gpus = defaultdict(list)
                for gpu_id in comm_gpus:
                    if gpu_id in gpu_to_pipeline:
                        owner_comm = gpu_to_pipeline[gpu_id]
                        pid = comm_to_pipeline_id[owner_comm]
                        # Parse GPU ID: hostname|gpu_uuid|pci_bus_id
                        parts = gpu_id.split("|")
                        if len(parts) >= 2:
                            pipeline_gpus[pid].append(
                                {
                                    "hostname": parts[0],
                                    "gpu_uuid": parts[1],
                                    "pci_bus_id": parts[2] if len(parts) > 2 else "",
                                }
                            )

                pipeline_assignments.append(
                    {
                        "pipeline_id": -1,
                        "communicator": comm,
                        "gpu_count": gpu_count,
                        "total_pipeline_gpus": 0,
                        "is_pipeline": False,
                        "type": "inter-pipeline",
                        "connects_pipelines": sorted(pipeline_owners),
                        "pipeline_gpus": dict(pipeline_gpus),
                    }
                )
            else:
                # GPUs not part of any identified pipeline
                pipeline_assignments.append(
                    {
                        "pipeline_id": -1,
                        "communicator": comm,
                        "gpu_count": gpu_count,
                        "total_pipeline_gpus": 0,
                        "is_pipeline": False,
                        "type": "unassigned",
                    }
                )

    return comm_to_gpus, gpu_to_comms, pipeline_assignments
