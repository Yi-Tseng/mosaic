#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
"""
Prometheus exporter for GPU to PCIe port mapping.

This script discovers GPUs and their PCIe ports, then exposes them as
Prometheus metrics with labels for host, gpu_id, and pcie_port.
"""

import argparse
import subprocess
import socket
import re
import sys
import time
from prometheus_client import Gauge, start_http_server

# Prometheus metric
gpu_pcie_mapping = Gauge(
    "gpu_pcie_port", "GPU to PCIe port mapping", ["host", "gpu_id", "pcie_port", "vendor", "gpu_uuid"]
)


def get_hostname():
    """Get the hostname of the current machine."""
    return socket.gethostname()


def get_nvidia_gpus():
    """Get NVIDIA GPU information using nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                if len(parts) >= 2:
                    gpu_id = parts[0].strip()
                    pcie_bus = parts[1].strip()
                    uuid = parts[2].strip().removeprefix("GPU-")
                    gpus.append({"gpu_id": gpu_id, "pcie_port": pcie_bus, "vendor": "nvidia", "gpu_uuid": uuid})
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError, OSError):
        # FileNotFoundError: nvidia-smi not found
        # PermissionError: nvidia-smi exists but not executable/accessible
        # OSError: other OS-level errors (e.g., file not found on some systems)
        pass

    return gpus


def get_amd_gpus():
    """Get AMD GPU information using rocm-smi to get GPU ID to PCIe bus mapping."""
    gpus = []
    gpu_ids = []

    # Try rocm-smi to get GPU count and PCIe bus IDs
    try:
        # First, get list of GPU IDs
        result = subprocess.run(["rocm-smi", "--showid"], capture_output=True, text=True, check=True)

        # Parse rocm-smi output to get GPU IDs
        for line in result.stdout.strip().split("\n"):
            # Look for GPU entries like "GPU[0]"
            match = re.search(r"GPU\[(\d+)\]", line)
            if match:
                gpu_ids.append(match.group(1))

        # Try rocm-smi --showbus first (most reliable)
        try:
            result = subprocess.run(["rocm-smi", "--showbus"], capture_output=True, text=True, check=True)

            # Parse output to map GPU IDs to PCIe bus IDs
            # Format: "GPU[0]		: PCI Bus: 0000:07:00.0"
            for line in result.stdout.split("\n"):
                # Look for GPU ID and PCI Bus in the same line
                gpu_match = re.search(r"GPU\[(\d+)\]", line)
                pcie_match = re.search(
                    r"PCI Bus:\s*([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])", line, re.IGNORECASE
                )

                if gpu_match and pcie_match:
                    gpu_id = gpu_match.group(1)
                    pcie_bus = pcie_match.group(1)

                    # Only add if we haven't already added this GPU
                    if not any(g["gpu_id"] == gpu_id for g in gpus):
                        gpus.append({"gpu_id": gpu_id, "pcie_port": pcie_bus, "vendor": "amd", "gpu_uuid": "n/a"})
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # If we didn't get all GPUs, try querying each GPU individually
        if len(gpus) < len(gpu_ids):
            for gpu_id in gpu_ids:
                # Skip if we already have this GPU
                if any(g["gpu_id"] == gpu_id for g in gpus):
                    continue

                try:
                    # Try rocm-smi -d <gpu_id> -a (all info for specific device)
                    result = subprocess.run(
                        ["rocm-smi", "-d", gpu_id, "-a"], capture_output=True, text=True, check=True
                    )

                    # Look for PCIe bus ID in the output
                    pcie_bus = None
                    for line in result.stdout.split("\n"):
                        # Try to find PCI Bus in the output
                        match = re.search(
                            r"PCI.*?Bus.*?([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])", line, re.IGNORECASE
                        )
                        if match:
                            pcie_bus = match.group(1)
                            break

                    if pcie_bus:
                        gpus.append({"gpu_id": gpu_id, "pcie_port": pcie_bus, "vendor": "amd", "gpu_uuid": "n/a"})
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue

        # Try rocm-smi --showhw as another fallback (shows BUS column)
        if len(gpus) < len(gpu_ids):
            try:
                result = subprocess.run(["rocm-smi", "--showhw"], capture_output=True, text=True, check=True)

                # Parse hardware info - look for BUS column
                # Format: "0    2     0x7551  28209  gfx1201  ...  0000:07:00.0  0"
                for line in result.stdout.split("\n"):
                    # Look for lines that start with a GPU number
                    match = re.match(
                        r"^\s*(\d+)\s+.*?([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])", line, re.IGNORECASE
                    )
                    if match:
                        gpu_id = match.group(1)
                        pcie_bus = match.group(2)

                        # Skip if we already have this GPU
                        if not any(g["gpu_id"] == gpu_id for g in gpus):
                            gpus.append({"gpu_id": gpu_id, "pcie_port": pcie_bus, "vendor": "amd", "gpu_uuid": "n/a"})
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Normalize PCIe port format and ensure we only have .0 functions
    # Also deduplicate by GPU ID (keep first occurrence)
    seen_gpu_ids = set()
    normalized_gpus = []
    for gpu in gpus:
        gpu_id = gpu["gpu_id"]
        if gpu_id in seen_gpu_ids:
            continue

        # Normalize PCIe port format
        pcie_port = gpu["pcie_port"]
        # Ensure it has domain prefix
        if not pcie_port.startswith("0000:"):
            # Check if it's missing domain
            if re.match(r"^[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$", pcie_port, re.IGNORECASE):
                pcie_port = "0000:" + pcie_port

        # Only use .0 function (primary PCIe function)
        if not pcie_port.endswith(".0"):
            # Replace function with .0
            pcie_port = re.sub(r"\.[0-9a-f]+$", ".0", pcie_port)

        normalized_gpus.append(
            {"gpu_id": gpu_id, "pcie_port": pcie_port, "vendor": gpu["vendor"], "gpu_uuid": gpu["gpu_uuid"]}
        )
        seen_gpu_ids.add(gpu_id)

    # Sort by GPU ID to ensure consistent ordering
    normalized_gpus.sort(key=lambda x: int(x["gpu_id"]))

    return normalized_gpus


def get_gpu_pcie_mappings():
    """
    Discover all GPUs and their PCIe ports.
    Returns a list of dicts with gpu_id and pcie_port.
    """
    gpus = []

    # Try NVIDIA first
    nvidia_gpus = get_nvidia_gpus()
    if nvidia_gpus:
        gpus.extend(nvidia_gpus)

    # Try AMD
    amd_gpus = get_amd_gpus()
    if amd_gpus:
        gpus.extend(amd_gpus)

    return gpus


def update_metrics():
    """Update Prometheus metrics with current GPU to PCIe mappings."""
    hostname = get_hostname()
    gpus = get_gpu_pcie_mappings()

    # Clear existing metrics
    gpu_pcie_mapping.clear()

    # Set new metrics
    for gpu in gpus:
        gpu_pcie_mapping.labels(
            host=hostname,
            gpu_id=gpu["gpu_id"],
            pcie_port=gpu["pcie_port"],
            vendor=gpu["vendor"],
            gpu_uuid=gpu["gpu_uuid"],
        ).set(1)

    return len(gpus)


def main():
    parser = argparse.ArgumentParser(
        description="Prometheus exporter for GPU to PCIe port mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start exporter on default port 8000
  %(prog)s

  # Start exporter on custom port
  %(prog)s --port 9090

  # Update metrics every 30 seconds
  %(prog)s --update-interval 30

  # Exit after 1 hour (default) so Docker restart policy brings process back
  %(prog)s

  # Disable periodic exit (run until killed)
  %(prog)s --max-uptime 0

  # Test mode: print mappings and exit
  %(prog)s --test
        """,
    )

    parser.add_argument("--port", type=int, default=8000, help="Port to expose Prometheus metrics on (default: 8000)")

    parser.add_argument(
        "--update-interval", type=int, default=60, help="Interval in seconds to refresh GPU mappings (default: 60)"
    )

    parser.add_argument(
        "--max-uptime",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Exit after this many seconds so the container can be restarted (default: 3600 = 1 hour). Use 0 to disable.",
    )

    parser.add_argument("--test", action="store_true", help="Test mode: print GPU mappings and exit")

    args = parser.parse_args()

    # Test mode
    if args.test:
        print("Discovering GPUs and PCIe ports...")
        hostname = get_hostname()
        gpus = get_gpu_pcie_mappings()

        if not gpus:
            print("No GPUs found!")
            sys.exit(1)

        print(f"\nHost: {hostname}")
        print(f"Found {len(gpus)} GPU(s):\n")
        for gpu in gpus:
            print(
                f"  GPU ID: {gpu['gpu_id']}, PCIe Port: {gpu['pcie_port']}, Vendor: {gpu['vendor']}, GPU UUID: {gpu['gpu_uuid']}"
            )

        print("\nPrometheus metric format:")
        for gpu in gpus:
            print(
                f'gpu_pcie_port{{host="{hostname}",gpu_id="{gpu["gpu_id"]}",pcie_port="{gpu["pcie_port"]}",vendor="{gpu["vendor"]}",gpu_uuid="{gpu["gpu_uuid"]}"}} 1'
            )

        return

    # Initial metric update
    num_gpus = update_metrics()
    if num_gpus == 0:
        print("Warning: No GPUs found! Metrics will be empty.")
    else:
        print(f"Found {num_gpus} GPU(s), metrics updated.")

    # Start Prometheus HTTP server
    print(f"Starting Prometheus exporter on port {args.port}")
    print(f"Metrics available at: http://0.0.0.0:{args.port}/metrics")
    start_http_server(args.port)

    # Periodically update metrics; exit after max_uptime so Docker restart brings us back
    start_time = time.monotonic()
    try:
        while True:
            time.sleep(args.update_interval)
            num_gpus = update_metrics()
            if num_gpus > 0:
                print(f"Updated metrics for {num_gpus} GPU(s)")

            if args.max_uptime > 0:
                elapsed = time.monotonic() - start_time
                if elapsed >= args.max_uptime:
                    print(f"Max uptime ({args.max_uptime}s) reached, exiting for restart...")
                    sys.exit(0)
    except KeyboardInterrupt:
        print("\nShutting down exporter...")


if __name__ == "__main__":
    main()
