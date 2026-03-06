<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Mosaic Dashboard Documentation

This document explains each panel in the Mosaic Grafana dashboard, including what they represent and how they are calculated.

## Dashboard Overview

The Mosaic dashboard provides a high-level view of GPU health, RDMA connectivity, transfer performance, and process monitoring across the cluster. It supports both AMD and Nvidia GPU exporters.

## Template Variables

The dashboard includes a template variable for filtering:

- **Host** (`$host`): Filters metrics by hostname. Supports multiple selection.

## Panels

### 1. # GPU Errors

**Description:** Overall GPU health status (supports both AMD and Nvidia)
- **0**: Normal (all GPUs healthy)
- **1+**: Number of GPUs that are unhealthy

**Calculation:**

The panel combines metrics from both AMD and Nvidia GPU exporters comparing the total number of GPUs present to the the total number of healthy GPUs:

**AMD GPU Errors:**
```
sum(
  clamp_min(
    max_over_time(sum(gpu_nodes_total{host=~"$host"})[$__range:])
    - sum(
      gpu_health{host=~"$host"}
        * on(host, gpu_id) group_left(pcie_port)
          gpu_pcie_port{host=~"$host", pcie_port=~"$pcie_port"}
    )
    , 0
  )
) or vector(0)
```

This calculates: `Total GPUs - Healthy GPUs` for AMD systems. The `max_over_time` ensures we capture the maximum GPU count over the time range, accounting for any GPUs that may have been present earlier but are now missing.

**Nvidia GPU Errors:**
```
sum(
  clamp_min(
    max_over_time(count(DCGM_FI_DEV_GPU_UTIL{host=~"$host"}) by (host)[$__range])
    - count(DCGM_FI_DEV_GPU_UTIL{host=~"$host"}) by (host)
    , 0
  )
) or vector(0)
```

This detects missing GPUs by comparing:
- `max_over_time(count(DCGM_FI_DEV_GPU_UTIL))`: The expected number of GPUs per host by determining the maximum number of unique GPUs that have reported over the selected time range.
- `count(DCGM_FI_DEV_GPU_UTIL)`: The actual number of GPUs currently reporting metrics for each host.

If a GPU stops reporting metrics, it's considered unhealthy/missing.

**Combined Result:**
The two results are added together: `(AMD errors or 0) + (Nvidia errors or 0)`

**Color Coding:**
- Green: 0 errors
- Red: 1+ errors

---

### 2. # RDMA Errors

**Description:** Overall RDMA health status
- **0**: Normal (all InfiniBand links healthy)
- **1+**: Number of IB links degraded

**Calculation:**

The panel uses InfiniBand state metrics provided by the prometheus node exporter to determine degredation.
```
clamp_min(
  (
    count((node_infiniband_physical_state_id offset $__range) == 5)
    - count(node_infiniband_physical_state_id == 5)
  ), 0
)
```

This detects InfiniBand links that have degraded during the time range:
- `node_infiniband_physical_state_id == 5` represents links in a degraded state
- The query compares the count of degraded links at the start of the range (using `offset $__range`) with the current count
- The difference shows how many links have become degraded during the time period

**Color Coding:**
- Green: 0 errors
- Red: 1+ errors

---

### 3. # Transfer Time Warnings

**Description:** Number of GPU pairs that have transfer time over 1.5 standard deviations
- **0**: Normal
- **1+**: Number of GPU pairs with sub-optimal transfer time

**Calculation:**
```
count(
  (mosaic_gpu_transfer_time - scalar(avg(mosaic_gpu_transfer_time)))
  / scalar(stddev(mosaic_gpu_transfer_time)) > 1.5
) OR vector(0)
```

This identifies GPU pairs with transfer times significantly above average:
1. Calculates the z-score: `(transfer_time - average) / standard_deviation`
2. Counts pairs where z-score > 1.5 (indicating transfer time is 1.5 standard deviations above the mean)
3. Returns 0 if no metrics are available

**Color Coding:**
- Green: 0 warnings
- Dark Yellow: 1+ warnings

---

### 4. # Transfer Time Errors

**Description:** Number of GPU pairs that have transfer time over 3 standard deviations
- **0**: Normal
- **1+**: Number of GPU pairs with severely degraded transfer time

**Calculation:**
```
count(
  (mosaic_gpu_transfer_time - scalar(avg(mosaic_gpu_transfer_time)))
  / scalar(stddev(mosaic_gpu_transfer_time)) > 3
) OR vector(0)
```

Similar to the warnings panel, but uses a threshold of 3 standard deviations instead of 1.5, indicating severely degraded performance.

**Color Coding:**
- Green: 0 errors
- Red: 1+ errors

---

### 5. # GPU CCL Processes

**Description:** Indicates if GPU collective communication (CCL) processes are running
- **0**: There is no workload, or collective communication processes have crashed
- **1+**: Normal (processes are running)

**Calculation:**

The panel uses metrics from the process exporter to display the total number of collective communication processes running. It may be necessary to adjust the processor-exporter.config.yml present in the amd-gpu-monitoring or nvidia-gpu-monitoring folder to capture the processes correctly.
```
sum(namedprocess_namegroup_num_threads{host="$host"})
```

This sums the number of threads from named processes, which indicates the presence of GPU CCL processes. A value of 0 suggests either no workload is running or the processes have crashed.

**Color Coding:**
- Red: 0 (no processes)
- Green: 1+ (processes running)

---

### 6. # Active Profiler Metrics

**Description:** Number of active metrics collected by NCCL profiler plugin
- **0**: Nothing is collected, indicating possible issues
- **1+**: Metrics are being collected

**Calculation:**
```
count({__name__=~"^(nccl_).*"}) OR vector(0)
```

This counts all metrics whose name starts with `nccl_`, which are generated by the NCCL profiler plugin. If no metrics are found, it returns 0, which may indicate the profiler is not running or not collecting data.

**Color Coding:**
- Red: 0 (no metrics)
- Green: 1+ (metrics being collected)

---

## Notes

- Host specific panels use the `$host` template variable to filter by hostname. This includes the # GPU Errors, # RDMA Errors, and # GPU CCL Processes panels.
- Time range queries use `$__range` to ensure calculations account for the selected time window
- The `or vector(0)` pattern ensures queries return 0 instead of empty results when metrics are unavailable
- All panels display the last non-null value using the `lastNotNull` calculation method
