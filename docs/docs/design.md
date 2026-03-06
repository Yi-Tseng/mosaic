---
icon: fontawesome/solid/compass-drafting
title: Design
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Codebase Structure

The plugin is organized into several key components:

## Core Plugin Interface
`profiler_otel.h` / `profiler_otel.cc`

- **Purpose**: Implements the NCCL Profiler Plugin v4 interface
- **Key Functions**:
    - `profiler_otel_init_v4()`: Initialize plugin for a communicator
    - `profiler_otel_start_event_v4()`: Handle event start (Coll, P2P, ProxyOp, ProxyStep, Group)
    - `profiler_otel_stop_event_v4()`: Handle event stop
    - `profiler_otel_record_event_state_v4()`: Record ProxyStep state transitions (SendWait, etc.)
    - `profiler_otel_finalize_v4()`: Cleanup plugin for a communicator
- **Event Filtering**: Skips ProxyCtrl, receive-side ProxyOps/ProxySteps, and P2P Recv events. Group events return a valid handle for correct parent-chain behavior, but they are not exported as metrics.

## Circular Buffer Management
`communicator_state.h` / `communicator_state.cc`

- **Purpose**: Lock-free circular buffer system for event storage
- **Key Components**:
    - `CommunicatorState`: Manages 4 circular buffers per communicator
    - `WindowMetadata`: Tracks window state, element count, and in-progress operations
    - Window state machine: FILLING → CLOSING → PROCESSING → READY
- **Features**:
    - Lock-free allocation using atomic operations
    - Parent-aware routing (events route to parent's window if parent is CLOSING)
    - Time-based and count-based window triggers
    - In-progress operation tracking for window boundary handling

## Event Aggregation
`aggregation.h` / `aggregation.cc`

- **Purpose**: Aggregate events within a window and calculate metrics
- **Key Components**:
    - `WindowAggregator`: Processes a single window of events
    - `AggregatedCollective`: Aggregated collective operation statistics
    - `AggregatedP2P`: Aggregated P2P operation statistics
    - `AggregatedTransfer`: Aggregated transfer statistics (rank-to-rank and per-channel)
- **Processing Phases**:
    1. Track Coll/P2P operations
    2. Aggregate ProxyStep transfers to ProxyOps
    3. Link ProxyOps to parent Collectives/P2Ps
    4. Calculate durations and prepare metrics for export

## Telemetry Export
`telemetry.h` / `telemetry.cc`

- **Purpose**: Export metrics to OpenTelemetry collectors
- **Key Components**:
    - Background telemetry thread for asynchronous processing
    - OpenTelemetry metrics API integration (histograms, counters)
    - Metric export functions for Collectives, P2P, rank transfers, and channel transfers
- **Metrics Exported**:
    - Collective metrics: bytes, time, transfer counts, transfer sizes/times
    - P2P metrics: bytes, time, transfer counts, transfer sizes/times
    - Rank transfer metrics: total bytes, latency (from linear regression), rate (from linear regression)
    - Channel transfer metrics: average transfer size, average transfer time, latency

## Linear Regression
`linear_regression.h` / `linear_regression.cc`

- **Purpose**: Calculate latency and transfer rate from transfer data
- **Key Features**:
  - Two modes: AVG (use all points) and MIN (use minimum time per size)
  - Calculates slope (rate in bytes/us) and intercept (latency in us)
  - R-squared calculation for goodness of fit
  - Supports merging data from multiple instances

## Event Structures
`events.h`

- **Purpose**: Define event handle structure for circular buffer storage
- **Key Structures**:
    - `otelEventHandle_t`: Lightweight event handle with union for type-specific data
    - `eventContext`: Plugin context per communicator
- **Event Types Supported**:
    - `ncclProfileGroup`: Group events (handle returned; used for parent-chain/window management; not exported as metrics)
    - `ncclProfileColl`: Collective operations
    - `ncclProfileP2p`: Point-to-point operations
    - `ncclProfileProxyOp`: Proxy operations (per channel)
    - `ncclProfileProxyStep`: Individual transfer steps

## Plugin Registration
`nccl_plugin.cc`

- **Purpose**: Register plugin with NCCL
- **Structure**: `ncclProfiler_v4` structure exported to NCCL

## Configuration
`param.h`

- **Purpose**: Environment variable parameter loading
- **Features**: Thread-safe parameter caching with atomic operations

# Profiler Architecture

## Overview

The NCCL Profiler Plugin tracks NCCL collective and P2P operations through their complete lifecycle, from the high-level API call to the actual data transfers. It uses NCCL's event hierarchy to correlate operations:

```
NCCL API Call (Coll/P2P)
  └── Collective/P2P Event (START) ← Enqueues work
       │
       ├── Collective/P2P Event (STOP) ← Work enqueued (happens quickly)
       │
       ├── ProxyOp (send, per channel) ← Links back to Coll/P2P via parentObj
       │    │
       │    ├── ProxyStep (multiple per ProxyOp)
       │    │    ├── RecordEventState(ProxyStepSendWait) ← Actual transfer start + size
       │    │    └── ProxyStep STOP ← Transfer complete
       │    │
       │    └── ProxyOp STOP ← All steps for this channel complete
       │
       └── [Timing: Coll/P2P START → Last ProxyOp STOP = Total collective time]
```

## Event Flow and Timing

**Key Insight**: NCCL's collective operations are asynchronous. The `Coll/P2P START` and `STOP` events only reflect work enqueuing, not actual data transfer.

1. **Collective/P2P Start**: NCCL API called, work begins enqueuing
2. **Collective/P2P Stop**: Work enqueued (happens in microseconds)
3. **ProxyOp Start** (per channel): Proxy thread begins processing channel operations
    - `parentObj` points back to the parent `Coll/P2P` event
    - Only send-side `ProxyOp` events are tracked (receive-side filtered out)
4. **ProxyStep Start**: Individual transfer step begins
     - `parentObj` points to the parent `ProxyOp`
5. **RecordEventState(ProxyStepSendWait)**: Actual transfer starts
    - Provides the **real transfer size** (not the buffer size)
    - Timestamp marks the **start of actual data transfer**
6. **ProxyStep Stop**: Transfer complete for this step
    - Transfer time = `ProxyStep Stop - SendWait timestamp`
7. **ProxyOp Stop**: All steps for this channel complete
8. **Total Collective Time** = `Coll START → Last ProxyOp STOP`

## Circular Buffer Design

The plugin uses a lock-free circular buffer design with 4 pre-allocated buffers per communicator:
- Each buffer holds 100,000 events
- Windows are triggered when:
    - **Count-based**: 50,000 events collected, OR
    - **Time-based**: Configured interval elapsed (default 5 seconds, configurable via `NCCL_PROFILER_OTEL_TELEMETRY_INTERVAL_SEC`)
- Event recording on the critical path involves only:
    - Atomic increment of buffer index
    - Writing to pre-allocated buffer slot
    - No memory allocation or locks

## Window Management and State Machine

Each window transitions through these states:

1. **FILLING**: Actively collecting events
    - Events written to buffer via atomic operations
    - Count and time triggers checked on each event
    - When trigger reached: transition to CLOSING

2. **CLOSING**: Waiting for in-progress operations to complete
    - New events directed to next buffer
    - `in_progress_count` tracks operations that span window boundary
    - When `in_progress_count` reaches 0: transition to PROCESSING

3. **PROCESSING**: Telemetry thread processes events
    - Aggregates events by type
    - Links events via `parentObj` hierarchy
    - Calculates metrics and exports to OpenTelemetry
    - When processing complete: transition to READY

4. **READY**: Window cleared and available for reuse

**Window Boundary Handling**: The plugin ensures that related events (a `Coll/P2P` and its `ProxyOp`/`ProxyStep` children) stay in the same window by:
- Tracking `in_progress_count` for each window
- Incrementing on `Coll/P2P START` (by `nChannels` for expected ProxyOps)
- Decrementing on `ProxyOp STOP`
- Delaying buffer switch until `in_progress_count == 0`

## Aggregation and Metrics

The telemetry thread processes windows in multiple phases:

**Phase 1: Track Collective/P2P Operations**

- Stores `Coll/P2P` START events with expected number of `ProxyOps`
- Maps event handle pointer to operation metadata

**Phase 2: Aggregate ProxyStep Transfers**

- Groups `ProxyStep` transfers by parent `ProxyOp` (via `parentObj`)
- Only processes `ProxySteps` with `SendWait` state (actual transfers)
- Calculates transfer time = `ProxyStep STOP - SendWait timestamp`
- Uses transfer size from `SendWait` state (not buffer size)

**Phase 3: Link ProxyOps to Collectives**

- Matches `ProxyOp` to parent `Coll/P2P` (via `parentObj`)
- Aggregates all `ProxyStep` transfers to parent collective
- Updates collective duration with latest `ProxyOp STOP` time
- Calculates: `Total Collective Time = START → Last ProxyOp STOP`

**Phase 4: Export Metrics**

- Collective/P2P metrics: average bytes, time, transfer counts
- Rank-to-rank metrics: total bytes, latency, rate (via linear regression)
- Per-channel metrics: average transfer size and time

## Metric Types Exported

1. **Collective Metrics**:
    - Average bytes per collective operation
    - Average time per collective operation (START → Last ProxyOp STOP)
    - Average number of transfers (ProxySteps) per collective
    - Average transfer size and time

2. **P2P Metrics**:
    - Average bytes per P2P operation
    - Average time per P2P operation
    - Average number of transfers per P2P operation

3. **Rank Transfer Metrics**:
    - Total bytes transferred between ranks
    - Latency (from linear regression intercept)
    - Transfer rate in MB/s (from linear regression slope)
    - **Note**: Keys include communicator hash prefix (e.g., `Comm<hash>_RankXToRankY`) to avoid collisions in pipeline parallelism scenarios

4. **Channel Transfer Metrics**:
    - Average transfer size per channel
    - Average transfer time per channel
