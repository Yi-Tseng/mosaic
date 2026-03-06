// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>
#include <unistd.h>

#include "profiler_otel.h"
#include "profiler_v4_compat.h"

// NCCL Profiler Plugin — dual v4/v5 export.
// Newer NCCL (v5+) resolves ncclProfiler_v5 first for best fidelity.
// Older NCCL (v4-only) falls back to ncclProfiler_v4 via thin adapters.
extern "C"
{
    volatile ncclProfiler_v5_t ncclProfiler_v5 = {
        "otel-profiler",                      // name
        profiler_otel_init_v5,                // init
        profiler_otel_start_event_v5,         // startEvent
        profiler_otel_stop_event_v5,          // stopEvent
        profiler_otel_record_event_state_v5,  // recordEventState
        profiler_otel_finalize_v5,            // finalize
    };

    volatile ncclProfiler_v4_t ncclProfiler_v4 = {
        "otel-profiler",                      // name
        profiler_otel_init_v4,                // init (v4 param order → v5)
        profiler_otel_start_event_v4,         // startEvent (v4 descriptor → v5)
        profiler_otel_stop_event_v5,          // stopEvent (identical signature)
        profiler_otel_record_event_state_v4,  // recordEventState (v4 types → v5)
        profiler_otel_finalize_v5,            // finalize (identical signature)
    };
}  // extern "C"
