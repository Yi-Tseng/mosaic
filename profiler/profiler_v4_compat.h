// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef PROFILER_V4_COMPAT_H
#define PROFILER_V4_COMPAT_H

#include "profiler_otel.h"

#ifdef __cplusplus
extern "C"
{
#endif

    OTEL_HIDDEN ncclResult_t profiler_otel_init_v4(void** context, int* eActivationMask, const char* commName,
                                                   uint64_t commHash, int nNodes, int nranks, int rank,
                                                   ncclDebugLogger_t logfn);

    OTEL_HIDDEN ncclResult_t profiler_otel_start_event_v4(void* context, void** eHandle,
                                                          ncclProfilerEventDescr_v4_t* eDescr);

    OTEL_HIDDEN ncclResult_t profiler_otel_record_event_state_v4(void* eHandle, ncclProfilerEventState_v4_t eState,
                                                                 ncclProfilerEventStateArgs_v4_t* eStateArgs);

#ifdef __cplusplus
}
#endif

#endif  // PROFILER_V4_COMPAT_H
