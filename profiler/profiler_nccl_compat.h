// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0
//
// Compatibility header for building against different NCCL profiler API versions.
//
// When NCCL provides profiler_v5.h, it is used directly.  When only v4 is
// available (older NCCL), this header supplies ABI-compatible v5 type
// definitions and the event-type constants that were introduced alongside v5.
// The resulting .so exports both ncclProfiler_v4 and ncclProfiler_v5 symbols
// regardless of which NCCL it was compiled against, so a newer NCCL loading
// the plugin will resolve ncclProfiler_v5 and an older one will fall back
// to ncclProfiler_v4.

#ifndef PROFILER_NCCL_COMPAT_H
#define PROFILER_NCCL_COMPAT_H

#include <sys/types.h>

#include <cstdint>

// Core NCCL headers (always available when a profiler API exists)
#include "nccl.h"
#include "nccl_common.h"

// ============================================================================
// Include NCCL profiler headers that are present
// ============================================================================

#if __has_include("plugin/nccl_profiler.h")
#include "plugin/nccl_profiler.h"
#endif

#if __has_include("plugin/profiler/profiler_v4.h")
#include "plugin/profiler/profiler_v4.h"
#endif

#if __has_include("plugin/profiler/profiler_v5.h")
#include "plugin/profiler/profiler_v5.h"
#endif

// ============================================================================
// Fallback: v5 type definitions for v4-only NCCL
//
// The guard checks the header-include macro that profiler_v5.h sets.
// If profiler_v5.h was included above, these definitions are skipped.
// ============================================================================

#ifndef PROFILER_V5_H_

// Event types introduced alongside profiler v5.
// In v4-only NCCL these values never appear in actual events, but our code
// references them in filter/skip checks so the constants must exist.
enum
{
    ncclProfileGroupApi     = (1 << 8),
    ncclProfileCollApi      = (1 << 9),
    ncclProfileP2pApi       = (1 << 10),
    ncclProfileKernelLaunch = (1 << 11),
};

// State enum typedef (v4 already defines ncclProfilerEventState_v4_t as a
// typedef of ncclProfilerEventState_t; v5 is the same underlying type)
typedef ncclProfilerEventState_t ncclProfilerEventState_v5_t;

// v5 event descriptor — superset of v4: wider type field (uint64_t) and
// additional union members for API-level / kernel-launch events.
typedef struct
{
    uint64_t type;
    void* parentObj;
    int rank;
    union
    {
        struct
        {
            bool graphCaptured;
            int groupDepth;
        } groupApi;

        struct
        {
            const char* func;
            size_t count;
            const char* datatype;
            int root;
            void* stream;
            bool graphCaptured;
        } collApi;

        struct
        {
            const char* func;
            size_t count;
            const char* datatype;
            void* stream;
            bool graphCaptured;
        } p2pApi;

        struct
        {
            void* stream;
        } kernelLaunch;

        struct
        {
            uint64_t seqNumber;
            const char* func;
            void const* sendBuff;
            void* recvBuff;
            size_t count;
            int root;
            const char* datatype;
            uint8_t nChannels;
            uint8_t nWarps;
            const char* algo;
            const char* proto;
            void* parentGroup;
        } coll;

        struct
        {
            const char* func;
            void* buff;
            const char* datatype;
            size_t count;
            int peer;
            uint8_t nChannels;
            void* parentGroup;
        } p2p;

        struct
        {
            pid_t pid;
            uint8_t channelId;
            int peer;
            int nSteps;
            int chunkSize;
            int isSend;
        } proxyOp;

        struct
        {
            int step;
        } proxyStep;

        struct
        {
            uint8_t channelId;
            uint64_t pTimer;
        } kernelCh;

        struct
        {
            int64_t id;
            void* data;
        } netPlugin;
    };
} ncclProfilerEventDescr_v5_t;

// v5 state args — same layout as v4
typedef union
{
    struct
    {
        size_t transSize;
    } proxyStep;

    struct
    {
        int appendedProxyOps;
    } proxyCtrl;

    struct
    {
        void* data;
    } netPlugin;

    struct
    {
        uint64_t pTimer;
    } kernelCh;
} ncclProfilerEventStateArgs_v5_t;

// v5 plugin structure
typedef struct
{
    const char* name;

    ncclResult_t (*init)(void** context, uint64_t commId, int* eActivationMask, const char* commName, int nNodes,
                         int nranks, int rank, ncclDebugLogger_t logfn);
    ncclResult_t (*startEvent)(void* context, void** eHandle, ncclProfilerEventDescr_v5_t* eDescr);
    ncclResult_t (*stopEvent)(void* eHandle);
    ncclResult_t (*recordEventState)(void* eHandle, ncclProfilerEventState_v5_t eState,
                                     ncclProfilerEventStateArgs_v5_t* eStateArgs);
    ncclResult_t (*finalize)(void* context);
} ncclProfiler_v5_t;

#endif  // PROFILER_V5_H_

#endif  // PROFILER_NCCL_COMPAT_H
