// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef MOCK_NCCL_H_
#define MOCK_NCCL_H_

#include <gmock/gmock.h>

#include <cstdint>

// Mock NCCL profiler event types
enum ncclProfilerEventType
{
    ncclProfileColl      = 0,
    ncclProfileGroup     = 1,
    ncclProfileP2p       = 2,
    ncclProfileProxyOp   = 4,
    ncclProfileProxyStep = 5
};

// Mock NCCL result type
typedef enum
{
    ncclSuccess       = 0,
    ncclInternalError = 1
} ncclResult_t;

// Mock class for NCCL profiler operations
class MockNCCL
{
public:
    virtual ~MockNCCL() = default;

    // Mock profiler operations
    MOCK_METHOD(ncclResult_t, ProfilerInit,
                (void** context, int* eActivationMask, const char* commName, uint64_t commHash, int nNodes, int nranks,
                 int rank));

    MOCK_METHOD(ncclResult_t, ProfilerStartEvent, (void* context, void** eHandle, void* eDescr));

    MOCK_METHOD(ncclResult_t, ProfilerStopEvent, (void* eHandle));

    MOCK_METHOD(ncclResult_t, ProfilerFinalize, (void* context));
};

// Global mock instance pointer
extern MockNCCL* g_mockNCCL;

#endif  // MOCK_NCCL_H_
