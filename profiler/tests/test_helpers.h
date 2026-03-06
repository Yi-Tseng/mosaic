// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <utility>
#include <vector>

#include "../communicator_state.h"
#include "../profiler_otel.h"

// Helper functions for tests to reset global state
// These should only be used in test files where UNIT_TESTING is defined
// The setInitialized, setPid functions are already
// declared in profiler_otel.h when UNIT_TESTING is defined

inline void resetProfilerState()
{
    extern ncclDebugLogger_t otel_log_func;
    setInitialized(0);
    otel_log_func = nullptr;
    setPid(0);
}

// Mock tracking functions (implemented in test_mocks.cc)
void reset_notify_window_ready_tracking();
int get_notify_window_ready_call_count();
std::vector<std::pair<CommunicatorState*, int>> get_notify_window_ready_calls();  // Returns copy for thread-safety

#endif  // TEST_HELPERS_H
