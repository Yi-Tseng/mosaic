// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef OTEL_TELEMETRY_H_
#define OTEL_TELEMETRY_H_

#include <stddef.h>

#include "communicator_state.h"

/**
 * @brief Initialize the telemetry collection system.
 *
 * Sets up OpenTelemetry metrics, creates metric instruments, and starts the background
 * telemetry thread for asynchronous metric processing and export.
 *
 * @note Only initializes if NCCL_PROFILER_OTEL_TELEMETRY_ENABLE is set (default: 1).
 * @note Called automatically on first communicator initialization.
 * @note Thread-safe: uses atomic counters to ensure single initialization.
 */
void profiler_otel_telemetry_init();

/**
 * @brief Cleanup the telemetry collection system.
 *
 * Stops the telemetry thread, cleans up OpenTelemetry resources, and resets state.
 * Called when the last communicator is finalized.
 *
 * @note Only cleans up if telemetry was initialized.
 * @note Thread-safe: uses atomic counters to track active communicators.
 */
void profiler_otel_telemetry_cleanup();

/**
 * @brief Notify the telemetry thread that a window is ready for processing.
 *
 * Called when a window transitions to PROCESSING state. Registers the communicator
 * state if not already registered and wakes up the telemetry thread to process the window.
 *
 * @param[in] commState Communicator state containing the window to process.
 * @param[in] window_idx Index of the window that is ready (0-3).
 *
 * @note Thread-safe: uses mutexes for communicator state registration.
 */
void profiler_otel_telemetry_notify_window_ready(struct CommunicatorState* commState, int window_idx);

#endif  // OTEL_TELEMETRY_H_
