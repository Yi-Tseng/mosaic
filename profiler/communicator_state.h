// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef COMMUNICATOR_STATE_H_
#define COMMUNICATOR_STATE_H_

#include <atomic>
#include <cstdint>
#include <string>

#include "events.h"

#define BUFFER_SIZE             100000  // 100k elements per buffer
#define NUM_BUFFERS             4       // 4 circular buffers
#define WINDOW_TRIGGER_COUNT    50000   // Trigger window closing at 50k elements
#define WINDOW_TRIGGER_TIME_SEC 5       // Trigger window closing after 5 seconds

/**
 * Window states for the circular buffer state machine
 */
enum WindowState : uint8_t
{
    WINDOW_FILLING = 0,  // Window is actively being filled
    WINDOW_CLOSING,      // Window has reached trigger, closing in-progress operations
    WINDOW_PROCESSING,   // Window is being processed by background thread
    WINDOW_READY         // Window is cleared and ready to be reused
};

/**
 * Per-window metadata for tracking state and in-progress operations
 */
struct WindowMetadata
{
    std::atomic<WindowState> state;
    std::atomic<uint32_t> element_count;          // Number of elements in this window
    std::atomic<uint32_t> in_progress_count;      // Number of in-progress coll/p2p/transfers
    std::atomic<uint32_t> groups_in_progress;     // Number of in-progress Group operations
    std::atomic<uint32_t> proxy_ops_in_progress;  // Number of ProxyOps currently in-progress in this window
    std::atomic<uint32_t> kernel_ch_in_progress;  // Number of KernelCh events currently in-progress in this window
    std::atomic<uint32_t> pending_first_child;    // Coll/P2P events awaiting their first child (KernelCh/ProxyOp)
    double start_time;                            // Time when window started filling

    WindowMetadata()
        : state(WINDOW_READY),
          element_count(0),
          in_progress_count(0),
          groups_in_progress(0),
          proxy_ops_in_progress(0),
          kernel_ch_in_progress(0),
          pending_first_child(0),
          start_time(0.0)
    {
    }
};

/**
 * Communicator state managing circular buffers for event storage
 * Each communicator has its own set of 4 buffers that rotate independently
 *
 * Note: Buffers are heap-allocated to avoid stack overflow (~40 MB total)
 */
struct CommunicatorState
{
    // 4 circular buffer arrays (heap-allocated to avoid stack overflow)
    // Each buffer is NUM_BUFFERS x BUFFER_SIZE = 4 x 100k = 400k events
    // With ~100 bytes per event, this is ~40 MB total
    otelEventHandle_t** buffers;

    // Metadata for each window
    WindowMetadata windows[NUM_BUFFERS];

    // Active buffer index (0-3)
    std::atomic<uint8_t> active_buffer_idx;

    // Next element index within active buffer
    std::atomic<uint32_t> next_element_idx;

    // Communicator metadata
    const char* comm_name;
    uint64_t comm_hash;
    int rank;
    int nranks;
    int nNodes;

    // Rank and hostname information
    std::string hostname;        // Hostname of the node running this rank
    int local_rank;              // Local rank within the node
    std::string gpu_pci_bus_id;  // GPU PCI BUS ID (e.g., "00000000:01:00.0")
    std::string gpu_uuid;        // GPU UUID

    // Communicator type classification
    // P2P communicators always have exactly 2 ranks (point-to-point)
    // Collective communicators have more than 2 ranks
    enum class CommType
    {
        UNKNOWN = 0,
        P2P,        // Exactly 2 ranks - point-to-point inter-pipeline communication
        COLLECTIVE  // More than 2 ranks - collective operations (AllReduce, etc.)
    };
    CommType comm_type;  // Inferred from nranks

    // Scale-up execution mode classification.
    //
    // We assume a communicator is either CUDA-Graph-driven or not. Once determined,
    // the mode is persisted here and used to:
    //  - annotate exported OTEL metrics, and
    //  - select the appropriate scale-up aggregation path (CUDA Graph vs non-CUDA Graph).
    enum class ScaleUpExecMode : uint8_t
    {
        UNKNOWN = 0,
        NON_CUDA_GRAPH,
        CUDA_GRAPH
    };
    std::atomic<uint8_t> scaleUpExecMode;  // stores ScaleUpExecMode as uint8_t

    bool isScaleUpCudaGraphDriven() const
    {
        return scaleUpExecMode.load(std::memory_order_acquire) == static_cast<uint8_t>(ScaleUpExecMode::CUDA_GRAPH);
    }
    const char* getScaleUpExecModeString() const
    {
        auto mode = static_cast<ScaleUpExecMode>(scaleUpExecMode.load(std::memory_order_acquire));
        switch (mode)
        {
            case ScaleUpExecMode::NON_CUDA_GRAPH:
                return "non_cuda_graph";
            case ScaleUpExecMode::CUDA_GRAPH:
                return "cuda_graph";
            default:
                return "unknown";
        }
    }

    // Get human-readable communicator type string
    const char* getCommTypeString() const
    {
        switch (comm_type)
        {
            case CommType::P2P:
                return "P2P";
            case CommType::COLLECTIVE:
                return "COLLECTIVE";
            default:
                return "UNKNOWN";
        }
    }

    // Window management configuration
    double window_timeout_usec;  // Window closing timeout in microseconds

    // Compatibility aliases
    std::string commName;  // String version for easier use

    /**
     * @brief Construct a CommunicatorState with initialized buffers.
     *
     * Initializes all 4 circular buffers and sets the first buffer to FILLING state.
     */
    CommunicatorState();

    /**
     * @brief Destructor for CommunicatorState.
     *
     * Frees heap-allocated buffers.
     */
    ~CommunicatorState();

    /**
     * @brief Allocate a slot in the circular buffer (lock-free).
     *
     * Routes events to the appropriate window based on parent object:
     * - If parentObj is in a CLOSING window, route to that window
     * - Otherwise route to the active FILLING window
     * - Checks for window closing triggers before allocation
     *
     * @param[in] parentObj Parent event handle (nullptr for root events).
     * @param[in] current_time Current time in microseconds (for time-based closing).
     *
     * @return Pointer to allocated event slot, or nullptr if allocation failed.
     *
     * @note Lock-free implementation using atomic operations.
     * @note Retries on race conditions or invalid window states.
     */
    otelEventHandle_t* allocate_event_slot(void* parentObj = nullptr, double current_time = 0.0);

    /**
     * @brief Check if a window should close.
     *
     * Windows close when either:
     * - Element count reaches WINDOW_TRIGGER_COUNT, OR
     * - Time elapsed exceeds window_timeout_usec (configurable)
     *
     * @param[in] buffer_idx Window index to check (0-3).
     * @param[in] current_time Current time in microseconds.
     *
     * @return true if window should close, false otherwise.
     *
     * @note Only checks FILLING windows (other states return false).
     */
    bool should_close_window(uint8_t buffer_idx, double current_time);

    /**
     * @brief Set window start time if not already set.
     *
     * Sets the start_time for time-based window closing. Only sets if:
     * - start_time is 0.0 (unset), AND
     * - element_count > 0 (window has events)
     *
     * @param[in] buffer_idx Window index (0-3).
     * @param[in] current_time Current time in microseconds.
     */
    void set_window_start_time_if_needed(uint8_t buffer_idx, double current_time);

    /**
     * @brief Trigger window closing and switch to next buffer.
     *
     * Transitions a window from FILLING to CLOSING, switches active buffer to the
     * next window, and initializes the next window for FILLING. If no operations
     * are in-progress, immediately transitions to PROCESSING.
     *
     * @param[in] buffer_idx Window index to close (0-3).
     *
     * @note Thread-safe: uses compare-and-swap for state transitions.
     * @note If next window is not READY/PROCESSING, forces it to READY.
     */
    void trigger_window_closing(uint8_t buffer_idx);

    /**
     * @brief Switch window from CLOSING to PROCESSING.
     *
     * Called when in_progress_count reaches 0. Transitions the window to PROCESSING
     * and notifies the telemetry thread. The active buffer was already switched
     * in trigger_window_closing().
     *
     * @param[in] current_idx Window index to transition (0-3).
     *
     * @note Thread-safe: uses compare-and-swap for state transition.
     */
    void switch_to_next_buffer(uint8_t current_idx);

    /**
     * @brief Mark an operation as in-progress.
     *
     * Increments the window's in_progress_count. Called when a Coll/P2P starts
     * and when a ProxyOp starts (for send-side operations).
     *
     * @param[in] buffer_idx Window index (0-3).
     *
     * @note Thread-safe: uses atomic increment.
     */
    void mark_operation_start(uint8_t buffer_idx);

    /**
     * @brief Mark an operation as completed.
     *
     * Decrements the window's in_progress_count. Called when a ProxyOp completes.
     * If this was the last operation, triggers window transition to PROCESSING.
     *
     * @param[in] buffer_idx Window index (0-3).
     *
     * @note Thread-safe: uses atomic decrement.
     * @note Only processes CLOSING or FILLING windows (ignores PROCESSING/READY).
     * @note Prevents underflow by checking count before decrementing.
     */
    void mark_operation_complete(uint8_t buffer_idx);

    /**
     * @brief Get the current active buffer index.
     *
     * @return Active buffer index (0-3).
     *
     * @note Thread-safe: uses atomic load.
     */
    uint8_t get_active_buffer_idx() const;

    /**
     * @brief Get window metadata for a buffer.
     *
     * @param[in] buffer_idx Window index (0-3).
     *
     * @return Pointer to WindowMetadata, or nullptr if buffer_idx is invalid.
     */
    WindowMetadata* get_window_metadata(uint8_t buffer_idx);
};

/**
 * @brief Helper function to get next event handle from circular buffer.
 *
 * Wrapper around CommunicatorState::allocate_event_slot() for easier use.
 *
 * @param[in] state Communicator state containing buffers.
 * @param[in] parentObj Parent event handle (nullptr for root events).
 * @param[in] current_time Current time in microseconds.
 *
 * @return Pointer to allocated event slot, or nullptr if allocation failed.
 */
otelEventHandle_t* get_next_event_handle(CommunicatorState* state, void* parentObj, double current_time);

#endif  // COMMUNICATOR_STATE_H_
