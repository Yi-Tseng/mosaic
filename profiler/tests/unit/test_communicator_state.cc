// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <set>
#include <thread>
#include <vector>

#include "../../communicator_state.h"
#include "../test_helpers.h"

class CommunicatorStateTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        reset_notify_window_ready_tracking();
        state            = new CommunicatorState();
        state->comm_name = "test_comm";
        state->comm_hash = 12345;
        state->nNodes    = 2;
        state->nranks    = 4;
        state->rank      = 0;
        state->commName  = "test_comm";
    }

    void TearDown() override
    {
        if (state)
        {
            delete state;
            state = nullptr;
        }
    }

    CommunicatorState* state = nullptr;
};

// =============================================================================
// Basic State & Allocation
// =============================================================================

TEST_F(CommunicatorStateTest, Creation)
{
    ASSERT_NE(state, nullptr);
    EXPECT_STREQ(state->comm_name, "test_comm");
    EXPECT_EQ(state->comm_hash, 12345);
    EXPECT_EQ(state->rank, 0);
    EXPECT_EQ(state->nranks, 4);
    EXPECT_EQ(state->nNodes, 2);
}

TEST_F(CommunicatorStateTest, InitialState)
{
    EXPECT_EQ(state->get_active_buffer_idx(), 0);

    WindowMetadata* window0 = state->get_window_metadata(0);
    ASSERT_NE(window0, nullptr);
    EXPECT_EQ(window0->state.load(), WINDOW_FILLING);
    EXPECT_EQ(window0->element_count.load(), 0);
    EXPECT_EQ(window0->in_progress_count.load(), 0);
    EXPECT_EQ(window0->proxy_ops_in_progress.load(), 0u);
    EXPECT_EQ(window0->kernel_ch_in_progress.load(), 0u);

    for (int i = 1; i < NUM_BUFFERS; i++)
    {
        WindowMetadata* window = state->get_window_metadata(i);
        ASSERT_NE(window, nullptr);
        EXPECT_EQ(window->state.load(), WINDOW_READY);
    }
}

TEST_F(CommunicatorStateTest, AllocateSlot)
{
    otelEventHandle_t* slot = state->allocate_event_slot();
    ASSERT_NE(slot, nullptr);

    WindowMetadata* window = state->get_window_metadata(0);
    EXPECT_EQ(window->element_count.load(), 1);
}

TEST_F(CommunicatorStateTest, MultipleAllocations)
{
    std::vector<otelEventHandle_t*> slots;
    for (int i = 0; i < 100; i++)
    {
        otelEventHandle_t* slot = state->allocate_event_slot();
        ASSERT_NE(slot, nullptr);
        slots.push_back(slot);
    }

    WindowMetadata* window = state->get_window_metadata(0);
    EXPECT_EQ(window->element_count.load(), 100);

    // Verify uniqueness using a set
    std::set<otelEventHandle_t*> unique_slots(slots.begin(), slots.end());
    EXPECT_EQ(unique_slots.size(), slots.size());
}

TEST_F(CommunicatorStateTest, ConcurrentAllocations)
{
    const int num_threads            = 4;
    const int allocations_per_thread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; t++)
    {
        threads.emplace_back(
            [this]()
            {
                for (int i = 0; i < 100; i++)
                {
                    otelEventHandle_t* slot = state->allocate_event_slot();
                    EXPECT_NE(slot, nullptr);
                }
            });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    WindowMetadata* window = state->get_window_metadata(0);
    EXPECT_EQ(window->element_count.load(), num_threads * allocations_per_thread);
}

// =============================================================================
// Window Closing & State Transitions
// =============================================================================

TEST_F(CommunicatorStateTest, WindowClosingNoInProgress)
{
    uint8_t buffer_idx = state->get_active_buffer_idx();
    EXPECT_EQ(buffer_idx, 0);

    state->trigger_window_closing(buffer_idx);

    WindowMetadata* window0 = state->get_window_metadata(0);
    EXPECT_EQ(window0->state.load(), WINDOW_PROCESSING);
    EXPECT_EQ(state->get_active_buffer_idx(), 1);

    WindowMetadata* window1 = state->get_window_metadata(1);
    EXPECT_EQ(window1->state.load(), WINDOW_FILLING);
    EXPECT_EQ(get_notify_window_ready_call_count(), 1);
}

TEST_F(CommunicatorStateTest, ZeroElementWindowClosing)
{
    uint8_t buffer_idx     = 0;
    WindowMetadata* window = state->get_window_metadata(buffer_idx);
    EXPECT_EQ(window->element_count.load(), 0);

    state->trigger_window_closing(buffer_idx);
    EXPECT_EQ(window->state.load(), WINDOW_PROCESSING);
    EXPECT_EQ(get_notify_window_ready_call_count(), 1);
}

TEST_F(CommunicatorStateTest, WindowClosingIdempotent)
{
    uint8_t buffer_idx = 0;
    state->trigger_window_closing(buffer_idx);
    EXPECT_EQ(state->get_active_buffer_idx(), 1);

    state->trigger_window_closing(buffer_idx);
    EXPECT_EQ(state->get_active_buffer_idx(), 1);
}

TEST_F(CommunicatorStateTest, WindowWrapAround)
{
    for (int i = 0; i < NUM_BUFFERS * 2; i++)
    {
        uint8_t current = state->get_active_buffer_idx();
        EXPECT_LT(current, NUM_BUFFERS);
        state->trigger_window_closing(current);
    }

    uint8_t final_idx = state->get_active_buffer_idx();
    EXPECT_LT(final_idx, NUM_BUFFERS);
}

// =============================================================================
// In-Progress Tracking
// =============================================================================

TEST_F(CommunicatorStateTest, InProgressTracking)
{
    uint8_t buffer_idx = 0;

    state->mark_operation_start(buffer_idx);
    state->mark_operation_start(buffer_idx);

    WindowMetadata* window = state->get_window_metadata(buffer_idx);
    EXPECT_EQ(window->in_progress_count.load(), 2);

    state->mark_operation_complete(buffer_idx);
    EXPECT_EQ(window->in_progress_count.load(), 1);

    state->mark_operation_complete(buffer_idx);
    EXPECT_EQ(window->in_progress_count.load(), 0);
}

TEST_F(CommunicatorStateTest, MarkOperationCompleteUnderflowProtection)
{
    uint8_t buffer_idx     = 0;
    WindowMetadata* window = state->get_window_metadata(buffer_idx);

    state->mark_operation_complete(buffer_idx);
    EXPECT_GE(window->in_progress_count.load(), 0);

    state->mark_operation_start(buffer_idx);
    state->mark_operation_complete(buffer_idx);
    state->mark_operation_complete(buffer_idx);
    EXPECT_EQ(window->in_progress_count.load(), 0);
}

TEST_F(CommunicatorStateTest, TransitionToProcessingOnLastComplete)
{
    uint8_t buffer_idx = 0;

    state->windows[buffer_idx].proxy_ops_in_progress.store(1, std::memory_order_release);
    state->windows[buffer_idx].kernel_ch_in_progress.store(1, std::memory_order_release);

    state->mark_operation_start(buffer_idx);
    state->trigger_window_closing(buffer_idx);

    WindowMetadata* window = state->get_window_metadata(buffer_idx);
    EXPECT_EQ(window->state.load(), WINDOW_CLOSING);

    state->mark_operation_complete(buffer_idx);
    EXPECT_EQ(window->state.load(), WINDOW_PROCESSING);
}

// =============================================================================
// ProxyOps and KernelCh Counters
// =============================================================================

TEST_F(CommunicatorStateTest, ProxyOpsInProgressCounter)
{
    uint8_t buffer_idx     = 0;
    WindowMetadata* window = state->get_window_metadata(buffer_idx);

    EXPECT_EQ(window->proxy_ops_in_progress.load(), 0u);

    window->proxy_ops_in_progress.fetch_add(1, std::memory_order_release);
    EXPECT_EQ(window->proxy_ops_in_progress.load(), 1u);

    window->proxy_ops_in_progress.fetch_add(1, std::memory_order_release);
    EXPECT_EQ(window->proxy_ops_in_progress.load(), 2u);

    window->proxy_ops_in_progress.fetch_sub(1, std::memory_order_release);
    EXPECT_EQ(window->proxy_ops_in_progress.load(), 1u);

    window->proxy_ops_in_progress.fetch_sub(1, std::memory_order_release);
    EXPECT_EQ(window->proxy_ops_in_progress.load(), 0u);
}

TEST_F(CommunicatorStateTest, KernelChInProgressCounter)
{
    uint8_t buffer_idx     = 0;
    WindowMetadata* window = state->get_window_metadata(buffer_idx);

    EXPECT_EQ(window->kernel_ch_in_progress.load(), 0u);

    window->kernel_ch_in_progress.fetch_add(1, std::memory_order_release);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), 1u);

    window->kernel_ch_in_progress.fetch_add(1, std::memory_order_release);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), 2u);

    window->kernel_ch_in_progress.fetch_sub(1, std::memory_order_release);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), 1u);

    window->kernel_ch_in_progress.fetch_sub(1, std::memory_order_release);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), 0u);
}

TEST_F(CommunicatorStateTest, ForceProcessingWhenAllChildCountersZero)
{
    uint8_t buffer_idx = 0;

    // in_progress_count > 0 but all child counters (proxy_ops, kernel_ch, groups) are 0.
    // trigger_window_closing should detect orphaned ops and force PROCESSING immediately.
    state->mark_operation_start(buffer_idx);

    WindowMetadata* window = state->get_window_metadata(buffer_idx);
    EXPECT_EQ(window->in_progress_count.load(), 1);
    EXPECT_EQ(window->proxy_ops_in_progress.load(), 0u);
    EXPECT_EQ(window->kernel_ch_in_progress.load(), 0u);

    state->trigger_window_closing(buffer_idx);

    // Should force-process immediately since child counters are all zero
    EXPECT_EQ(window->state.load(), WINDOW_PROCESSING);
    EXPECT_EQ(window->in_progress_count.load(), 0);
    EXPECT_EQ(get_notify_window_ready_call_count(), 1);
}

TEST_F(CommunicatorStateTest, StaysClosingWhileChildOpsActive)
{
    uint8_t buffer_idx = 0;

    state->windows[buffer_idx].proxy_ops_in_progress.store(2, std::memory_order_release);
    state->windows[buffer_idx].kernel_ch_in_progress.store(3, std::memory_order_release);

    state->mark_operation_start(buffer_idx);
    state->mark_operation_start(buffer_idx);

    state->trigger_window_closing(buffer_idx);

    WindowMetadata* window = state->get_window_metadata(buffer_idx);
    EXPECT_EQ(window->state.load(), WINDOW_CLOSING);

    // Complete one of two in_progress ops: still CLOSING
    state->mark_operation_complete(buffer_idx);
    EXPECT_EQ(window->state.load(), WINDOW_CLOSING);

    // Complete last in_progress op: should transition since proxy_ops and kernel_ch
    // counters are non-zero only in the accounting, the in_progress_count
    // reaching 0 is what triggers PROCESSING
    state->mark_operation_complete(buffer_idx);
    EXPECT_EQ(window->state.load(), WINDOW_PROCESSING);
}

TEST_F(CommunicatorStateTest, CountersResetOnWindowTransition)
{
    uint8_t buffer_idx = 0;

    state->windows[buffer_idx].proxy_ops_in_progress.store(5, std::memory_order_release);
    state->windows[buffer_idx].kernel_ch_in_progress.store(3, std::memory_order_release);

    state->trigger_window_closing(buffer_idx);

    // After closing, the *next* window should have clean counters
    uint8_t next_idx     = state->get_active_buffer_idx();
    WindowMetadata* next = state->get_window_metadata(next_idx);
    EXPECT_EQ(next->proxy_ops_in_progress.load(), 0u);
    EXPECT_EQ(next->kernel_ch_in_progress.load(), 0u);
}

// =============================================================================
// Automatic Window Transitions (Event Filling)
// =============================================================================

TEST_F(CommunicatorStateTest, AutomaticWindowTransitionOnEventFilling)
{
    uint8_t initial_buffer = state->get_active_buffer_idx();
    EXPECT_EQ(initial_buffer, 0);

    for (int i = 0; i < WINDOW_TRIGGER_COUNT; i++)
    {
        otelEventHandle_t* slot = state->allocate_event_slot();
        ASSERT_NE(slot, nullptr) << "Failed to allocate slot " << i;
    }

    EXPECT_EQ(state->get_active_buffer_idx(), 1);
    WindowMetadata* window0 = state->get_window_metadata(0);
    EXPECT_EQ(window0->state.load(), WINDOW_PROCESSING);
    EXPECT_EQ(window0->element_count.load(), WINDOW_TRIGGER_COUNT);

    EXPECT_EQ(get_notify_window_ready_call_count(), 1);

    const auto& calls = get_notify_window_ready_calls();
    ASSERT_EQ(calls.size(), 1u);
    EXPECT_EQ(calls[0].first, state);
    EXPECT_EQ(calls[0].second, initial_buffer);
}

TEST_F(CommunicatorStateTest, AutomaticWindowSwitchAndContinue)
{
    for (int i = 0; i < WINDOW_TRIGGER_COUNT; i++)
    {
        otelEventHandle_t* slot = state->allocate_event_slot();
        ASSERT_NE(slot, nullptr);
    }

    EXPECT_EQ(state->get_active_buffer_idx(), 1);
    EXPECT_EQ(get_notify_window_ready_call_count(), 1);

    for (int i = 0; i < 100; i++)
    {
        otelEventHandle_t* slot = state->allocate_event_slot();
        ASSERT_NE(slot, nullptr);
    }

    WindowMetadata* window1 = state->get_window_metadata(1);
    EXPECT_EQ(window1->element_count.load(), 100);
    EXPECT_EQ(window1->state.load(), WINDOW_FILLING);
}

TEST_F(CommunicatorStateTest, MultipleWindowAutomaticTransitions)
{
    for (int i = 0; i < WINDOW_TRIGGER_COUNT; i++)
    {
        otelEventHandle_t* slot = state->allocate_event_slot();
        ASSERT_NE(slot, nullptr);
    }

    WindowMetadata* window0 = state->get_window_metadata(0);
    EXPECT_EQ(window0->state.load(), WINDOW_PROCESSING);
    EXPECT_EQ(get_notify_window_ready_call_count(), 1);

    for (int i = 0; i < WINDOW_TRIGGER_COUNT; i++)
    {
        otelEventHandle_t* slot = state->allocate_event_slot();
        ASSERT_NE(slot, nullptr);
    }

    WindowMetadata* window1 = state->get_window_metadata(1);
    EXPECT_EQ(window1->state.load(), WINDOW_PROCESSING);
    EXPECT_EQ(get_notify_window_ready_call_count(), 2);
}

TEST_F(CommunicatorStateTest, WindowTransitionWithInProgressOperations)
{
    state->windows[0].proxy_ops_in_progress.store(1, std::memory_order_release);
    state->windows[0].kernel_ch_in_progress.store(1, std::memory_order_release);

    for (int i = 0; i < WINDOW_TRIGGER_COUNT; i++)
    {
        otelEventHandle_t* slot = state->allocate_event_slot();
        ASSERT_NE(slot, nullptr);
        state->mark_operation_start(0);
    }

    WindowMetadata* window0 = state->get_window_metadata(0);
    EXPECT_EQ(window0->state.load(), WINDOW_CLOSING);
    EXPECT_EQ(get_notify_window_ready_call_count(), 0);

    for (int i = 0; i < WINDOW_TRIGGER_COUNT / 2; i++)
    {
        state->mark_operation_complete(0);
    }

    EXPECT_EQ(window0->state.load(), WINDOW_CLOSING);

    for (int i = WINDOW_TRIGGER_COUNT / 2; i < WINDOW_TRIGGER_COUNT; i++)
    {
        state->mark_operation_complete(0);
    }

    EXPECT_EQ(window0->state.load(), WINDOW_PROCESSING);
    EXPECT_EQ(get_notify_window_ready_call_count(), 1);
}

TEST_F(CommunicatorStateTest, WindowWrapAroundWithAutoTransition)
{
    for (int window = 0; window < NUM_BUFFERS; window++)
    {
        for (int i = 0; i < WINDOW_TRIGGER_COUNT; i++)
        {
            otelEventHandle_t* slot = state->allocate_event_slot();
            if (slot == nullptr && window == NUM_BUFFERS - 1)
            {
                break;
            }
            ASSERT_NE(slot, nullptr);
        }
    }

    EXPECT_EQ(get_notify_window_ready_call_count(), NUM_BUFFERS);

    uint8_t active = state->get_active_buffer_idx();
    EXPECT_LT(active, NUM_BUFFERS);

    int processing_count = 0;
    for (int i = 0; i < NUM_BUFFERS; i++)
    {
        WindowMetadata* window = state->get_window_metadata(i);
        if (window->state.load() == WINDOW_PROCESSING)
        {
            processing_count++;
        }
    }
    EXPECT_GE(processing_count, NUM_BUFFERS - 1);
}

// =============================================================================
// Concurrent Stress Tests
// =============================================================================

TEST_F(CommunicatorStateTest, ConcurrentWindowClosing)
{
    const int num_threads = 10;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; t++)
    {
        threads.emplace_back(
            [this]()
            {
                for (int i = 0; i < 10; i++)
                {
                    uint8_t current = state->get_active_buffer_idx();
                    state->trigger_window_closing(current);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }
            });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_LT(state->get_active_buffer_idx(), NUM_BUFFERS);
}

TEST_F(CommunicatorStateTest, ConcurrentAllocateAndClose)
{
    std::atomic<bool> stop{false};
    std::atomic<int> alloc_count{0};
    std::atomic<int> close_count{0};

    auto allocator = [&]()
    {
        while (!stop.load())
        {
            otelEventHandle_t* slot = state->allocate_event_slot();
            if (slot)
            {
                alloc_count++;
            }
        }
    };

    auto closer = [&]()
    {
        while (!stop.load())
        {
            uint8_t idx = state->get_active_buffer_idx();
            state->trigger_window_closing(idx);
            close_count++;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 3; i++)
    {
        threads.emplace_back(allocator);
    }
    threads.emplace_back(closer);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true);

    for (auto& t : threads)
    {
        t.join();
    }

    EXPECT_GT(alloc_count.load(), 0);
    EXPECT_GT(close_count.load(), 0);
    EXPECT_LT(state->get_active_buffer_idx(), NUM_BUFFERS);
}

TEST_F(CommunicatorStateTest, StressTestAllocateAndCloseRepeatedly)
{
    for (int cycle = 0; cycle < 10; cycle++)
    {
        for (int i = 0; i < BUFFER_SIZE / 2; i++)
        {
            otelEventHandle_t* slot = state->allocate_event_slot();
            ASSERT_NE(slot, nullptr) << "Failed at cycle " << cycle << ", slot " << i;
        }

        uint8_t current = state->get_active_buffer_idx();
        state->trigger_window_closing(current);
    }

    otelEventHandle_t* slot = state->allocate_event_slot();
    EXPECT_NE(slot, nullptr);
}
