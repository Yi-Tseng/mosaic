// SPDX-FileCopyrightText: 2025 Delos Data Inc
// SPDX-License-Identifier: Apache-2.0

#ifndef PROFILER_OTEL_ADAPTER_NCCL_PARAM_H_
#define PROFILER_OTEL_ADAPTER_NCCL_PARAM_H_

#include <stdint.h>
#include <stdlib.h>

#include <mutex>
#include <string>

// Self-contained parameter loading function
static inline void otelLoadParam(const char* env, int64_t deftVal, int64_t uninitialized, int64_t* cache)
{
    if (__builtin_expect(__atomic_load_n(cache, __ATOMIC_RELAXED) == uninitialized, false))
    {
        const char* str = getenv(env);
        int64_t value   = deftVal;

        if (str && str[0] != '\0')
        {
            char* endptr;
            long long parsed = strtoll(str, &endptr, 0);
            if (endptr != str && *endptr == '\0')
            {
                value = (int64_t)parsed;
            }
        }

        __atomic_store_n(cache, value, __ATOMIC_RELAXED);
    }
}

// Self-contained string parameter loading function.
// Note: this helper does not provide synchronization by itself; callers should ensure one-time initialization.
static inline void otelLoadStringParam(const char* env, const char* deftVal, std::string* cache)
{
    const char* str = getenv(env);
    if (str && str[0] != '\0')
        *cache = str;
    else
        *cache = deftVal;
}

#ifdef UNIT_TESTING
// In unit tests, prefer correctness/reproducibility over caching: allow env var changes at runtime.
#define NCCL_PARAM1(name, env, deftVal)                                                                                \
    int64_t ncclParam##name()                                                                                          \
    {                                                                                                                  \
        int64_t value   = deftVal;                                                                                     \
        const char* str = getenv("NCCL_" env);                                                                         \
        if (str && str[0] != '\0')                                                                                     \
        {                                                                                                              \
            char* endptr;                                                                                              \
            long long parsed = strtoll(str, &endptr, 0);                                                               \
            if (endptr != str && *endptr == '\0') value = (int64_t)parsed;                                             \
        }                                                                                                              \
        return value;                                                                                                  \
    }
#else
#define NCCL_PARAM1(name, env, deftVal)                                                                                \
    int64_t ncclParam##name()                                                                                          \
    {                                                                                                                  \
        constexpr int64_t uninitialized = INT64_MIN;                                                                   \
        static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value.");                   \
        static int64_t cache = uninitialized;                                                                          \
        if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false))                       \
        {                                                                                                              \
            otelLoadParam("NCCL_" env, deftVal, uninitialized, &cache);                                                \
        }                                                                                                              \
        return cache;                                                                                                  \
    }
#endif

#ifdef UNIT_TESTING
#define NCCL_STRING_PARAM1(name, env, deftVal)                                                                         \
    const char* ncclParam##name()                                                                                      \
    {                                                                                                                  \
        static thread_local std::string tmp;                                                                           \
        otelLoadStringParam("NCCL_" env, deftVal, &tmp);                                                               \
        return tmp.c_str();                                                                                            \
    }
#else
#define NCCL_STRING_PARAM1(name, env, deftVal)                                                                         \
    const char* ncclParam##name()                                                                                      \
    {                                                                                                                  \
        static std::once_flag once;                                                                                    \
        static std::string cache;                                                                                      \
        std::call_once(once, []() { otelLoadStringParam("NCCL_" env, deftVal, &cache); });                             \
        return cache.c_str();                                                                                          \
    }
#endif

#define __OTEL_PARAM(name, env, deftVal) NCCL_PARAM1(name, env, deftVal)

#define __OTEL_STRING_PARAM(name, env, deftVal) NCCL_STRING_PARAM1(name, env, deftVal)

#define __OTEL_GET_PARAM(name) ncclParam##name()

#define OTEL_PARAM(name, env, deftVal) __OTEL_PARAM(name, env, deftVal)

#define OTEL_STRING_PARAM(name, env, deftVal) __OTEL_STRING_PARAM(name, env, deftVal)

#define OTEL_GET_PARAM(name) (__OTEL_GET_PARAM(name))

// Forward declarations for parameters (defined in profiler_otel.cc)
const char* ncclParamLinearRegressionMode();
int64_t ncclParamScaleUpNetworkPct();

#endif  // PROFILER_OTEL_ADAPTER_NCCL_PARAM_H_
