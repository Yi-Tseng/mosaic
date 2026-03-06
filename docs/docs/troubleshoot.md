---
icon: fontawesome/solid/bug-slash
title: Troubleshoot
---

<!--
SPDX-FileCopyrightText: 2025 Delos Data Inc
SPDX-License-Identifier: Apache-2.0
-->

# Common Issues

1. **Plugin not loading**:
    - Verify `NCCL_PROFILER_PLUGIN=otel` is set
    - Check that `LD_LIBRARY_PATH` includes the plugin directory
    - Ensure the plugin library exists and is executable

2. **No metrics being sent**:
    - Verify `NCCL_PROFILER_OTEL_TELEMETRY_ENDPOINT` is correctly set
    - Check that `NCCL_PROFILER_OTEL_ENABLE=1` (default)
    - Check that `NCCL_PROFILER_OTEL_TELEMETRY_ENABLE=1` (default)
    - Verify OpenTelemetry collector is running and accessible

3. **Connection issues**:
    - Verify the OpenTelemetry collector is running and accessible
    - Check network connectivity to the collector endpoint
    - Ensure the endpoint URL format is correct (`http://host:port`)

4. **Buffer overflow warnings**:
    - Increase window trigger settings if events are generated too quickly
    - Check that telemetry thread is processing windows in time
    - Verify system has sufficient CPU for background processing

# Debug Logging

Enable trace logging for detailed debugging:

```bash
# Build with trace logging
make TRACE=1

# Run with NCCL debug output
export NCCL_DEBUG=INFO
export NCCL_PROFILER_PLUGIN=otel
# ... other environment variables
```

This will provide detailed logs about plugin initialization, metric collection, and telemetry export.
