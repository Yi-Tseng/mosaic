#!/bin/bash
# SPDX-FileCopyrightText: 2025 Delos Data Inc
# SPDX-License-Identifier: Apache-2.0
# Script to run NCCL Profiler OTEL Plugin unit tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}NCCL Profiler OTEL Plugin - Unit Tests${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""

# Check if NCCL_PATH is set (optional for tests)
if [ -z "$NCCL_PATH" ]; then
    echo -e "${YELLOW}Warning: NCCL_PATH not set. Some tests may not compile correctly.${NC}"
    echo -e "${YELLOW}Set it with: export NCCL_PATH=/path/to/nccl${NC}"
    echo ""
fi

# Parse command line arguments
VERBOSE=0
FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -f|--filter)
            FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose    Run tests with verbose output"
            echo "  -f, --filter     Run only tests matching the filter pattern"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all tests"
            echo "  $0 -v                        # Run with verbose output"
            echo "  $0 -f ProfilerOtelTest.*     # Run only ProfilerOtelTest tests"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build tests
echo -e "${GREEN}Building tests...${NC}"
cd "$(dirname "$0")/.."
make test-build

# Run tests
echo ""
echo -e "${GREEN}Running tests...${NC}"
cd tests/build

if [ -n "$FILTER" ]; then
    echo -e "${YELLOW}Filter: $FILTER${NC}"
    if [ $VERBOSE -eq 1 ]; then
        ./nccl_profiler_otel_tests --gtest_filter="$FILTER" --gtest_verbose=1
    else
        ./nccl_profiler_otel_tests --gtest_filter="$FILTER"
    fi
else
    if [ $VERBOSE -eq 1 ]; then
        ./nccl_profiler_otel_tests --gtest_verbose=1
    else
        ./nccl_profiler_otel_tests
    fi
fi

echo ""
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}Tests completed successfully!${NC}"
echo -e "${GREEN}====================================${NC}"
