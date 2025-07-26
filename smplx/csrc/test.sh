#!/bin/bash

# CUDA Sanity Check Test Script
# This script builds and runs the CUDA sanity check

set -e  # Exit on any error

echo "CUDA Sanity Check Test"
echo "====================="

# Build first
echo "Building CUDA kernel..."
./build.sh

echo
echo "Running CUDA test..."
echo "-------------------"

# Check if executable exists
if [ ! -f "build/test_cuda" ]; then
    echo "Error: Executable not found. Build may have failed."
    exit 1
fi

# Run the test
cd build
./test_cuda

# Check return code
if [ $? -eq 0 ]; then
    echo
    echo "🎉 CUDA sanity check completed successfully!"
    echo "   Your NVCC and GCC setup is working correctly."
else
    echo
    echo "❌ CUDA test failed!"
    echo "   Check your CUDA installation and GPU drivers."
    exit 1
fi