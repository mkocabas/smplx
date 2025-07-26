#!/bin/bash

# CUDA Sanity Check Build Script
# This script compiles a simple CUDA kernel to verify NVCC and GCC compatibility

set -e  # Exit on any error

echo "CUDA Sanity Check Build Script"
echo "=============================="

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed and in PATH."
    exit 1
fi

# Check if g++ is available
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ not found. Please ensure GCC is installed."
    exit 1
fi

# Print versions
echo "NVCC version:"
nvcc --version
echo
echo "GCC version:"
g++ --version | head -1
echo

# Set build directory
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# CUDA compilation flags
CUDA_FLAGS="-std=c++14 -O2 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90"
if [ -n "$CUDA_ARCH" ]; then
    CUDA_FLAGS="-std=c++14 -O2 -arch=$CUDA_ARCH"
fi

# Include paths (add more if needed)
INCLUDE_FLAGS=""

# Library paths and libraries
LIB_FLAGS="-lcudart"

echo "Compiling CUDA kernel..."
echo "Command: nvcc $CUDA_FLAGS $INCLUDE_FLAGS cuda_sanity_check.cu test_cuda.cpp -o $BUILD_DIR/test_cuda $LIB_FLAGS"

# Compile
nvcc $CUDA_FLAGS $INCLUDE_FLAGS cuda_sanity_check.cu test_cuda.cpp -o $BUILD_DIR/test_cuda $LIB_FLAGS

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo "Executable created: $BUILD_DIR/test_cuda"
    echo
    echo "To run the test:"
    echo "  cd $BUILD_DIR && ./test_cuda"
    echo "Or use the test script:"
    echo "  ./test.sh"
else
    echo "✗ Compilation failed!"
    exit 1
fi