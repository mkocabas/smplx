#!/bin/bash

# Build script for LBS CUDA extension

set -e

echo "Building LBS CUDA Extension"
echo "=========================="

# Check dependencies
if ! python -c "import torch" 2>/dev/null; then
    echo "Error: PyTorch not found. Please install PyTorch first."
    exit 1
fi

if ! python -c "import pybind11" 2>/dev/null; then
    echo "Error: pybind11 not found. Installing..."
    pip install pybind11
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/
rm -f lbs_cuda_ext*.so

# Build extension
echo "Building CUDA extension..."
python setup.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo "Extension created: lbs_cuda_ext*.so"
    echo
    echo "To test the extension:"
    echo "  python test_lbs_cuda.py"
else
    echo "✗ Build failed!"
    exit 1
fi