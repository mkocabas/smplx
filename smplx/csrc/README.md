# CUDA Sanity Check

This directory contains a simple CUDA kernel implementation for sanity checking NVCC and GCC compatibility.

## Files

- `cuda_sanity_check.cu` - Simple CUDA kernel that performs vector addition
- `test_cuda.cpp` - C++ test program that calls the CUDA kernel
- `build.sh` - Build script to compile the CUDA code
- `test.sh` - Test script that builds and runs the sanity check
- `README.md` - This file

## Usage

### Quick Test
```bash
cd smplx/csrc
./test.sh
```

### Manual Build and Test
```bash
cd smplx/csrc
./build.sh
cd build
./test_cuda
```

### Custom CUDA Architecture
If you need to target a specific GPU architecture:
```bash
CUDA_ARCH=sm_80 ./build.sh
```

## What it does

1. **Vector Addition Kernel**: Implements a simple element-wise vector addition on GPU
2. **Device Info**: Displays information about available CUDA devices
3. **Verification**: Compares GPU results with CPU calculation to ensure correctness

## Requirements

- CUDA Toolkit (tested with CUDA 12.8)
- Compatible GCC compiler (GCC 11-12 recommended)
- NVIDIA GPU with CUDA support

## Troubleshooting

If compilation fails:
1. Check NVCC version: `nvcc --version`
2. Check GCC version: `gcc --version`
3. Ensure CUDA toolkit is properly installed
4. Try using conda GCC: `conda install -c conda-forge gcc_linux-64=11.4.0`

If runtime fails:
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA installation: `nvidia-smi` should show CUDA version
3. Check if GPU is accessible to your user account