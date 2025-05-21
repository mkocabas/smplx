# CUDA SMPL-X Module (`cuda_smplx`)

## Overview

`cuda_smplx` provides a CUDA-accelerated implementation of core operations for the SMPL-X body model. The goal is to offer improved performance for both forward and backward passes compared to native PyTorch implementations, especially for batch processing.

This module includes:
- CUDA kernels for SMPL-X components (blend shapes, LBS, kinematic transformations).
- C++ wrappers and a PyTorch C++ extension using PyBind11.
- A `torch.autograd.Function` for integration into PyTorch models, enabling automatic differentiation.

## Prerequisites

- CUDA Toolkit (e.g., 11.x, 12.x - compatible with your PyTorch installation).
- PyTorch (ensure it's built with the same CUDA version you intend to use).
- C++ Compiler (g++, MSVC, etc., compatible with PyTorch C++ extensions).
- Python 3.x.

## Installation

1.  **Clone the repository (if applicable).**

2.  **Install Python Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    pip install torch numpy smplx open3d # Add other specific versions if necessary
    # For plotting in benchmarks:
    pip install matplotlib
    ```
    (A `requirements.txt` or `optional-requirements.txt` would typically list these).

3.  **Compile the CUDA Extension:**
    Navigate to the root directory of this project (where `setup.py` for `cuda_smplx_layer` is located).
    Run the setup script:
    ```bash
    python setup.py install
    ```
    Alternatively, to build in place for development:
    ```bash
    python setup.py build_ext --inplace
    ```
    This will compile the C++/CUDA code and create the `cuda_smplx_ops` Python module.

## Running Unit Tests

Unit tests are located in the `cuda_smplx/tests/` directory.

1.  **Download SMPL-X Models:**
    The tests require access to SMPL-X model files (`.npz` or `.pkl`). Download these from the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/) and place them in a directory.
    Set the environment variable `SMPLX_TEST_MODELS_PATH` to point to this directory. For example:
    ```bash
    export SMPLX_TEST_MODELS_PATH=/path/to/your/smplx_models
    ```
    If not set, tests will look in `data/smplx_models` relative to where tests are run.

2.  **Run Tests:**
    It's recommended to use `pytest`:
    ```bash
    pytest cuda_smplx/tests/
    ```
    Or run individual test files:
    ```bash
    python cuda_smplx/tests/test_forward.py
    python cuda_smplx/tests/test_backward.py
    ```

    **Note on `test_backward.py`**: The `gradcheck` test in `test_backward.py` is currently **expected to fail**. This is because the backward pass implementation within the `torch.autograd.Function` (`SMPLXCUDAAutoGrad`) is a placeholder due to a known issue (see "Known Issues/Limitations" below).

## Running Examples

Example scripts are located in `cuda_smplx/examples/`. Ensure you have installed dependencies and compiled the module. Also, ensure SMPL-X models are available as per the testing section.

1.  **Visualization Demo (`demo_visualization.py`):**
    This demo runs the forward pass with sample inputs and visualizes the resulting mesh using Open3D.
    ```bash
    python cuda_smplx/examples/demo_visualization.py
    ```
    Requires Open3D (`pip install open3d`).

2.  **Performance Benchmark (`benchmark_performance.py`):**
    This script benchmarks the forward pass of the CUDA implementation against the original `smplx` PyTorch library.
    ```bash
    python cuda_smplx/examples/benchmark_performance.py
    ```
    It will print timing results to the console and save a plot (`smplx_forward_benchmark.png`) if Matplotlib is installed.

## Module Structure

-   `cuda_smplx/csrc/`: Contains all CUDA (`.cu`) and C++ header (`.h`) source files for the core operations and their forward/backward passes.
-   `cuda_smplx/python/`: Contains the PyBind11 C++ code (`bindings.cpp`) that creates the Python interface for the CUDA/C++ functions and defines the `torch.autograd.Function`.
-   `cuda_smplx/tests/`: Unit tests.
-   `cuda_smplx/examples/`: Demo and benchmark scripts.
-   `cuda_smplx/docs/`: This README file.
-   `cuda_smplx_ops` (after compilation): The compiled Python module that you import (e.g., `import cuda_smplx_ops`).

## Known Issues/Limitations

-   **Autograd Backward Pass Incomplete**: The `SMPLXCUDAAutoGrad` class, which enables automatic differentiation via `cuda_smplx_ops.smplx_cuda_traced`, currently has a **placeholder implementation for its `backward` method**.
    -   **Reason**: The C++ function `smplx_forward_cuda` (which performs the main forward computation) needs to be modified to return all intermediate tensors (e.g., `v_shaped`, `J_shaped`, `rot_mats`, `A_global`, `v_posed`) that are required by the C++ `smplx_backward_cuda` function.
    -   Currently, `SMPLXCUDAAutoGrad::forward` does not save these necessary intermediates (because they are not returned by `smplx_forward_cuda`).
    -   **Impact**: `torch.autograd.gradcheck` will fail for `smplx_cuda_traced`. Gradients computed through `smplx_cuda_traced(...).backward()` will be incorrect.
    -   **Resolution**: This requires modifying `cuda_smplx/csrc/smplx_cuda_forward.cu` and `cuda_smplx/python/bindings.cpp` (specifically `SMPLXCUDAAutoGrad::forward` and `SMPLXCUDAAutoGrad::backward`) to correctly propagate and use these intermediate tensors.
