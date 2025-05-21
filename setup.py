from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
# import torch # Not strictly needed for setup.py unless get_cuda_arch_list is used directly here

def get_cuda_arch_flags():
    # This is a common way to get PyTorch's detected CUDA architecture
    # However, torch.utils.cpp_extension.CUDAExtension usually handles this.
    # For explicit control or if issues arise:
    # import torch # Moved inside if needed
    # arch_list = torch.cuda.get_arch_list()
    # if arch_list:
    #    flags = []
    #    for arch in arch_list:
    #        num = arch.split('_')[-1]
    #        flags.append(f'-gencode=arch=compute_{num},code=sm_{num}')
    #    return flags
    return [] # Let CUDAExtension handle it by default

# Define source files
csrc_dir = os.path.join('cuda_smplx', 'csrc')
python_dir = os.path.join('cuda_smplx', 'python')

source_files = [
    os.path.join(csrc_dir, 'blend_shapes.cu'),
    os.path.join(csrc_dir, 'vertices_to_joints.cu'),
    os.path.join(csrc_dir, 'batch_rodrigues.cu'),
    os.path.join(csrc_dir, 'batch_rigid_transform.cu'),
    os.path.join(csrc_dir, 'skinning.cu'),
    os.path.join(csrc_dir, 'smplx_cuda_forward.cu'),
    os.path.join(csrc_dir, 'smplx_cuda_backward.cu'),
    os.path.join(python_dir, 'bindings.cpp'),
]

# Define include directories
include_dirs = [
    os.path.abspath(csrc_dir), # To find .h files in csrc
]

# Define extra compile arguments
# Add -std=c++17 for C++17 features if used (Pybind11 benefits from it)
cpp_args = ['-O3', '-std=c++17'] 
nvcc_args = ['-O3', '-std=c++17'] 
# nvcc_args.extend(get_cuda_arch_flags()) # Optionally add specific arch flags

setup(
    name='cuda_smplx_layer', # Name of the package
    version='0.1.0',
    author='AI Agent', # Updated author
    description='CUDA implementation for SMPL-X layer components',
    long_description='Provides an optimized CUDA forward and backward pass for SMPL-X body model components, including autograd support.',
    ext_modules=[
        CUDAExtension(
            name='cuda_smplx_ops', # How it's imported: import cuda_smplx_ops
            sources=source_files,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': cpp_args, 'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(where='.', include=['cuda_smplx_ops*']), 
    # The prompt notes this might not be strictly necessary for a pure extension.
    # Keeping as per prompt for now.
    zip_safe=False, 
)
