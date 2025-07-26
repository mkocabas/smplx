from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import pybind11

# Get CUDA and PyTorch include paths
torch_include = torch.utils.cpp_extension.include_paths()
cuda_include = ['/usr/local/cuda/include']

ext_modules = [
    CUDAExtension(
        'lbs_cuda_ext',
        [
            'lbs_cuda_binding.cpp',
            'lbs_cuda.cu',
        ],
        include_dirs=[
            '/usr/local/cuda/include',
            *torch_include,
            pybind11.get_cmake_dir() + '/../../../include'
        ],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17'],
            'nvcc': [
                '-O3',
                '-std=c++17',
                '--expt-relaxed-constexpr',
                '-gencode', 'arch=compute_75,code=sm_75',
                '-gencode', 'arch=compute_80,code=sm_80',
                '-gencode', 'arch=compute_86,code=sm_86',
                '-gencode', 'arch=compute_90,code=sm_90',
            ]
        }
    )
]

setup(
    name='lbs_cuda_ext',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)