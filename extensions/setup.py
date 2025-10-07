from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

if __name__=="__main__":
    
    sources = [
        "resampling.cu", "PISOtorch.cpp", "domain_structs.cpp", "PISO_multiblock_cuda_kernel.cu", "cg_solver_kernel.cu", "bicgstab_solver_kernel.cu", "grid_gen.cu",
        "eigenvalue.cu", "ortho_basis.cu", "matrix_vector_ops.cu", "matrix_vector_ops_grads.cu",
    ]
    macros=[("PYTHON_EXTENSION_BUILD", "1")]
    
    setup(name="PytorchPiso",
        ext_modules=[
            cpp_extension.CUDAExtension("PISOtorch", sources,
                include_dirs=["./"], 
                extra_compile_args={"cxx":["-fvisibility=hidden", "-Ofast"], "nvcc": [f"--threads={os.cpu_count()}", "-O3", "--use_fast_math"]},
                extra_link_args=[],
                define_macros=macros),
                
        ],
        cmdclass={"build_ext":cpp_extension.BuildExtension})
