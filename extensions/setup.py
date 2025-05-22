from setuptools import setup, Extension
from torch.utils import cpp_extension

if __name__=="__main__":
	setup(name="PytorchPiso",
		ext_modules=[
			cpp_extension.CUDAExtension("PISOtorch", ["resampling.cu", "PISOtorch.cpp", "domain_structs.cpp", "PISO_multiblock_cuda_kernel.cu", "cg_solver_kernel.cu", "bicgstab_solver_kernel.cu", "grid_gen.cu"], include_dirs=["./"], extra_compile_args={"cxx":["-fvisibility=hidden"], "nvcc":["-O2"]})  #"domain_structs_pybind.cpp", 
		],
		cmdclass={"build_ext":cpp_extension.BuildExtension})
