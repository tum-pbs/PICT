from setuptools import setup, Extension
from torch.utils import cpp_extension

if __name__=="__main__":
	setup(name="SimplexNoiseVariations",
		ext_modules=[
			cpp_extension.CUDAExtension("SimplexNoiseVariations", ["simplex_noise.cu", "SimplexNoiseVariations.cpp"], include_dirs=["./", "../"], extra_compile_args={"cxx":["-fvisibility=hidden"]})
		],
		cmdclass={"build_ext":cpp_extension.BuildExtension})