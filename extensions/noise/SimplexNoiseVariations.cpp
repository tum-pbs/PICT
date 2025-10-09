
#include "simplex_noise.hpp"

using namespace SimplexNoise;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	
	py::enum_<NoiseVariation>(m, "NoiseVariation")
		.value("SIMPLEX", NoiseVariation::SIMPLEX)
		.value("WORLEY", NoiseVariation::WORLEY)
		.value("FRACTAL_BROWNIAN_MOTION", NoiseVariation::FRACTAL_BROWNIAN_MOTION)
		.value("RIDGED", NoiseVariation::RIDGED)
		.value("RIDGED_MULTI_FRACTAL", NoiseVariation::RIDGED_MULTI_FRACTAL)
		.value("GRADIENT", NoiseVariation::GRADIENT)
		.value("GRADIENT_FBM", NoiseVariation::GRADIENT_FBM)
		.value("CURL", NoiseVariation::CURL)
		.value("CURL_FBM", NoiseVariation::CURL_FBM)
		.export_values();
	
	m.def("GenerateSimplexNoiseVariation", &GenerateSimplexNoiseVariation, "Create a tensor with simplex noise.",
		py::arg("output_shape"), py::arg("GPUdevice"), py::arg("scale"), py::arg("offset"), py::arg("variation"),
		py::arg("ridgeOffset")=1.0f, py::arg("octaves")=4, py::arg("lacunarity")=2.0f, py::arg("gain")=0.5f);
}