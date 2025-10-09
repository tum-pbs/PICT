#pragma once

#ifndef _INCLUDE_NOISE_SETTINGS
#define _INCLUDE_NOISE_SETTINGS
#include <torch/extension.h>
#include <vector>


namespace SimplexNoise{
enum NoiseVariation{
	SIMPLEX=0,
	WORLEY=1,
	FRACTAL_BROWNIAN_MOTION=2,
	RIDGED=3,
	RIDGED_MULTI_FRACTAL=4,
	GRADIENT=5,
	GRADIENT_FBM=6,
	CURL=7,
	CURL_FBM=8
};


torch::Tensor GenerateSimplexNoiseVariation(const std::vector<int32_t> output_shape, const torch::Device GPUdevice,
	const std::vector<float> scale, const std::vector<float> offset,
	const NoiseVariation variation, const float ridgeOffset, const int32_t octaves, const float lacunarity, const float gain);

} //namespace SimplexNoise

#endif //_INCLUDE_NOISE_SETTINGS