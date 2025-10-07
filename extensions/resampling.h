#pragma once

#ifndef _INCLUDE_RESAMPLING
#define _INCLUDE_RESAMPLING

#include "custom_types.h"
#include <torch/extension.h>

enum class BoundarySampling : int8_t{
	CONSTANT=0,
	CLAMP=1
	
};

torch::Tensor SampleTransformedGridGlobalToLocal(const torch::Tensor &globalData, const torch::Tensor &globalTransform, const torch::Tensor &localCoords,const BoundarySampling boundaryMode, const torch::Tensor &constantValue);
std::vector<torch::Tensor> SampleTransformedGridLocalToGlobal(const torch::Tensor &localData, const torch::Tensor &localCoords, const torch::Tensor &globalTransform, const torch::Tensor &globalShape, const index_t fillMaxSteps);
std::vector<torch::Tensor> SampleTransformedGridLocalToGlobalMulti(const std::vector<torch::Tensor> &localData, const std::vector<torch::Tensor> &localCoords, const torch::Tensor &globalTransform, const torch::Tensor &globalShape, const index_t fillMaxSteps);
torch::Tensor WorldPosFromGridPos(const torch::Tensor& vertex_grid, const torch::Tensor &grid_pos);
#endif //_INCLUDE_RESAMPLING