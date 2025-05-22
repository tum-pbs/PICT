#pragma once

#ifndef _INCLUDE_GRIDGEN
#define _INCLUDE_GRIDGEN

#include "custom_types.h"
#include <torch/extension.h>

torch::Tensor MakeGrid2DNonUniformScale(const index_t sizeX, const index_t sizeY, const torch::Tensor &scaleStrength);
torch::Tensor MakeGridNDNonUniformScaleNormalized(const index_t sizeX, const index_t sizeY, const index_t sizeZ, const torch::Tensor &scaleStrength);
torch::Tensor MakeGridNDExpScaleNormalized(const index_t sizeX, const index_t sizeY, const index_t sizeZ, const torch::Tensor &scaleStrength);
torch::Tensor MakeCoordsNDNonUniformScaleNormalized(const index_t sizeX, const index_t sizeY, const index_t sizeZ, const torch::Tensor &scaleStrength);
torch::Tensor CoordsToTransforms(const torch::Tensor &coords);
torch::Tensor CoordsToFaceTransforms(const torch::Tensor &coords);

#endif //_INCLUDE_GRIDGEN