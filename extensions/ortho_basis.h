#pragma once

#ifndef _INCLUDE_ORTHO_BASIS
#define _INCLUDE_ORTHO_BASIS

#include "custom_types.h"
//#include "optional.h"
#include <torch/extension.h>

torch::Tensor MakeBasisUnique(const torch::Tensor &basisMatrices, const torch::Tensor &sortingVectors, const bool inPlace);

#endif //_INCLUDE_ORTHO_BASIS