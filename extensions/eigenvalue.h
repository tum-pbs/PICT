#pragma once

#ifndef _INCLUDE_EIGENVALUE
#define _INCLUDE_EIGENVALUE

#include "custom_types.h"
#include "optional.h"

std::vector<optional<torch::Tensor>> EigenDecomposition(const torch::Tensor &matrices, const bool outputEigenvalues, const bool outputEigenvectors, const bool normalizeEigenvectors);

#endif //_INCLUDE_EIGENVALUE