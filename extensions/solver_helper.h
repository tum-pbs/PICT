#pragma once

#ifndef _INCLUDE_SOLVER_HELPER
#define _INCLUDE_SOLVER_HELPER

//#include <cuda.h>
//#include <cuda_runtime.h>
#include <cublas_v2.h>

// defined in cg_solver_kernel.cu
template< typename scalar_t>
extern scalar_t ComputeConvergenceCriterion(cublasHandle_t cublasHandle, const scalar_t *r, const index_t n, const ConvergenceCriterion conv);

#endif //_INCLUDE_SOLVER_HELPER