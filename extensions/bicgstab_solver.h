#pragma once

#ifndef _INCLUDE_BICGSTAB_SOLVER
#define _INCLUDE_BICGSTAB_SOLVER

#include "custom_types.h"
#include <sstream>
#include <vector>
#include <string>

enum class ConvergenceCriterion : int8_t{
	NORM2 = 0,
	NORM2_NORMALIZED = 1,
	ABS_SUM = 2,
	ABS_MEAN = 3,
	ABS_MAX = 4,
	
};

struct LinearSolverResultInfo{
	double finalResidual;
	index_t usedIterations;
	bool converged;
	bool isFiniteResidual;
};

inline
std::string LinearSolverResultInfoToString(const LinearSolverResultInfo &info){
	std::ostringstream repr;
	repr << "LinearSolverResultInfo(finalResidual=" << info.finalResidual << ", usedIterations=" << info.usedIterations << ", converged=" << info.converged << ", isFiniteResidual=" << info.isFiniteResidual << ")";
	return repr.str();
}

using solverReturn_t = std::vector<LinearSolverResultInfo>; //LinearSolverResultInfo

template <typename scalar_t>
extern solverReturn_t bicgstabSolveGPU(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
	const scalar_t *_f, scalar_t *_x, const index_t nBatches,
	const bool withPreconditioner,
	const index_t maxit, const scalar_t tol, const ConvergenceCriterion conv, const bool transposeA);

template <typename scalar_t>
extern solverReturn_t cgSolveGPU(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz, const scalar_t *_f, scalar_t *_x, const index_t nBatches,
	const index_t maxit, const scalar_t tol, const ConvergenceCriterion conv,  const bool laplace_rank_deficient, const index_t residual_reset_steps, const scalar_t *aDiag, const bool transposeA, const bool printResidual, const bool returnBestResult);

/* needs update for CUDA 12
template <typename scalar_t>
extern solverReturn_t cgSolvePreconGPU(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz, const scalar_t *_f, scalar_t *_x, const index_t nBatches,
	const index_t maxit, const scalar_t tol, const ConvergenceCriterion conv, const index_t residual_reset_steps, const scalar_t *aDiag, const bool transposeA, const bool printResidual, const bool returnBestResult);
*/
template <typename scalar_t>
extern void OuterProductToSparseMatrix(const scalar_t *a, const scalar_t *b, scalar_t *outVal, const index_t *outIndex, const index_t *outRow, const index_t n, const index_t nnz);

#endif //_INCLUDE_BICGSTAB_SOLVER