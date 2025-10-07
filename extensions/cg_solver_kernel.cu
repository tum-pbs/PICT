
#include "bicgstab_solver.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "cublas_templates.h"
#include "solver_helper.h"

#include <cmath>


using index_t = int32_t;

#include <iostream>
#include <cmath>

static void CheckErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
			<< err << ") at " << file << ":" << line << std::endl;
  exit(10);
}

//cublas
static void CheckErrorAux(const char* file, unsigned line, const char* statement, cublasStatus_t err) {
  if (err == CUBLAS_STATUS_SUCCESS) return;
  std::cerr << statement << " returned " << "("
			<< err << ") at " << file << ":" << line << std::endl;
  exit(10);
}

//cusparse
static void CheckErrorAux(const char* file, unsigned line, const char* statement, cusparseStatus_t err) {
  if (err == CUSPARSE_STATUS_SUCCESS) return;
  std::cerr << statement << " returned " << "("
			<< err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUDA_CHECK_RETURN(value) CheckErrorAux(__FILE__, __LINE__, #value, value)



template <typename scalar_t>
__global__ void k_csrExtractDiag(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, scalar_t *aDiag, const index_t n){
	for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += blockDim.x * gridDim.x){
		bool foundDiagonal = false;
		const index_t rowEnd = aRow[row+1];
		for(size_t index = aRow[row]; index < rowEnd; index++){
			if(aIndex[index]==row){
				aDiag[row] = aVal[index];
				foundDiagonal = true;
				break;
			}
		}
		if (foundDiagonal==false) aDiag[row]=0;
	}
}

template <typename scalar_t>
scalar_t csrMatrixTrace(cublasHandle_t cublasHandle, const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n){
	scalar_t trace;
	scalar_t* aDiag;
	CUDA_CHECK_RETURN(cudaMalloc(&aDiag, n*sizeof(scalar_t)));
	int minGridSize = 0, blockSize = 0, gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_csrExtractDiag<scalar_t>, 0, 0);
	gridSize = (n + blockSize - 1) / blockSize;
	k_csrExtractDiag<scalar_t><<<gridSize, blockSize>>>(aVal, aIndex, aRow, aDiag, n);
	//print_array<<<1,1,0,0>>>(aDiag, n);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, aDiag, 1, &trace));
	//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasDasum(blasHandle, n, aDiag, 1, &trace));
	cudaFree(aDiag);
	return trace;
}

template< typename scalar_t>
scalar_t ComputeConvergenceCriterion(cublasHandle_t cublasHandle, const scalar_t *r, const index_t n, const ConvergenceCriterion conv){
	
	scalar_t criterion = 0;
	switch(conv){
		case ConvergenceCriterion::NORM2:
			CUDA_CHECK_RETURN(cublasTnrm2<scalar_t>(cublasHandle, n, r, 1, &criterion));
			break;
		case ConvergenceCriterion::NORM2_NORMALIZED:
		{
			scalar_t norm2Normalization = static_cast<scalar_t>(1) / std::sqrt(static_cast<scalar_t>(n));
			CUDA_CHECK_RETURN(cublasTnrm2<scalar_t>(cublasHandle, n, r, 1, &criterion));
			criterion = criterion * norm2Normalization;
			break;
		}
		case ConvergenceCriterion::ABS_SUM:
			CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, r, 1, &criterion));
			break;
		case ConvergenceCriterion::ABS_MEAN:
			CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, r, 1, &criterion));
			criterion = criterion/n;
			break;
		case ConvergenceCriterion::ABS_MAX:
		{
			int idx_max = 0;
			CUDA_CHECK_RETURN(cublasITamax<scalar_t>(cublasHandle, n, r, 1, &idx_max));
			CUDA_CHECK_RETURN(cudaMemcpy(&criterion, r+idx_max, sizeof(scalar_t), cudaMemcpyDeviceToHost));
			criterion = abs(criterion);
			break;
		}
	}
	
	return criterion;
}
template float ComputeConvergenceCriterion<float>(cublasHandle_t cublasHandle, const float *r, const index_t n, const ConvergenceCriterion conv);
template double ComputeConvergenceCriterion<double>(cublasHandle_t cublasHandle, const double *r, const index_t n, const ConvergenceCriterion conv);

template <typename scalar_t>
solverReturn_t cgSolveGPU(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz, const scalar_t *_f, scalar_t *_x, const index_t nBatches,
		const index_t maxit, const scalar_t tol, const ConvergenceCriterion conv, const bool laplaceRankDeficient, const index_t residualResetSteps, const scalar_t *aDiag,
		const bool transposeA, const bool printResidual, const bool returnBestResult){
	
	const bool resetResidual = residualResetSteps>0;
	
	//residual inspection
	scalar_t lowestNorm=0, lowestMean=0, lowestMax=0;
	index_t lowestNormIt=-1,lowestMeanIt=-1, lowestMaxIt=-1;
	bool lowestMaxSet=false;
	
	scalar_t norm2Normalization = static_cast<scalar_t>(1) / std::sqrt(static_cast<scalar_t>(n));
	scalar_t r_norm2 = 0;
	//scalar_t lastRNorm = 0;
	
	
	scalar_t rho, rhop, beta, alpha, temp;
	rho = 0.0;
	rhop = 1.0;
	const scalar_t zero = 0.0;
	const scalar_t one  = 1.0;
	const scalar_t mone = -1.0;
	scalar_t rankDeficientScaling = 1;
	
	cusparseMatDescr_t descr_a= 0;
	size_t bufferSizeA_MV1 = 0;
	size_t bufferSizeA_MV2 = 0;
	void *buffer = nullptr;
	const cudaDataType scalarType = getCudaDataType<scalar_t>();
	cusparseSpMatDescr_t descrSp_a=0;
	cusparseDnVecDescr_t descr_x=0;
	cusparseDnVecDescr_t descr_r=0;
	cusparseDnVecDescr_t descr_rw=0;
	cusparseDnVecDescr_t descr_p=0;
	cusparseSpMVAlg_t mvAlg = CUSPARSE_SPMV_CSR_ALG2; //CUSPARSE_SPMV_CSR_ALG1 is faster but not deterministic
	
	cublasHandle_t cublasHandle  = 0;
	cusparseHandle_t cusparseHandle  = 0;
	
	const cusparseOperation_t transOp = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
	
	CUDA_CHECK_RETURN(cublasCreate(&cublasHandle));
	CUDA_CHECK_RETURN(cusparseCreate(&cusparseHandle));
	
	
	scalar_t *r=nullptr;
	scalar_t *rw=nullptr;
	scalar_t *p=nullptr;
	scalar_t *gpuOne=nullptr;
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&r, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&rw, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&p, sizeof(scalar_t) * n));
	if(laplaceRankDeficient){
		CUDA_CHECK_RETURN(cudaMalloc ((void**)&gpuOne, sizeof(scalar_t)));
		CUDA_CHECK_RETURN(cudaMemcpy(gpuOne, &one, sizeof(scalar_t), cudaMemcpyHostToDevice));
	}
	
	const index_t criterionRisingStepsCutoff = 100;
	scalar_t *best_x;
	if(returnBestResult){
		CUDA_CHECK_RETURN(cudaMalloc ((void**)&best_x, sizeof(scalar_t) * n));
	}
	
	CUDA_CHECK_RETURN(cusparseCreateMatDescr(&descr_a));
	CUDA_CHECK_RETURN(cusparseSetMatType(descr_a,CUSPARSE_MATRIX_TYPE_GENERAL));
	CUDA_CHECK_RETURN(cusparseSetMatIndexBase(descr_a,CUSPARSE_INDEX_BASE_ZERO));
	
	CUDA_CHECK_RETURN(cusparseCreateCsr(&descrSp_a, n, n, nnz, const_cast<index_t*>(aRow), const_cast<index_t*>(aIndex), const_cast<scalar_t*>(aVal),
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO , scalarType));
		
		
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_x , n, _x , scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_r , n, r , scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_rw, n, rw, scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_p, n, p, scalarType));
	
	
	CUDA_CHECK_RETURN(cusparseSpMV_bufferSize(cusparseHandle, transOp, 
		&one, descrSp_a, descr_x, &zero, descr_r, scalarType, mvAlg, &bufferSizeA_MV1));
	CUDA_CHECK_RETURN(cusparseSpMV_bufferSize(cusparseHandle, transOp, 
		&one, descrSp_a, descr_p, &zero, descr_rw, scalarType, mvAlg, &bufferSizeA_MV2));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	const size_t bufferSizeMax = static_cast<size_t>(max(bufferSizeA_MV1, bufferSizeA_MV2));
	CUDA_CHECK_RETURN(cudaMalloc (&buffer, bufferSizeMax));
	
	// free since we loop over batches in _x
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_x));
	
	std::vector<LinearSolverResultInfo> resultInfos;
	resultInfos.reserve(nBatches);
	
	/* Begin solve*/
	//vector loop
	index_t i=0;
	for (index_t batchIdx=0; batchIdx<nBatches; ++batchIdx){
		
		const scalar_t *f = _f + n*batchIdx;
		scalar_t *x = _x + n*batchIdx;
		CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_x , n, x , scalarType));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemset((void *)r, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)rw, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)p, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		
		scalar_t bestCriterion = 0;
		index_t bestCriterionIt = -1;
		scalar_t lastCriterion = 0;
		index_t criterionRisingSteps = 0;

		LinearSolverResultInfo resultInfo = {
			.finalResidual = 0,
			.usedIterations = -1,
			.converged = false,
			.isFiniteResidual = true,
		};
		
		
		//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasScopy(blasHandle, matrix_shape, x_old, 1, x, 1));
		// 1. Initial residual r_0 = b - A*x_0
		CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
			&mone, descrSp_a, descr_x, &zero, descr_r, scalarType, mvAlg, buffer));
		//cusparseSetStream(sparseHandle, stream); cdpErrchk_sparse(cusparseScsrmv(sparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix_shape, matrix_shape, nnz_a,
		//	  &negative_one, descrA, csr_valuesA, csr_row_ptr, csr_col_ind,
		//	  x , &zero, r));
		//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		if(laplaceRankDeficient){  // compute (A + e^Te) * x instead
			if(aDiag==nullptr){
				rankDeficientScaling = csrMatrixTrace<scalar_t>(cublasHandle, aVal, aIndex, aRow, n) / n / n / n;
			}else{
				CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, aDiag, 1, &rankDeficientScaling));
				rankDeficientScaling = rankDeficientScaling / n / n / n;
			}
			
			CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, x, 1, &temp));
			//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSasum(blasHandle, matrix_shape, x, 1, &temp));
			temp *= rankDeficientScaling;
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &temp, gpuOne, 0, r, 1)); //r=temp*1 + r
			//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &temp, gpuOne, 0, r, 1));
		//printf("new lapRankdef with scaling %f \n", rankDeficientScaling);
		}
		CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle, n, &one, f, 1, r, 1)); //r = 1*f + r
		//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &one, rhs, 1, r, 1));
		// 2. set rhat=r and p=r, compute rho = <r,r>
		CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, r, 1, rw, 1));
		CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, r, 1, p, 1)); 
		CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, r, 1, r, 1, &rho));
		
		/* solver loop */
		for (i=0; i<maxit; ++i){
			if(resetResidual && (i+1)%residualResetSteps==0){
				
				// 1. compute initial residual r0=b-Ax0 (using initial guess in x)
				CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
					&mone, descrSp_a, descr_x, &zero, descr_r, scalarType, mvAlg, buffer)); // r=-Ax
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
				if(laplaceRankDeficient){  // compute (A + e^Te) * x instead
					CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, x, 1, &temp));
					//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSasum(blasHandle, matrix_shape, x, 1, &temp));
					temp *= rankDeficientScaling;
					CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &temp, gpuOne, 0, r, 1));
					//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &temp, gpuOne, 0, r, 1));
				}
				
				//CUDA_CHECK_RETURN(cublasTscal<scalar_t>(cublasHandle, n, &mone, r, 1)); //r = -1*r
				CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle, n, &one, f, 1, r, 1)); //r = 1*f + r
				
				// 2. set rhat=r and p=r, compute rho = <r,r>
				CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, r, 1, rw, 1));
				CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, r, 1, p, 1)); 
				CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, r, 1, r, 1, &rho));
			}

			// 3. compute rhat as A*p
			CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
				&one, descrSp_a, descr_p, &zero, descr_rw, scalarType, mvAlg, buffer));

			if(laplaceRankDeficient){  // compute (A + e^Te) * p instead
				CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, p, 1, &temp));
				//cublasSasum(blasHandle, matrix_shape, p, 1, &temp);
				temp *= rankDeficientScaling;
				CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &temp, gpuOne, 0, rw, 1));
				//cublasSaxpy(blasHandle, matrix_shape, &temp, gpuOne, 0, rhat, 1);
			}
			// 4. compute alpha = rho/<p,rhat>
			// printf("step 4\n");
			CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, p, 1, rw, 1, &temp));
			//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSdot(blasHandle, matrix_shape, p, 1, rhat, 1, &temp));
			alpha = rho/temp;
			// 5.  x = x + alpha*p
			// printf("step 5\n");
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &alpha, p, 1, x, 1));
			//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &alpha, p, 1, x, 1));
			// 6. r_new = r - alpha*rhat
			// printf("step 6\n");
			temp = -alpha;
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &temp, rw, 1, r, 1));
			//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &temp, rhat, 1, r, 1));
			// 7. convergence check norm(Ax-b)  =~= r < tol
			// printf("step 7\n");
			
			scalar_t criterion = ComputeConvergenceCriterion(cublasHandle, r, n, conv);

			auto fpclass = std::fpclassify(criterion);
			if(fpclass==FP_INFINITE or fpclass==FP_NAN){
				printf("CG residual is not finite in interation %d. residual=%.03e.\n", i, criterion);

				resultInfo.converged = false;
				resultInfo.usedIterations = i;
				resultInfo.finalResidual = criterion;
				resultInfo.isFiniteResidual = false;
				break;
			}
			
			if(returnBestResult){
				if(i==0 || criterion<bestCriterion){
					bestCriterion = criterion;
					bestCriterionIt = i;
					CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, x, 1, best_x, 1));
				}
				
				if(i>0 && criterion>=lastCriterion){
					// criterion rising
					++criterionRisingSteps;
				} else {
					criterionRisingSteps = 0;
				}
				
				
				lastCriterion = criterion;
			}
			
			if(printResidual){
				CUDA_CHECK_RETURN(cublasTnrm2<scalar_t>(cublasHandle, n, r, 1, &temp));
				//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSnrm2(blasHandle, matrix_shape, r , 1, &temp));
				r_norm2 = temp*norm2Normalization;
				int idx_max=-1;
				scalar_t r_sum, r_mean, r_max;
				CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, r, 1, &r_sum));
				r_mean = r_sum/n;
				CUDA_CHECK_RETURN(cublasITamax<scalar_t>(cublasHandle, n, r, 1, &idx_max));
				//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
				// result of cublasITamax is 1-based indexing
				idx_max -= 1;
				bool valid_max = 0<=idx_max && idx_max<n;
				if(valid_max){
					//idx_max = idx_max<0 ? 0 : (idx_max>=n ? n-1 : idx_max);
					CUDA_CHECK_RETURN(cudaMemcpy(&r_max, r+idx_max, sizeof(scalar_t), cudaMemcpyDeviceToHost));
					//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
					r_max = abs(r_max);
				}
				
				if(i==0 || r_norm2<lowestNorm){
					lowestNorm = r_norm2;
					lowestNormIt = i;
				}
				if(i==0 || r_mean<lowestMean){
					lowestMean = r_mean;
					lowestMeanIt = i;
				}
				if(valid_max && (!lowestMaxSet || r_max<lowestMax)){
					lowestMax = r_max;
					lowestMaxIt = i;
					lowestMaxSet = true;
				}
			}
			
			
			/*if(printResidual){
				printf("CG step %d residual[%d]: norm=%.03e, mean=%.03e, max=%.03e (at %d), \n", i, n, temp, r_sum/n, r_max, idx_max);
			}*/
			/*if(i!=0 && printResidual && lastRNorm<r_norm2){
				printf("CG residual rising in it %d: %.03e -> %.03e.\n", i, lastRNorm, r_norm2);
			}
			lastRNorm = r_norm2;*/
			resultInfo.usedIterations = i;
			resultInfo.finalResidual = criterion;
			if (criterion < tol) {
				resultInfo.converged = true;
				resultInfo.isFiniteResidual = true;
				break;
			}
			
			if(returnBestResult && (i==(maxit-1) || criterionRisingSteps>=criterionRisingStepsCutoff)){ // last iteration done and did not converge
				CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, best_x, 1, x, 1));
				if(criterionRisingSteps>=criterionRisingStepsCutoff){
					printf("CG residual rising for %d iterations, using best result from iteration %d with residual=%.03e.\n", criterionRisingSteps, bestCriterionIt, bestCriterion);
				}else{
					printf("CG did not converge after %d iterations, using best result from iteration %d with residual=%.03e.\n", maxit, bestCriterionIt, bestCriterion);
				}

				resultInfo.converged = false;
				resultInfo.usedIterations = bestCriterionIt;
				resultInfo.finalResidual = bestCriterion;
				resultInfo.isFiniteResidual = true;
				break;
			}
			// 8. rhoprev = rho; rho = <r,r>
			// printf("step 8\n");
			rhop = rho;
			CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, r, 1, r, 1, &rho));
			//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSdot(blasHandle, matrix_shape, r, 1, r, 1, &rho));
			// 9. beta = rho/rhoprev
			// printf("step 9\n");
			beta = rho/rhop;
			// 10. p = r + beta*p
			// printf("step 10\n");
			CUDA_CHECK_RETURN(cublasTscal<scalar_t>(cublasHandle, n, &beta, p, 1)); 
			//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSscal(blasHandle, matrix_shape, &beta, p, 1));
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &one, r, 1, p, 1));
			//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSaxpy(blasHandle, matrix_shape, &one, r, 1, p, 1));
		}
		CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_x));

		resultInfos.push_back(resultInfo);
	}

	if(printResidual){
		// norm grows with the resolution (n), so normalize the norm?
		printf("CG finished after %d iterations with residual norm=%.03e (tol=%.03e) (raw=%.03e). Lowest: norm=%.03e (it %d), mean=%.03e (it %d), max=%.03e (it %d).\n",
				i, r_norm2, tol, temp, lowestNorm, lowestNormIt, lowestMean, lowestMeanIt, lowestMax, lowestMaxIt);
	}
	
	
	if(returnBestResult) CUDA_CHECK_RETURN(cudaFree(best_x));
	
	if (buffer) CUDA_CHECK_RETURN(cudaFree(buffer));
	cusparseDestroyMatDescr(descr_a);
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_r));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_rw));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_p));
	CUDA_CHECK_RETURN(cusparseDestroySpMat(descrSp_a));
	if (r) CUDA_CHECK_RETURN(cudaFree(r));
	if (rw) CUDA_CHECK_RETURN(cudaFree(rw));
	if (p) CUDA_CHECK_RETURN(cudaFree(p));
	if (gpuOne) CUDA_CHECK_RETURN(cudaFree(gpuOne));
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	return resultInfos;
}

template solverReturn_t cgSolveGPU<float>(const float *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz, const float *_f, float *_x, const index_t nBatches,
	const index_t maxit, const float tol, const ConvergenceCriterion conv,  const bool laplaceRankDeficient, const index_t residualResetSteps, const float *aDiag, const bool transposeA,
	const bool printResidual, const bool returnBestResult);
template solverReturn_t cgSolveGPU<double>(const double *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz, const double *_f, double *_x, const index_t nBatches,
	const index_t maxit, const double tol, const ConvergenceCriterion conv,  const bool laplaceRankDeficient, const index_t residualResetSteps, const double *aDiag, const bool transposeA,
	const bool printResidual, const bool returnBestResult);

/*
template <typename scalar_t>
solverReturn_t cgSolvePreconGPU(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz, const scalar_t *_f, scalar_t *_x, const index_t nBatches,
		const index_t maxit, const scalar_t tol, const ConvergenceCriterion conv, const index_t residualResetSteps, const scalar_t *aDiag,
		const bool transposeA, const bool printResidual, const bool returnBestResult){
	
	const bool calculatePreconditioner = true;
	const bool flexiblePrecondition = true; // Polak-Ribiere
	const bool resetResidual = residualResetSteps>0;
	
	//residual inspection
	scalar_t lowestNorm=0, lowestMean=0, lowestMax=0;
	index_t lowestNormIt=-1,lowestMeanIt=-1, lowestMaxIt=-1;
	bool lowestMaxSet=false;
	
	scalar_t norm2Normalization = static_cast<scalar_t>(1) / std::sqrt(static_cast<scalar_t>(n));
	scalar_t r_norm2 = 0;
	//scalar_t lastRNorm = 0;
	
	
	scalar_t rho, rhop, beta, alpha, temp;
	rho = 0.0;
	rhop = 1.0;
	const scalar_t zero = 0.0;
	const scalar_t one  = 1.0;
	const scalar_t mone = -1.0;
	scalar_t rankDeficientScaling = 1;
	
	const cusparseOperation_t transOp = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
	
	const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseOperation_t trans_L  = transOp;
	const cusparseOperation_t trans_U  = transOp;
	cusparseMatDescr_t descr_a= 0;
	cusparseMatDescr_t descr_m= 0;
	cusparseMatDescr_t descr_l= 0;
	cusparseMatDescr_t descr_u= 0;
	csrilu02Info_t info_m  = 0;
	csrsv2Info_t info_l  = 0;
	csrsv2Info_t info_u  = 0;
	int bufferSizeM = 0;
	int bufferSizeL = 0;
	int bufferSizeU = 0;
	size_t bufferSizeA_MV1 = 0;
	size_t bufferSizeA_MV2 = 0;
	void *buffer = nullptr;
	const cudaDataType scalarType = getCudaDataType<scalar_t>();
	cusparseSpMatDescr_t descrSp_a=0;
	cusparseDnVecDescr_t descr_x=0;
	cusparseDnVecDescr_t descr_r=0;
	cusparseDnVecDescr_t descr_rw=0;
	cusparseDnVecDescr_t descr_p=0;
	cusparseSpMVAlg_t mvAlg = CUSPARSE_SPMV_CSR_ALG2; //CUSPARSE_SPMV_CSR_ALG1 is faster but not deterministic
	
	cublasHandle_t cublasHandle  = 0;
	cusparseHandle_t cusparseHandle  = 0;
	
	CUDA_CHECK_RETURN(cublasCreate(&cublasHandle));
	CUDA_CHECK_RETURN(cusparseCreate(&cusparseHandle));
	
	scalar_t *mVal=nullptr;
	const index_t *mIndex=aIndex;
	const index_t *mRow=aRow;
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&mVal, sizeof(scalar_t) * nnz));
	
	scalar_t *r=nullptr;
	scalar_t *rw=nullptr;
	scalar_t *p=nullptr;
	scalar_t *t=nullptr;
	scalar_t *z=nullptr;
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&r, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&rw, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&p, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&t, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&z, sizeof(scalar_t) * n));
	
	const index_t criterionRisingStepsCutoff = 100;
	scalar_t *best_x;
	if(returnBestResult){
		CUDA_CHECK_RETURN(cudaMalloc ((void**)&best_x, sizeof(scalar_t) * n));
	}
	
	CUDA_CHECK_RETURN(cusparseCreateMatDescr(&descr_a));
	CUDA_CHECK_RETURN(cusparseSetMatType(descr_a,CUSPARSE_MATRIX_TYPE_GENERAL));
	CUDA_CHECK_RETURN(cusparseSetMatIndexBase(descr_a,CUSPARSE_INDEX_BASE_ZERO));
	
	CUDA_CHECK_RETURN(cusparseCreateCsr(&descrSp_a, n, n, nnz, const_cast<index_t*>(aRow), const_cast<index_t*>(aIndex), const_cast<scalar_t*>(aVal),
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO , scalarType));
	
	CUDA_CHECK_RETURN(cusparseCreateMatDescr(&descr_m));
	CUDA_CHECK_RETURN(cusparseSetMatType(descr_m,CUSPARSE_MATRIX_TYPE_GENERAL));
	CUDA_CHECK_RETURN(cusparseSetMatIndexBase(descr_m,CUSPARSE_INDEX_BASE_ZERO));
	
	CUDA_CHECK_RETURN(cusparseCreateMatDescr(&descr_l));
	CUDA_CHECK_RETURN(cusparseSetMatType(descr_l,CUSPARSE_MATRIX_TYPE_GENERAL));
	CUDA_CHECK_RETURN(cusparseSetMatIndexBase(descr_l,CUSPARSE_INDEX_BASE_ZERO));
	CUDA_CHECK_RETURN(cusparseSetMatFillMode(descr_l,CUSPARSE_FILL_MODE_LOWER));
	CUDA_CHECK_RETURN(cusparseSetMatDiagType(descr_l,CUSPARSE_DIAG_TYPE_UNIT));
	
	CUDA_CHECK_RETURN(cusparseCreateMatDescr(&descr_u));
	CUDA_CHECK_RETURN(cusparseSetMatType(descr_u,CUSPARSE_MATRIX_TYPE_GENERAL));
	CUDA_CHECK_RETURN(cusparseSetMatIndexBase(descr_u,CUSPARSE_INDEX_BASE_ZERO));
	CUDA_CHECK_RETURN(cusparseSetMatFillMode(descr_u,CUSPARSE_FILL_MODE_UPPER));
	CUDA_CHECK_RETURN(cusparseSetMatDiagType(descr_u,CUSPARSE_DIAG_TYPE_NON_UNIT));
	
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_x , n, _x , scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_r , n, r , scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_rw, n, rw, scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_p, n, p, scalarType));
	
	CUDA_CHECK_RETURN(cusparseCreateCsrilu02Info(&info_m));
	CUDA_CHECK_RETURN(cusparseCreateCsrsv2Info(&info_l));
	CUDA_CHECK_RETURN(cusparseCreateCsrsv2Info(&info_u));
	
	CUDA_CHECK_RETURN(cusparseTcsrilu02_bufferSize<scalar_t>(cusparseHandle, n, nnz,
		descr_m, mVal, aRow, aIndex, info_m, &bufferSizeM));
	CUDA_CHECK_RETURN(cusparseTcsrsv2_bufferSize<scalar_t>(cusparseHandle, trans_L, n, nnz,
		descr_l, mVal, aRow, aIndex, info_l, &bufferSizeL));
	CUDA_CHECK_RETURN(cusparseTcsrsv2_bufferSize<scalar_t>(cusparseHandle, trans_U, n, nnz,
		descr_u, mVal, aRow, aIndex, info_u, &bufferSizeU));
	CUDA_CHECK_RETURN(cusparseSpMV_bufferSize(cusparseHandle, transOp, 
		&one, descrSp_a, descr_x, &zero, descr_r, scalarType, mvAlg, &bufferSizeA_MV1));
	CUDA_CHECK_RETURN(cusparseSpMV_bufferSize(cusparseHandle, transOp, 
		&one, descrSp_a, descr_p, &zero, descr_rw, scalarType, mvAlg, &bufferSizeA_MV2));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	const size_t bufferSizeMax = max(static_cast<size_t>(max(bufferSizeM, max(bufferSizeL, bufferSizeU))), max(bufferSizeA_MV1, bufferSizeA_MV2));
	CUDA_CHECK_RETURN(cudaMalloc (&buffer, bufferSizeMax));
	
	// free since we loop over batches in _x
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_x));
	
	if(calculatePreconditioner){
		CUDA_CHECK_RETURN(cudaMemcpy(mVal, aVal, sizeof(scalar_t) * nnz, cudaMemcpyDeviceToDevice));
		
		CUDA_CHECK_RETURN(cusparseTcsrilu02_analysis<scalar_t>(cusparseHandle, n, nnz, descr_m, mVal, mRow, mIndex, info_m, policy_M, buffer));
		
		int structural_zero=-1;
		auto status = cusparseXcsrilu02_zeroPivot(cusparseHandle, info_m, &structural_zero);
		if(CUSPARSE_STATUS_ZERO_PIVOT == status){
			std::cerr << "A(" << structural_zero << "," << structural_zero << ") is missing" << std::endl;
		}
		// compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) 
		CUDA_CHECK_RETURN(cusparseTcsrilu02<scalar_t>(cusparseHandle,n, nnz, descr_m, mVal, mRow, mIndex, info_m, policy_M, buffer));
		
		status = cusparseXcsrilu02_zeroPivot(cusparseHandle, info_m, &structural_zero);
		if (CUSPARSE_STATUS_ZERO_PIVOT == status){
			std::cerr << "L(" << structural_zero << "," << structural_zero << ") is zero" << std::endl;
		}
	}
	
	// analyse the lower and upper triangular factors 
	CUDA_CHECK_RETURN(cusparseTcsrsv2_analysis<scalar_t>(cusparseHandle,trans_L, n, nnz, descr_l, mVal, mRow, mIndex, info_l, policy_L, buffer));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	//CUDA_CHECK_RETURN(cusparseTcsrsv2_analysis<scalar_t>(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,nnz,descr_u,aVal,aIndex,aRow,info_u));
	CUDA_CHECK_RETURN(cusparseTcsrsv2_analysis<scalar_t>(cusparseHandle,trans_U, n, nnz, descr_u, mVal, mRow, mIndex, info_u, policy_U, buffer));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	std::vector<LinearSolverResultInfo> resultInfos;
	resultInfos.reserve(nBatches);
	
	// Begin solve
	//vector loop
	index_t i=0;
	for (index_t batchIdx=0; batchIdx<nBatches; ++batchIdx){
		
		const scalar_t *f = _f + n*batchIdx;
		scalar_t *x = _x + n*batchIdx;
		CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_x , n, x , scalarType));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemset((void *)r, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)rw, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)p, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)t, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)z, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		
		scalar_t bestCriterion = 0;
		index_t bestCriterionIt = -1;
		scalar_t lastCriterion = 0;
		index_t criterionRisingSteps = 0;

		LinearSolverResultInfo resultInfo = {
			.finalResidual = 0,
			.usedIterations = -1,
			.converged = false,
			.isFiniteResidual = true,
		};
		
		
		//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasScopy(blasHandle, matrix_shape, x_old, 1, x, 1));
		// 1. Initial residual r_0 = b - A*x_0
		CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
			&mone, descrSp_a, descr_x, &zero, descr_r, scalarType, mvAlg, buffer)); // r = -A*x_0
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle, n, &one, f, 1, r, 1)); //r = 1*f + r
		
		// 2. Compute z_0 = L^-T L^-1 r_0
		CUDA_CHECK_RETURN(cusparseTcsrsv2_solve<scalar_t>(cusparseHandle,trans_L, n, nnz,&one,descr_l,mVal,mRow,mIndex,info_l,r,t, policy_L, buffer));
		CUDA_CHECK_RETURN(cusparseTcsrsv2_solve<scalar_t>(cusparseHandle,trans_U, n, nnz,&one,descr_u,mVal,mRow,mIndex,info_u,t,z, policy_U, buffer));
		
		// 2. and p=z, compute rho = <r,z>
		CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, z, 1, p, 1)); 
		CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, r, 1, z, 1, &rho));
		
		// solver loop 
		for (i=0; i<maxit; ++i){
			if(resetResidual && (i+1)%residualResetSteps==0){
				
				// 1. compute initial residual r0=b-Ax0 (using initial guess in x)
				CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
					&mone, descrSp_a, descr_x, &zero, descr_r, scalarType, mvAlg, buffer)); // r=-Ax
				CUDA_CHECK_RETURN(cudaDeviceSynchronize());
				CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle, n, &one, f, 1, r, 1)); //r = 1*f + r
		
				// 2. Compute z_0 = L^-T L^-1 r_0
				CUDA_CHECK_RETURN(cusparseTcsrsv2_solve<scalar_t>(cusparseHandle,trans_L, n, nnz,&one,descr_l,mVal,mRow,mIndex,info_l,r,t, policy_L, buffer));
				CUDA_CHECK_RETURN(cusparseTcsrsv2_solve<scalar_t>(cusparseHandle,trans_U, n, nnz,&one,descr_u,mVal,mRow,mIndex,info_u,t,z, policy_U, buffer));
				
				// 2. and p=z, compute rho = <r,z>
				CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, z, 1, p, 1)); 
				CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, r, 1, z, 1, &rho));
			}

			// 3. compute rhat as A*p
			CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
				&one, descrSp_a, descr_p, &zero, descr_rw, scalarType, mvAlg, buffer));

			// 4. compute alpha = rho/<p,rhat>
			CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, p, 1, rw, 1, &temp));
			alpha = rho/temp;
			
			// 5.  x = x + alpha*p
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &alpha, p, 1, x, 1));
			
			// 6. r_new = r - alpha*rhat
			temp = -alpha;
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &temp, rw, 1, r, 1));
			
			// 7. convergence check norm(Ax-b)  =~= r < tol
			scalar_t criterion = ComputeConvergenceCriterion(cublasHandle, r, n, conv);

			auto fpclass = std::fpclassify(criterion);
			if(fpclass==FP_INFINITE or fpclass==FP_NAN){
				printf("CG residual is not finite in interation %d. residual=%.03e.\n", i, criterion);

				resultInfo.converged = false;
				resultInfo.usedIterations = i;
				resultInfo.finalResidual = criterion;
				resultInfo.isFiniteResidual = false;
				break;
			}
			
			if(returnBestResult){
				if(i==0 || criterion<bestCriterion){
					bestCriterion = criterion;
					bestCriterionIt = i;
					CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, x, 1, best_x, 1));
				}
				
				if(i>0 && criterion>=lastCriterion){
					// criterion rising
					++criterionRisingSteps;
				} else {
					criterionRisingSteps = 0;
				}
				lastCriterion = criterion;
			}
			
			if(printResidual){
				CUDA_CHECK_RETURN(cublasTnrm2<scalar_t>(cublasHandle, n, r, 1, &temp));
				//cublasSetStream(blasHandle, stream); cdpErrchk_blas(cublasSnrm2(blasHandle, matrix_shape, r , 1, &temp));
				r_norm2 = temp*norm2Normalization;
				int idx_max=-1;
				scalar_t r_sum, r_mean, r_max;
				CUDA_CHECK_RETURN(cublasTasum<scalar_t>(cublasHandle, n, r, 1, &r_sum));
				r_mean = r_sum/n;
				CUDA_CHECK_RETURN(cublasITamax<scalar_t>(cublasHandle, n, r, 1, &idx_max));
				//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
				// result of cublasITamax is 1-based indexing
				idx_max -= 1;
				bool valid_max = 0<=idx_max && idx_max<n;
				if(valid_max){
					//idx_max = idx_max<0 ? 0 : (idx_max>=n ? n-1 : idx_max);
					CUDA_CHECK_RETURN(cudaMemcpy(&r_max, r+idx_max, sizeof(scalar_t), cudaMemcpyDeviceToHost));
					//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
					r_max = abs(r_max);
				}
				
				if(i==0 || r_norm2<lowestNorm){
					lowestNorm = r_norm2;
					lowestNormIt = i;
				}
				if(i==0 || r_mean<lowestMean){
					lowestMean = r_mean;
					lowestMeanIt = i;
				}
				if(valid_max && (!lowestMaxSet || r_max<lowestMax)){
					lowestMax = r_max;
					lowestMaxIt = i;
					lowestMaxSet = true;
				}
			}
			
			
			resultInfo.usedIterations = i;
			resultInfo.finalResidual = criterion;
			if (criterion < tol) {
				resultInfo.converged = true;
				resultInfo.isFiniteResidual = true;
				break;
			}
			
			if(returnBestResult && (i==(maxit-1) || criterionRisingSteps>=criterionRisingStepsCutoff)){ // last iteration done and did not converge
				CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, best_x, 1, x, 1));
				if(criterionRisingSteps>=criterionRisingStepsCutoff){
					printf("CG residual rising for %d iterations, using best result from iteration %d with residual=%.03e.\n", criterionRisingSteps, bestCriterionIt, bestCriterion);
				}else{
					printf("CG did not converge after %d iterations, using best result from iteration %d with residual=%.03e.\n", maxit, bestCriterionIt, bestCriterion);
				}

				resultInfo.converged = false;
				resultInfo.usedIterations = bestCriterionIt;
				resultInfo.finalResidual = bestCriterion;
				resultInfo.isFiniteResidual = true;
				break;
			}
			
			if(flexiblePrecondition){
				CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, z, 1, rw, 1)); 
			}
			
			CUDA_CHECK_RETURN(cusparseTcsrsv2_solve<scalar_t>(cusparseHandle,trans_L, n, nnz,&one,descr_l,mVal,mRow,mIndex,info_l,r,t, policy_L, buffer));
			CUDA_CHECK_RETURN(cusparseTcsrsv2_solve<scalar_t>(cusparseHandle,trans_U, n, nnz,&one,descr_u,mVal,mRow,mIndex,info_u,t,z, policy_U, buffer));
			
			// 8. rhoprev = rho; rho = <r,z>
			rhop = rho;
			if(flexiblePrecondition){
				// rho = <r,z-zp>: zp = z - zp, rho = <r,zp>
				CUDA_CHECK_RETURN(cublasTscal<scalar_t>(cublasHandle, n, &mone, rw, 1)); 
				CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle, n, &one, z, 1, rw, 1)); 
				CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, r, 1, rw, 1, &rho));
				beta = rho/rhop;
				// set correct rho for next iteration
				CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, r, 1, z, 1, &rho));
			} else {
				// rho = <r,z>
				CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, r, 1, z, 1, &rho));
				// 9. beta = rho/rhoprev
				beta = rho/rhop;
			}
			
			// 10. p = z + beta*p
			CUDA_CHECK_RETURN(cublasTscal<scalar_t>(cublasHandle, n, &beta, p, 1)); 
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &one, z, 1, p, 1));
		}
		CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_x));
		resultInfos.push_back(resultInfo);
	}

	if(printResidual){
		// norm grows with the resolution (n), so normalize the norm?
		printf("CG finished after %d iterations with residual norm=%.03e (tol=%.03e) (raw=%.03e). Lowest: norm=%.03e (it %d), mean=%.03e (it %d), max=%.03e (it %d).\n",
				i, r_norm2, tol, temp, lowestNorm, lowestNormIt, lowestMean, lowestMeanIt, lowestMax, lowestMaxIt);
	}
	
	
	if(returnBestResult) CUDA_CHECK_RETURN(cudaFree(best_x));
	
	if (buffer) CUDA_CHECK_RETURN(cudaFree(buffer));
	cusparseDestroyMatDescr(descr_a);
	cusparseDestroyMatDescr(descr_m);
	cusparseDestroyMatDescr(descr_l);
	cusparseDestroyMatDescr(descr_u);
	cusparseDestroyCsrilu02Info(info_m);
	cusparseDestroyCsrsv2Info(info_l);
	cusparseDestroyCsrsv2Info(info_u);
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_r));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_rw));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_p));
	CUDA_CHECK_RETURN(cusparseDestroySpMat(descrSp_a));
	if (mVal) CUDA_CHECK_RETURN(cudaFree(mVal));
	if (r) CUDA_CHECK_RETURN(cudaFree(r));
	if (rw) CUDA_CHECK_RETURN(cudaFree(rw));
	if (p) CUDA_CHECK_RETURN(cudaFree(p));
	if (t) CUDA_CHECK_RETURN(cudaFree(t));
	if (z) CUDA_CHECK_RETURN(cudaFree(z));
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	return resultInfos;
}

template solverReturn_t cgSolvePreconGPU<float>(const float *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz, const float *_f, float *_x, const index_t nBatches,
	const index_t maxit, const float tol, const ConvergenceCriterion conv, const index_t residualResetSteps, const float *aDiag, const bool transposeA,
	const bool printResidual, const bool returnBestResult);
template solverReturn_t cgSolvePreconGPU<double>(const double *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz, const double *_f, double *_x, const index_t nBatches,
	const index_t maxit, const double tol, const ConvergenceCriterion conv, const index_t residualResetSteps, const double *aDiag, const bool transposeA,
	const bool printResidual, const bool returnBestResult);
*/

template <typename scalar_t>
void OuterProductToSparseMatrix(const scalar_t *a, const scalar_t *b, scalar_t *outVal, const index_t *outIndex, const index_t *outRow, const index_t n, const index_t nnz){
	
	scalar_t one = 1;
	scalar_t zero = 0;
	size_t bufferSize = 0;
	void *buffer = nullptr;
	const cudaDataType scalarType = getCudaDataType<scalar_t>();
	cusparseDnMatDescr_t  descr_a=0;
	cusparseDnMatDescr_t  descr_b=0;
	cusparseSpMatDescr_t descr_c=0;
	
	//cublasHandle_t cublasHandle  = 0;
	cusparseHandle_t cusparseHandle  = 0;
	//CUDA_CHECK_RETURN(cublasCreate(&cublasHandle));
	CUDA_CHECK_RETURN(cusparseCreate(&cusparseHandle));
	
	CUDA_CHECK_RETURN(cusparseCreateDnMat(&descr_b, n, 1, 1, const_cast<scalar_t*>(b), scalarType, CUSPARSE_ORDER_ROW)); //rows, cols, ld , *data, CUSPARSE_ORDER_ROW/CUSPARSE_ORDER_COL
	CUDA_CHECK_RETURN(cusparseCreateDnMat(&descr_a, n, 1, 1, const_cast<scalar_t*>(a), scalarType, CUSPARSE_ORDER_ROW)); //rows, cols, ld , *data, CUSPARSE_ORDER_ROW/CUSPARSE_ORDER_COL
	
	CUDA_CHECK_RETURN(cusparseCreateCsr(&descr_c, n, n, nnz, const_cast<index_t*>(outRow), const_cast<index_t*>(outIndex), outVal,
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO , scalarType));
	
	CUDA_CHECK_RETURN(cusparseSDDMM_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, &one, descr_a, descr_b,
		&zero, descr_c, scalarType, CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize));
	CUDA_CHECK_RETURN(cudaMalloc (&buffer, bufferSize));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	
	CUDA_CHECK_RETURN(cusparseSDDMM(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, &one, descr_a, descr_b,
		&zero, descr_c, scalarType, CUSPARSE_SDDMM_ALG_DEFAULT, buffer));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	if (buffer) CUDA_CHECK_RETURN(cudaFree(buffer));
	CUDA_CHECK_RETURN(cusparseDestroyDnMat(descr_a));
	CUDA_CHECK_RETURN(cusparseDestroyDnMat(descr_b));
	CUDA_CHECK_RETURN(cusparseDestroySpMat(descr_c));
	cusparseDestroy(cusparseHandle);
	//cublasDestroy(cublasHandle);
}

template void OuterProductToSparseMatrix(const float *a, const float *b, float *outVal, const index_t *outIndex, const index_t *outRow, const index_t n, const index_t nnz);
template void OuterProductToSparseMatrix(const double *a, const double *b, double *outVal, const index_t *outIndex, const index_t *outRow, const index_t n, const index_t nnz);