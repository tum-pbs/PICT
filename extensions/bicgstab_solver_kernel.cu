/*
  https://github.com/tpn/cuda-samples/blob/master/v8.0/7_CUDALibraries/BiCGStab/pbicgstab.cpp
  https://docs.nvidia.com/cuda/incomplete-lu-cholesky/index.html

  https://docs.nvidia.com/cuda/archive/11.7.1/cublas/index.html
*/

#include "bicgstab_solver.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "cublas_templates.h"
#include "solver_helper.h"



using index_t = int32_t;

#include <iostream>

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
solverReturn_t bicgstabSolveGPU(const scalar_t *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
		const scalar_t *_f, scalar_t *_x, const index_t nBatches,
		const bool withPreconditioner,
		const index_t maxit, const scalar_t tol, const ConvergenceCriterion conv, const bool transposeA){
	
	// const index_t maxit = 100;
	// const scalar_t tol = 1e-4;
	
	scalar_t rho, rhop, beta, alpha, negalpha, omega, negomega, temp, temp2;
	scalar_t nrmr, nrmr0;
	rho = 0.0;
	const scalar_t zero = 0.0;
	const scalar_t one  = 1.0;
	const scalar_t mone = -1.0;
	
	cublasHandle_t cublasHandle  = 0;
	cusparseHandle_t cusparseHandle  = 0;
	
	const cusparseOperation_t transOp = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
	
	const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	//const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
	//const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
	const cusparseSpSVAlg_t spSValg = CUSPARSE_SPSV_ALG_DEFAULT;
	const cusparseOperation_t trans_L  = transOp;
	cusparseFillMode_t fillmode_L = CUSPARSE_FILL_MODE_LOWER;
	cusparseDiagType_t diagtype_L = CUSPARSE_DIAG_TYPE_UNIT;
	const cusparseOperation_t trans_U  = transOp;
	cusparseFillMode_t fillmode_U = CUSPARSE_FILL_MODE_UPPER;
	cusparseDiagType_t diagtype_U = CUSPARSE_DIAG_TYPE_NON_UNIT;
	cusparseMatDescr_t descr_a= 0;
	cusparseMatDescr_t descr_m= 0;
	//cusparseMatDescr_t descr_l= 0;
	//cusparseMatDescr_t descr_u= 0;
	csrilu02Info_t info_m  = 0;
	cusparseSpSVDescr_t info_l  = 0;
	cusparseSpSVDescr_t info_u  = 0;
	int bufferSizeM = 0;
	size_t bufferSizeL = 0;
	size_t bufferSizeU = 0;
	size_t bufferSizeA_MV1 = 0;
	size_t bufferSizeA_MV2 = 0;
	void *buffer = nullptr;
	//cusparseSolveAnalysisInfo_t info_l = 0;
	//cusparseSolveAnalysisInfo_t info_u = 0;
	const cudaDataType scalarType = getCudaDataType<scalar_t>();
	cusparseSpMatDescr_t descrSp_a=0;
	cusparseSpMatDescr_t descrSp_l=0;
	cusparseSpMatDescr_t descrSp_u=0;
	cusparseDnVecDescr_t descr_x=0;
	cusparseDnVecDescr_t descr_r=0;
	cusparseDnVecDescr_t descr_p=0;
	cusparseDnVecDescr_t descr_pw=0;
	cusparseDnVecDescr_t descr_v=0;
	cusparseDnVecDescr_t descr_s=0;
	cusparseDnVecDescr_t descr_t=0;
	cusparseSpMVAlg_t mvAlg = CUSPARSE_SPMV_CSR_ALG2; //CUSPARSE_SPMV_CSR_ALG1 is faster but not deterministic
	
	CUDA_CHECK_RETURN(cublasCreate(&cublasHandle));
	CUDA_CHECK_RETURN(cusparseCreate(&cusparseHandle));
	
	
	scalar_t *mVal=nullptr;
	const index_t *mIndex=aIndex;
	const index_t *mRow=aRow;
	if(withPreconditioner){
		CUDA_CHECK_RETURN(cudaMalloc ((void**)&mVal, sizeof(scalar_t) * nnz));
		CUDA_CHECK_RETURN(cudaMemcpy(mVal, aVal, sizeof(scalar_t) * nnz, cudaMemcpyDeviceToDevice));
	}
	
	scalar_t *r=nullptr;
	scalar_t *rw=nullptr;
	scalar_t *p=nullptr;
	scalar_t *pw=nullptr;
	scalar_t *s=nullptr;
	scalar_t *t=nullptr;
	scalar_t *v=nullptr;
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&r, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&rw, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&p, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&pw, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&s, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&t, sizeof(scalar_t) * n));
	CUDA_CHECK_RETURN(cudaMalloc ((void**)&v, sizeof(scalar_t) * n));
	
	CUDA_CHECK_RETURN(cusparseCreateMatDescr(&descr_a));
	CUDA_CHECK_RETURN(cusparseSetMatType(descr_a,CUSPARSE_MATRIX_TYPE_GENERAL));
	CUDA_CHECK_RETURN(cusparseSetMatIndexBase(descr_a,CUSPARSE_INDEX_BASE_ZERO));
	
	// https://docs.nvidia.com/cuda/archive/11.7.1/cusparse/index.html#cusparse-generic-spmat-create-csr
	// "NOTE: it is safe to cast away constness (const_cast) for input pointers if the descriptor will not be used as an output parameter of a routine (e.g. conversion functions)."
	CUDA_CHECK_RETURN(cusparseCreateCsr(&descrSp_a, n, n, nnz, const_cast<index_t*>(aRow), const_cast<index_t*>(aIndex), const_cast<scalar_t*>(aVal),
		CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO , scalarType));
		
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_x , n, _x , scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_r , n, r , scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_p, n, p, scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_pw, n, pw, scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_v , n, v , scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_s , n, s , scalarType));
	CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_t , n, t , scalarType));
	
	if(withPreconditioner){
		CUDA_CHECK_RETURN(cusparseCreateMatDescr(&descr_m));
		CUDA_CHECK_RETURN(cusparseSetMatType(descr_m,CUSPARSE_MATRIX_TYPE_GENERAL));
		CUDA_CHECK_RETURN(cusparseSetMatIndexBase(descr_m,CUSPARSE_INDEX_BASE_ZERO));
		
		CUDA_CHECK_RETURN(cusparseCreateCsr(&descrSp_l, n, n, nnz, const_cast<index_t*>(mRow), const_cast<index_t*>(mIndex), const_cast<scalar_t*>(mVal),
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO , scalarType));
		CUDA_CHECK_RETURN(cusparseSpMatSetAttribute(descrSp_l, CUSPARSE_SPMAT_FILL_MODE, &fillmode_L, sizeof(fillmode_L)));
		CUDA_CHECK_RETURN(cusparseSpMatSetAttribute(descrSp_l, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype_L, sizeof(diagtype_L)));
		
		CUDA_CHECK_RETURN(cusparseCreateCsr(&descrSp_u, n, n, nnz, const_cast<index_t*>(mRow), const_cast<index_t*>(mIndex), const_cast<scalar_t*>(mVal),
			CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO , scalarType));
		CUDA_CHECK_RETURN(cusparseSpMatSetAttribute(descrSp_l, CUSPARSE_SPMAT_FILL_MODE, &fillmode_U, sizeof(fillmode_U)));
		CUDA_CHECK_RETURN(cusparseSpMatSetAttribute(descrSp_l, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype_U, sizeof(diagtype_U)));
		
		CUDA_CHECK_RETURN(cusparseCreateCsrilu02Info(&info_m));
		CUDA_CHECK_RETURN(cusparseSpSV_createDescr(&info_l));
		CUDA_CHECK_RETURN(cusparseSpSV_createDescr(&info_u));
		// CUDA_CHECK_RETURN(cusparseCreateSolveAnalysisInfo(&info_l));
		// CUDA_CHECK_RETURN(cusparseCreateSolveAnalysisInfo(&info_u));
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	// query how much memory used in csrilu02 and csrsv2, and allocate the buffer
	if(withPreconditioner){
		CUDA_CHECK_RETURN(cusparseTcsrilu02_bufferSize<scalar_t>(cusparseHandle, n, nnz,
			descr_m, mVal, aRow, aIndex, info_m, &bufferSizeM));
		CUDA_CHECK_RETURN(cusparseSpSV_bufferSize(cusparseHandle, trans_L, &one,
			descrSp_l, descr_p, descr_t, scalarType, spSValg, info_l, &bufferSizeL));
		CUDA_CHECK_RETURN(cusparseSpSV_bufferSize(cusparseHandle, trans_U, &one,
			descrSp_u, descr_t, descr_pw, scalarType, spSValg, info_u, &bufferSizeU));
	}
	CUDA_CHECK_RETURN(cusparseSpMV_bufferSize(cusparseHandle, transOp, 
		&one, descrSp_a, descr_x, &zero, descr_r, scalarType, mvAlg, &bufferSizeA_MV1));
	CUDA_CHECK_RETURN(cusparseSpMV_bufferSize(cusparseHandle, transOp, 
		&one, descrSp_a, descr_pw, &zero, descr_v, scalarType, mvAlg, &bufferSizeA_MV2));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		
	const size_t bufferSizeMax = max(max(static_cast<size_t>(bufferSizeM), max(bufferSizeL, bufferSizeU)), max(bufferSizeA_MV1, bufferSizeA_MV2));
	CUDA_CHECK_RETURN(cudaMalloc (&buffer, bufferSizeMax));
	
	if(withPreconditioner){
		CUDA_CHECK_RETURN(cusparseTcsrilu02_analysis<scalar_t>(cusparseHandle, n, nnz, descr_m, mVal, mRow, mIndex, info_m, policy_M, buffer));
		int structural_zero=-1;
		auto status = cusparseXcsrilu02_zeroPivot(cusparseHandle, info_m, &structural_zero);
		if(CUSPARSE_STATUS_ZERO_PIVOT == status){
			std::cerr << "A(" << structural_zero << "," << structural_zero << ") is missing" << std::endl;
		}

		/* analyse the lower and upper triangular factors */
		CUDA_CHECK_RETURN(cusparseSpSV_analysis(cusparseHandle,trans_L, &one, descrSp_l, descr_p, descr_t, scalarType, spSValg, info_l, buffer));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		//CUDA_CHECK_RETURN(cusparseTcsrsv2_analysis<scalar_t>(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,n,nnz,descr_u,aVal,aIndex,aRow,info_u));
		CUDA_CHECK_RETURN(cusparseSpSV_analysis(cusparseHandle,trans_U, &one, descrSp_u, descr_t, descr_pw, scalarType, spSValg, info_u, buffer));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());

		/* compute the lower and upper triangular factors using CUSPARSE csrilu0 routine (on the GPU) */
		CUDA_CHECK_RETURN(cusparseTcsrilu02<scalar_t>(cusparseHandle,n, nnz, descr_m, mVal, mRow, mIndex, info_m, policy_M, buffer));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}
	
	// free since we loop over batches in _x
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_x));

	std::vector<LinearSolverResultInfo> resultInfos;
	resultInfos.reserve(nBatches);
	
	/* Begin solve*/
	//vector loop
	index_t i=0;
	for (index_t batchIdx=0; batchIdx<nBatches; ++batchIdx){

		LinearSolverResultInfo resultInfo = {
			.finalResidual = 0,
			.usedIterations = -1,
			.converged = false,
			.isFiniteResidual = true,
		};
		
		const scalar_t *f = _f + n*batchIdx;
		scalar_t *x = _x + n*batchIdx;
		CUDA_CHECK_RETURN(cusparseCreateDnVec(&descr_x , n, x , scalarType));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		CUDA_CHECK_RETURN(cudaMemset((void *)r, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)rw, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)p, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)pw, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)s, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)t, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaMemset((void *)v, 0, sizeof(scalar_t) * n));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		
		//compute initial residual r0=b-Ax0 (using initial guess in x)
		//CUDA_CHECK_RETURN(cusparseTcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descr_a, aVal,aIndex,aRow, x, &zero, r));
		// r=Ax
		CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
			&one, descrSp_a, descr_x, &zero, descr_r, scalarType, mvAlg, buffer));
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		
		CUDA_CHECK_RETURN(cublasTscal<scalar_t>(cublasHandle, n, &mone, r, 1)); //r = -1*r
		CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle, n, &one, f, 1, r, 1)); //r = 1*f + r
		//copy residual r into r^{\hat} and p
		CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, r, 1, rw, 1));
		CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, r, 1, p, 1)); 
		//CUDA_CHECK_RETURN(cublasTnrm2<scalar_t>(cublasHandle, n, r, 1, &nrmr0));
		nrmr0 = ComputeConvergenceCriterion(cublasHandle, r, n, conv);
		
		/* solver loop */
		if(!(nrmr0 < tol)){
		for (i=0; i<maxit; ++i){
			rhop = rho;
			CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle, n, rw, 1, r, 1, &rho));

			if (i > 0){
				beta= (rho/rhop) * (alpha/omega);
				negomega = -omega;
				CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &negomega, v, 1, p, 1));
				CUDA_CHECK_RETURN(cublasTscal<scalar_t>(cublasHandle,n, &beta, p, 1));
				CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &one, r, 1, p, 1));
			}
			
			if(withPreconditioner){
				//preconditioning step (lower and upper triangular solve)
				CUDA_CHECK_RETURN(cusparseSpSV_solve(cusparseHandle, trans_L, &one, descrSp_l, descr_p, descr_t, scalarType, spSValg, info_l)); // l,p,t
				
				CUDA_CHECK_RETURN(cusparseSpSV_solve(cusparseHandle, trans_U, &one, descrSp_u, descr_t, descr_pw, scalarType, spSValg, info_u)); // u,t,pw
			} else {
				CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, p, 1, pw, 1)); 
			}
			
			//matrix-vector multiplication
			//CUDA_CHECK_RETURN(cusparseTcsrmv<scalar_t>(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descr_a, aVal,aIndex,aRow, pw, &zero, v));
			CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
				&one, descrSp_a, descr_pw, &zero, descr_v, scalarType, mvAlg, buffer));

			CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle,n, rw, 1, v, 1,&temp));
			alpha= rho / temp;
			negalpha = -(alpha);
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &negalpha, v, 1, r, 1));
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &alpha, pw, 1, x, 1));
			//CUDA_CHECK_RETURN(cublasTnrm2<scalar_t>(cublasHandle, n, r, 1, &nrmr));
			nrmr = ComputeConvergenceCriterion(cublasHandle, r, n, conv);
			
			auto fpclass = std::fpclassify(nrmr);
			if(fpclass==FP_INFINITE or fpclass==FP_NAN){
				printf("BiCG residual is not finite in interation %d. residual=%.03e.\n", i, nrmr);

				resultInfo.converged = false;
				resultInfo.usedIterations = i;
				resultInfo.finalResidual = nrmr;
				resultInfo.isFiniteResidual = false;
				break;
			}

			//if (nrmr <= tol*nrmr0){
			resultInfo.usedIterations = i;
			resultInfo.finalResidual = nrmr;
			if (nrmr < tol){
				resultInfo.converged = true;
				resultInfo.isFiniteResidual = true;
				//j=5;
				break;
			}

			if(withPreconditioner){
				//preconditioning step (lower and upper triangular solve)
				CUDA_CHECK_RETURN(cusparseSpSV_solve(cusparseHandle,trans_L, &one, descrSp_l, descr_r, descr_t, scalarType, spSValg, info_l)); // l,r,t

				CUDA_CHECK_RETURN(cusparseSpSV_solve(cusparseHandle,trans_U, &one, descrSp_u, descr_t, descr_s, scalarType, spSValg, info_u)); // u,t,s
			} else {
				CUDA_CHECK_RETURN(cublasTcopy<scalar_t>(cublasHandle, n, r, 1, s, 1));
			}
			
			//matrix-vector multiplication
			//CUDA_CHECK_RETURN(cusparseTcsrmv<scalar_t>(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, nnz, &one, descr_a, aVal,aIndex,aRow, s, &zero, t));
			CUDA_CHECK_RETURN(cusparseSpMV(cusparseHandle, transOp, 
				&one, descrSp_a, descr_s, &zero, descr_t, scalarType, mvAlg, buffer));

			CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle,n, t, 1, r, 1,&temp));
			CUDA_CHECK_RETURN(cublasTdot<scalar_t>(cublasHandle,n, t, 1, t, 1,&temp2));
			omega= temp / temp2;
			negomega = -(omega);
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &omega, s, 1, x, 1));
			CUDA_CHECK_RETURN(cublasTaxpy<scalar_t>(cublasHandle,n, &negomega, t, 1, r, 1));

			//CUDA_CHECK_RETURN(cublasTnrm2<scalar_t>(cublasHandle,n, r, 1,&nrmr));
			nrmr = ComputeConvergenceCriterion(cublasHandle, r, n, conv);

			//if (nrmr <= tol*nrmr0){
			resultInfo.finalResidual = nrmr;
			if (nrmr < tol){
				i++;
				resultInfo.converged = true;
				resultInfo.usedIterations = i;
				resultInfo.isFiniteResidual = true;
				//j=0;
				break;
			}
			
		}}
		else { // nrmr0<tol
			resultInfo.converged = true;
			resultInfo.usedIterations = -1;
			resultInfo.finalResidual = nrmr0;
			resultInfo.isFiniteResidual = true;
		}
		
		CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_x));
		resultInfos.push_back(resultInfo);
	}

	/* destroy the analysis info (for lower and upper triangular factors) */
	// CUDA_CHECK_RETURN(cusparseDestroySolveAnalysisInfo(info_l));
	// CUDA_CHECK_RETURN(cusparseDestroySolveAnalysisInfo(info_u));
	/* free memory */
	if (buffer) CUDA_CHECK_RETURN(cudaFree(buffer));
	cusparseDestroyMatDescr(descr_a);
	if(withPreconditioner){
		cusparseDestroyMatDescr(descr_m);
		cusparseDestroyCsrilu02Info(info_m);
		CUDA_CHECK_RETURN(cusparseDestroySpMat(descrSp_l));
		CUDA_CHECK_RETURN(cusparseDestroySpMat(descrSp_u));
		CUDA_CHECK_RETURN(cusparseSpSV_destroyDescr(info_l));
		CUDA_CHECK_RETURN(cusparseSpSV_destroyDescr(info_u));
	}
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_r));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_p));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_pw));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_v));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_s));
	CUDA_CHECK_RETURN(cusparseDestroyDnVec(descr_t));
	CUDA_CHECK_RETURN(cusparseDestroySpMat(descrSp_a));
	if (mVal) CUDA_CHECK_RETURN(cudaFree(mVal));
	if (r) CUDA_CHECK_RETURN(cudaFree(r));
	if (rw) CUDA_CHECK_RETURN(cudaFree(rw));
	if (p) CUDA_CHECK_RETURN(cudaFree(p));
	if (pw) CUDA_CHECK_RETURN(cudaFree(pw));
	if (s) CUDA_CHECK_RETURN(cudaFree(s));
	if (t) CUDA_CHECK_RETURN(cudaFree(t));
	if (v) CUDA_CHECK_RETURN(cudaFree(v));
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);

	return resultInfos;
}

template solverReturn_t bicgstabSolveGPU<float>(const float *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
	const float *_f, float *_x, const index_t nBatches,
	const bool withPreconditioner,
	const index_t maxit, const float tol, const ConvergenceCriterion conv, const bool transposeA);
template solverReturn_t bicgstabSolveGPU<double>(const double *aVal, const index_t *aIndex, const index_t *aRow, const index_t n, const index_t nnz,
	const double *_f, double *_x, const index_t nBatches,
	const bool withPreconditioner,
	const index_t maxit, const double tol, const ConvergenceCriterion conv, const bool transposeA);