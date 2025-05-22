#pragma once

#ifndef _INCLUDE_CUBLAS_TEMPLATES
#define _INCLUDE_CUBLAS_TEMPLATES

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
//#include "cusparse.h"


template <typename scalar_t>
cublasStatus_t inline cublasTscal(cublasHandle_t handle, int n,
		const scalar_t           *alpha,
		scalar_t           *x, int incx);
template <>
cublasStatus_t inline cublasTscal<float>(cublasHandle_t handle, int n,
		const float           *alpha,
		float           *x, int incx){
	return cublasSscal(handle, n, alpha, x, incx);
}
template <>
cublasStatus_t inline cublasTscal<double>(cublasHandle_t handle, int n,
		const double           *alpha,
		double           *x, int incx){
	return cublasDscal(handle, n, alpha, x, incx);
}

template <typename scalar_t>
cublasStatus_t inline cublasITamax(cublasHandle_t handle, int n,
        const scalar_t           *x, int incx, int  *result);
template <>
cublasStatus_t inline cublasITamax<float>(cublasHandle_t handle, int n,
        const float           *x, int incx, int  *result){
	return cublasIsamax(handle, n, x, incx, result);
}
template <>
cublasStatus_t inline cublasITamax<double>(cublasHandle_t handle, int n,
        const double           *x, int incx, int  *result){
	return cublasIdamax(handle, n, x, incx, result);
}

template <typename scalar_t>
cublasStatus_t inline cublasTasum(cublasHandle_t handle, int n,
        const scalar_t           *x, int incx, scalar_t  *result);
template <>
cublasStatus_t inline cublasTasum<float>(cublasHandle_t handle, int n,
        const float           *x, int incx, float  *result){
	return cublasSasum(handle, n, x, incx, result);
}
template <>
cublasStatus_t inline cublasTasum<double>(cublasHandle_t handle, int n,
        const double           *x, int incx, double  *result){
	return cublasDasum(handle, n, x, incx, result);
}

template <typename scalar_t>
cublasStatus_t inline cublasTaxpy(cublasHandle_t handle, int n,
		const scalar_t           *alpha,
		const scalar_t           *x, int incx,
		scalar_t                 *y, int incy);
template <>
cublasStatus_t inline cublasTaxpy<float>(cublasHandle_t handle, int n,
		const float           *alpha,
		const float           *x, int incx,
		float                 *y, int incy){
	return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
template <>
cublasStatus_t inline cublasTaxpy<double>(cublasHandle_t handle, int n,
		const double           *alpha,
		const double           *x, int incx,
		double                 *y, int incy){
	return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

template <typename scalar_t>
cublasStatus_t inline cublasTdot (cublasHandle_t handle, int n,
		const scalar_t           *x, int incx,
		const scalar_t           *y, int incy,
		scalar_t           *result);
template <>
cublasStatus_t inline cublasTdot<float>(cublasHandle_t handle, int n,
		const float           *x, int incx,
		const float           *y, int incy,
		float           *result){
	return cublasSdot(handle, n, x, incx, y, incy, result);
}
template <>
cublasStatus_t inline cublasTdot<double>(cublasHandle_t handle, int n,
		const double           *x, int incx,
		const double           *y, int incy,
		double           *result){
	return cublasDdot(handle, n, x, incx, y, incy, result);
}

template <typename scalar_t>
cublasStatus_t inline cublasTnrm2(cublasHandle_t handle, int n,
		const scalar_t *x, int incx, scalar_t *result);
template <>
cublasStatus_t inline cublasTnrm2<float>(cublasHandle_t handle, int n,
		const float *x, int incx, float *result){
	return cublasSnrm2(handle, n, x, incx, result);
}
template <>
cublasStatus_t inline cublasTnrm2<double>(cublasHandle_t handle, int n,
		const double *x, int incx, double *result){
	return cublasDnrm2(handle, n, x, incx, result);
}


template <typename scalar_t>
cublasStatus_t inline cublasTcopy(cublasHandle_t handle, int n,
		const scalar_t           *x, int incx,
		scalar_t                 *y, int incy);
template <>
cublasStatus_t inline cublasTcopy<float>(cublasHandle_t handle, int n,
		const float           *x, int incx,
		float                 *y, int incy){
	return cublasScopy(handle, n, x, incx, y, incy);
}
template <>
cublasStatus_t inline cublasTcopy<double>(cublasHandle_t handle, int n,
		const double           *x, int incx,
		double                 *y, int incy){
	return cublasDcopy(handle, n, x, incx, y, incy);
}

/* --- cusparse --- */

// types
//https://docs.nvidia.com/cuda/archive/11.7.1/cusparse/index.html#cusparse-generic-type-ref
template <typename scalar_t>
cudaDataType constexpr getCudaDataType();
template <>
cudaDataType constexpr getCudaDataType<float>() {return CUDA_R_32F;}
template <>
cudaDataType constexpr getCudaDataType<double>() {return CUDA_R_64F;}

// structure descriptors

// ILU
// https://docs.nvidia.com/cuda/archive/11.7.1/cusparse/index.html#csrilu02_solve

template <typename scalar_t>
cusparseStatus_t inline 
cusparseTcsrilu02_bufferSize(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             scalar_t*                   csrValA,
                             const int*               csrRowPtrA,
                             const int*               csrColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes);
template <>
cusparseStatus_t inline 
cusparseTcsrilu02_bufferSize<float>(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             float*                   csrValA,
                             const int*               csrRowPtrA,
                             const int*               csrColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes){
	return cusparseScsrilu02_bufferSize(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes);
}
template <>
cusparseStatus_t inline 
cusparseTcsrilu02_bufferSize<double>(cusparseHandle_t         handle,
                             int                      m,
                             int                      nnz,
                             const cusparseMatDescr_t descrA,
                             double*                   csrValA,
                             const int*               csrRowPtrA,
                             const int*               csrColIndA,
                             csrilu02Info_t           info,
                             int*                     pBufferSizeInBytes){
	return cusparseDcsrilu02_bufferSize(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, pBufferSizeInBytes);
}

template <typename scalar_t>
cusparseStatus_t inline
cusparseTcsrilu02_analysis(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const scalar_t*            csrValA,
                           const int*               csrRowPtrA,
                           const int*               csrColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer);
template <>
cusparseStatus_t inline
cusparseTcsrilu02_analysis<float>(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const float*            csrValA,
                           const int*               csrRowPtrA,
                           const int*               csrColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer){
	return cusparseScsrilu02_analysis(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer);
}
template <>
cusparseStatus_t inline
cusparseTcsrilu02_analysis<double>(cusparseHandle_t         handle,
                           int                      m,
                           int                      nnz,
                           const cusparseMatDescr_t descrA,
                           const double*            csrValA,
                           const int*               csrRowPtrA,
                           const int*               csrColIndA,
                           csrilu02Info_t           info,
                           cusparseSolvePolicy_t    policy,
                           void*                    pBuffer){
	return cusparseDcsrilu02_analysis(handle, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, info, policy, pBuffer);
}

template <typename scalar_t>
cusparseStatus_t inline
cusparseTcsrilu02(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  scalar_t*                  csrValA_valM,
                  const int*               csrRowPtrA,
                  const int*               csrColIndA,
                  csrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer);
template <>
cusparseStatus_t inline
cusparseTcsrilu02<float>(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  float*                  csrValA_valM,
                  const int*               csrRowPtrA,
                  const int*               csrColIndA,
                  csrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer){
	return cusparseScsrilu02(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
}
template <>
cusparseStatus_t inline
cusparseTcsrilu02<double>(cusparseHandle_t         handle,
                  int                      m,
                  int                      nnz,
                  const cusparseMatDescr_t descrA,
                  double*                  csrValA_valM,
                  const int*               csrRowPtrA,
                  const int*               csrColIndA,
                  csrilu02Info_t           info,
                  cusparseSolvePolicy_t    policy,
                  void*                    pBuffer){
	return cusparseDcsrilu02(handle, m, nnz, descrA, csrValA_valM, csrRowPtrA, csrColIndA, info, policy, pBuffer);
}


#endif //_INCLUDE_CUBLAS_TEMPLATES