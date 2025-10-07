#include "matrix_vector_ops.h"
#include "dispatch.h"

#include <cuda.h>
#include <cuda_runtime.h>


template<typename scalar_t, int DIMS>
__global__
void k_mul_matrix_matrix_GRAD(
		const GridInfo grid,
		const scalar_t* __restrict__ matricesA, const bool transposeA, const bool invertA,
		const scalar_t* __restrict__ matricesB, const bool transposeB, const bool invertB,
		const scalar_t* __restrict__ matricesOut_grad, const bool transposeOutput, const bool invertOutput,
		scalar_t* __restrict__ matricesA_grad,
		scalar_t* __restrict__ matricesB_grad
	){
	
	if(invertOutput || (invertA && invertB)) return;
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		MatrixSquare<scalar_t, DIMS> mOut_grad = transposeOutput ? 
			loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesOut_grad, flatIdx, totalSize) :
			loadMatrixFromChannels<scalar_t, DIMS>(matricesOut_grad, flatIdx, totalSize);
		
		// d mOut / d mA
		if(!invertA){
			MatrixSquare<scalar_t, DIMS> mB = transposeB ? 
				loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesB, flatIdx, totalSize) :
				loadMatrixFromChannels<scalar_t, DIMS>(matricesB, flatIdx, totalSize);
			if(invertB) mB = inverse(mB);
			
			MatrixSquare<scalar_t, DIMS> mA_grad = {0};
			for(index_t row=0; row<DIMS; ++row){
				for(index_t col=0; col<DIMS; ++col){
					mA_grad.a[row][col] = dot(mB.v[col], mOut_grad.v[row]);
				}
			}
			
			if(transposeA) {
				writeMatrixToChannelsTransposed<scalar_t, DIMS>(mA_grad, matricesA_grad, flatIdx, totalSize);
			} else {
				writeMatrixToChannels<scalar_t, DIMS>(mA_grad, matricesA_grad, flatIdx, totalSize);
			}
		}
		
		// d mOut / d mB
		if(!invertB){
			MatrixSquare<scalar_t, DIMS> mA = transposeA ? 
				loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesA, flatIdx, totalSize) :
				loadMatrixFromChannels<scalar_t, DIMS>(matricesA, flatIdx, totalSize);
			if(invertA) mA = inverse(mA);
			
			mA = transposed(mA); // to use colum vectors in dot()
			mOut_grad = transposed(mOut_grad); // to use colum vectors in dot()
			
			MatrixSquare<scalar_t, DIMS> mB_grad = {0};
			for(index_t row=0; row<DIMS; ++row){
				for(index_t col=0; col<DIMS; ++col){
					mB_grad.a[row][col] = dot(mA.v[row], mOut_grad.v[col]);
				}
			}
			
			if(transposeB) {
				writeMatrixToChannelsTransposed<scalar_t, DIMS>(mB_grad, matricesB_grad, flatIdx, totalSize);
			} else {
				writeMatrixToChannels<scalar_t, DIMS>(mB_grad, matricesB_grad, flatIdx, totalSize);
			}
		}
		
	}
}


template<typename scalar_t, int DIMS>
__global__
void k_mul_matrix_vector_GRAD(
		const GridInfo grid,
		const scalar_t* __restrict__ matricesA, const bool transposeA, const bool invertA,
		const scalar_t* __restrict__ vectorsB,
		const scalar_t* __restrict__ vectorsOut_grad,
		scalar_t* __restrict__ matricesA_grad,
		scalar_t* __restrict__ vectorsB_grad
	){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		const Vector<scalar_t, DIMS> vOut_grad = loadVectorFromChannels<scalar_t, DIMS>(vectorsOut_grad, flatIdx, totalSize);
		
		// d vOut / d mA
		if(!invertA){
			const Vector<scalar_t, DIMS> vB = loadVectorFromChannels<scalar_t, DIMS>(vectorsB, flatIdx, totalSize);
			
			MatrixSquare<scalar_t, DIMS> mA_grad = {0};
			for(index_t dim=0; dim<DIMS; ++dim){
				mA_grad.v[dim] = vB * vOut_grad.a[dim];
			}
			
			if(transposeA) {
				writeMatrixToChannelsTransposed<scalar_t, DIMS>(mA_grad, matricesA_grad, flatIdx, totalSize);
			} else {
				writeMatrixToChannels<scalar_t, DIMS>(mA_grad, matricesA_grad, flatIdx, totalSize);
			}
		}
		
		// d vOut / d vB
		{
			MatrixSquare<scalar_t, DIMS> mA = transposeA ? 
				loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesA, flatIdx, totalSize) :
				loadMatrixFromChannels<scalar_t, DIMS>(matricesA, flatIdx, totalSize);
			if(invertA) mA = inverse(mA);
			
			mA = transposed(mA); // to use colum vectors in dot()
			
			Vector<scalar_t, DIMS> vB_grad = {0};
			for(index_t dim=0; dim<DIMS; ++dim){
				vB_grad.a[dim] = dot(mA.v[dim], vOut_grad);
			}
			
			writeVectorToChannels<scalar_t, DIMS>(vB_grad, vectorsB_grad, flatIdx, totalSize);
		}
	}
}


template<typename scalar_t, int DIMS>
__global__
void k_mul_vector_matrix_GRAD(
		const GridInfo grid,
		const scalar_t* __restrict__ vectorsA,
		const scalar_t* __restrict__ matricesB, const bool transposeB, const bool invertB,
		const scalar_t* __restrict__ vectorsOut_grad,
		scalar_t* __restrict__ vectorsA_grad,
		scalar_t* __restrict__ matricesB_grad
	){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		const Vector<scalar_t, DIMS> vOut_grad = loadVectorFromChannels<scalar_t, DIMS>(vectorsOut_grad, flatIdx, totalSize);
		
		// d vOut / d vA
		{
			MatrixSquare<scalar_t, DIMS> mB = transposeB ? 
				loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesB, flatIdx, totalSize) :
				loadMatrixFromChannels<scalar_t, DIMS>(matricesB, flatIdx, totalSize);
			if(invertB) mB = inverse(mB);
			
			Vector<scalar_t, DIMS> vA_grad = {0};
			for(index_t dim=0; dim<DIMS; ++dim){
				vA_grad.a[dim] = dot(mB.v[dim], vOut_grad);
			}
			
			writeVectorToChannels<scalar_t, DIMS>(vA_grad, vectorsA_grad, flatIdx, totalSize);
		}
		
		// d vOut / d mB
		if(!invertB) {
			const Vector<scalar_t, DIMS> vA = loadVectorFromChannels<scalar_t, DIMS>(vectorsA, flatIdx, totalSize);
			
			MatrixSquare<scalar_t, DIMS> mB_grad = {0};
			for(index_t dim=0; dim<DIMS; ++dim){
				mB_grad.v[dim] = vA.a[dim] * vOut_grad;
			}
			
			if(transposeB) {
				writeMatrixToChannelsTransposed<scalar_t, DIMS>(mB_grad, matricesB_grad, flatIdx, totalSize);
			} else {
				writeMatrixToChannels<scalar_t, DIMS>(mB_grad, matricesB_grad, flatIdx, totalSize);
			}
		}
	}
}


template<typename scalar_t, int DIMS>
__global__
void k_mul_vector_vector_GRAD(
		const GridInfo grid,
		const scalar_t* __restrict__ vectorsA,
		const scalar_t* __restrict__ vectorsB,
		const scalar_t* __restrict__ scalarOut_grad,
		scalar_t* __restrict__ vectorsA_grad,
		scalar_t* __restrict__ vectorsB_grad
	){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		const scalar_t sOut_grad = scalarOut_grad[flatIdx];
		
		// d sOut / d vB
		{
			const Vector<scalar_t, DIMS> vB_grad =
				loadVectorFromChannels<scalar_t, DIMS>(vectorsA, flatIdx, totalSize)
				* sOut_grad;
			
			writeVectorToChannels<scalar_t, DIMS>(vB_grad, vectorsB_grad, flatIdx, totalSize);
		}
		
		// d sOut / d vA
		{
			const Vector<scalar_t, DIMS> vA_grad = 
				loadVectorFromChannels<scalar_t, DIMS>(vectorsB, flatIdx, totalSize)
				* sOut_grad;
			
			writeVectorToChannels<scalar_t, DIMS>(vA_grad, vectorsA_grad, flatIdx, totalSize);
		}
		
	}
}


std::vector<torch::Tensor> matmulGrad(
		const torch::Tensor &vectorMatrixA, const torch::Tensor &vectorMatrixB, const torch::Tensor &outputGrad,
		const bool transposeA, const bool invertA,
		const bool transposeB, const bool invertB,
		const bool transposeOutput, const bool invertOutput
	){
	CHECK_INPUT_CUDA(vectorMatrixA);
	TORCH_CHECK(2<vectorMatrixA.dim() && vectorMatrixA.dim()<6, "vectorMatrixA must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(vectorMatrixA.size(0)==1, "vectorMatrixA batch dimension must be 1.");
	const index_t dims = vectorMatrixA.dim()-2;
	TORCH_CHECK(vectorMatrixA.size(1)==dims*dims || vectorMatrixA.size(1)==dims, "vectorMatrixA channel dimension must be a vector or flat square matrix that matches spatial dimensionality.");
	const bool isAmatrix = vectorMatrixA.size(1)==dims*dims;
	
	CHECK_INPUT_CUDA(vectorMatrixB);
	TORCH_CHECK(vectorMatrixB.dim()==(dims+2), "vectorMatrixB dimensionality must match vectorMatrixA.");
	TORCH_CHECK(vectorMatrixB.size(0)==1, "vectorMatrixB batch dimension must be 1.");
	TORCH_CHECK(vectorMatrixB.size(1)==dims*dims || vectorMatrixB.size(1)==dims, "vectorMatrixB channel dimension must be a vector or flat square matrix that matches spatial dimensionality.");
	const bool isBmatrix = vectorMatrixB.size(1)==dims*dims;
	TORCH_CHECK(vectorMatrixA.scalar_type()==vectorMatrixB.scalar_type(), "Data type of vectorMatrixB does not match vectorMatrixA.");
	
	index_t outChannels = 1;
	if(isAmatrix || isBmatrix) outChannels = dims;
	if(isAmatrix && isBmatrix) outChannels = dims*dims;
	
	CHECK_INPUT_CUDA(outputGrad);
	TORCH_CHECK(outputGrad.dim()==(dims+2), "outputGrad dimensionality must match vectorMatrixA.");
	TORCH_CHECK(outputGrad.size(0)==1, "outputGrad batch dimension must be 1.");
	TORCH_CHECK(outputGrad.size(1)==outChannels, "outputGrad channel dimension must match operation type.");
	TORCH_CHECK(vectorMatrixA.scalar_type()==outputGrad.scalar_type(), "Data type of outputGrad does not match vectorMatrixA.");
	
	const GridInfo grid = MakeGridInfo(vectorMatrixA.size(-1), dims>1?vectorMatrixA.size(-2):1, dims>2?vectorMatrixA.size(-3):1, vectorMatrixA.size(1));
	
	torch::Tensor vectorMatrixAgrad = torch::zeros_like(vectorMatrixA);
	torch::Tensor vectorMatrixBgrad = torch::zeros_like(vectorMatrixB);
	
	DISPATCH_FTYPES_DIMS(vectorMatrixA.scalar_type(), dims, "matmul",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
		
		if(isAmatrix && isBmatrix){
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_mul_matrix_matrix_GRAD<scalar_t, dim>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			
			k_mul_matrix_matrix_GRAD<scalar_t, dim><<<gridSize, blockSize>>>(
				grid,
				vectorMatrixA.data_ptr<scalar_t>(), transposeA, invertA,
				vectorMatrixB.data_ptr<scalar_t>(), transposeB, invertB,
				outputGrad.data_ptr<scalar_t>(), transposeOutput, invertOutput,
				vectorMatrixAgrad.data_ptr<scalar_t>(),
				vectorMatrixBgrad.data_ptr<scalar_t>()
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		} else if (isAmatrix && !isBmatrix){
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_mul_matrix_vector_GRAD<scalar_t, dim>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			
			k_mul_matrix_vector_GRAD<scalar_t, dim><<<gridSize, blockSize>>>(
				grid,
				vectorMatrixA.data_ptr<scalar_t>(), transposeA, invertA,
				vectorMatrixB.data_ptr<scalar_t>(),
				outputGrad.data_ptr<scalar_t>(),
				vectorMatrixAgrad.data_ptr<scalar_t>(),
				vectorMatrixBgrad.data_ptr<scalar_t>()
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		} else if (!isAmatrix && isBmatrix){
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_mul_vector_matrix_GRAD<scalar_t, dim>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			
			k_mul_vector_matrix_GRAD<scalar_t, dim><<<gridSize, blockSize>>>(
				grid,
				vectorMatrixA.data_ptr<scalar_t>(),
				vectorMatrixB.data_ptr<scalar_t>(), transposeB, invertB,
				outputGrad.data_ptr<scalar_t>(),
				vectorMatrixAgrad.data_ptr<scalar_t>(),
				vectorMatrixBgrad.data_ptr<scalar_t>()
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		} else {
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_mul_vector_vector_GRAD<scalar_t, dim>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			
			k_mul_vector_vector_GRAD<scalar_t, dim><<<gridSize, blockSize>>>(
				grid,
				vectorMatrixA.data_ptr<scalar_t>(),
				vectorMatrixB.data_ptr<scalar_t>(),
				outputGrad.data_ptr<scalar_t>(),
				vectorMatrixAgrad.data_ptr<scalar_t>(),
				vectorMatrixBgrad.data_ptr<scalar_t>()
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
	);
	
	return {vectorMatrixAgrad, vectorMatrixBgrad};
}