#include "matrix_vector_ops.h"
#include "dispatch.h"

#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t, int DIMS>
__global__
void k_mul_matrix_matrix(
		const GridInfo grid,
		const scalar_t* __restrict__ matricesA, const bool transposeA, const bool invertA,
		const scalar_t* __restrict__ matricesB, const bool transposeB, const bool invertB,
		scalar_t* __restrict__ matricesOut, const bool transposeOutput, const bool invertOutput
	){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		MatrixSquare<scalar_t, DIMS> mA = transposeA ? 
			loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesA, flatIdx, totalSize) :
			loadMatrixFromChannels<scalar_t, DIMS>(matricesA, flatIdx, totalSize);
		if(invertA) mA = inverse(mA);
		
		MatrixSquare<scalar_t, DIMS> mB = transposeB ? 
			loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesB, flatIdx, totalSize) :
			loadMatrixFromChannels<scalar_t, DIMS>(matricesB, flatIdx, totalSize);
		if(invertB) mB = inverse(mB);
		
		MatrixSquare<scalar_t, DIMS> mOut = matmul(mA, mB);
		if(invertOutput) mOut = inverse(mOut);
		
		if(transposeOutput){
			writeMatrixToChannelsTransposed<scalar_t, DIMS>(mOut, matricesOut, flatIdx, totalSize);
		} else {
			writeMatrixToChannels<scalar_t, DIMS>(mOut, matricesOut, flatIdx, totalSize);
		}
	}
}

template<typename scalar_t, int DIMS>
__global__
void k_mul_matrix_vector(
		const GridInfo grid,
		const scalar_t* __restrict__ matricesA, const bool transposeA, const bool invertA,
		const scalar_t* __restrict__ vectorsB,
		scalar_t* __restrict__ vectorsOut
	){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		MatrixSquare<scalar_t, DIMS> mA = transposeA ? 
			loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesA, flatIdx, totalSize) :
			loadMatrixFromChannels<scalar_t, DIMS>(matricesA, flatIdx, totalSize);
		if(invertA) mA = inverse(mA);
		
		const Vector<scalar_t, DIMS> vB = loadVectorFromChannels<scalar_t, DIMS>(vectorsB, flatIdx, totalSize);
		
		const Vector<scalar_t, DIMS> vOut = matmul(mA, vB);
		
		writeVectorToChannels<scalar_t, DIMS>(vOut, vectorsOut, flatIdx, totalSize);
	}
}

template<typename scalar_t, int DIMS>
__global__
void k_mul_vector_matrix(
		const GridInfo grid,
		const scalar_t* __restrict__ vectorsA,
		const scalar_t* __restrict__ matricesB, const bool transposeB, const bool invertB,
		scalar_t* __restrict__ vectorsOut
	){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		const Vector<scalar_t, DIMS> vA = loadVectorFromChannels<scalar_t, DIMS>(vectorsA, flatIdx, totalSize);
		
		MatrixSquare<scalar_t, DIMS> mB = transposeB ? 
			loadMatrixFromChannelsTransposed<scalar_t, DIMS>(matricesB, flatIdx, totalSize) :
			loadMatrixFromChannels<scalar_t, DIMS>(matricesB, flatIdx, totalSize);
		if(invertB) mB = inverse(mB);
		
		const Vector<scalar_t, DIMS> vOut = matmul(vA, mB);
		
		writeVectorToChannels<scalar_t, DIMS>(vOut, vectorsOut, flatIdx, totalSize);
	}
}

template<typename scalar_t, int DIMS>
__global__
void k_mul_vector_vector(
		const GridInfo grid,
		const scalar_t* __restrict__ vectorsA,
		const scalar_t* __restrict__ vectorsB,
		scalar_t* __restrict__ scalarOut
	){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		const Vector<scalar_t, DIMS> vA = loadVectorFromChannels<scalar_t, DIMS>(vectorsA, flatIdx, totalSize);
		const Vector<scalar_t, DIMS> vB = loadVectorFromChannels<scalar_t, DIMS>(vectorsB, flatIdx, totalSize);
		
		const scalar_t sOut = dot(vA, vB);
		
		scalarOut[flatIdx] = sOut;
	}
}



torch::Tensor matmul(
		const torch::Tensor &vectorMatrixA, const torch::Tensor &vectorMatrixB,
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
	
	const GridInfo grid = MakeGridInfo(vectorMatrixA.size(-1), dims>1?vectorMatrixA.size(-2):1, dims>2?vectorMatrixA.size(-3):1, vectorMatrixA.size(1));
	
	index_t outChannels = 1;
	if(isAmatrix || isBmatrix) outChannels = dims;
	if(isAmatrix && isBmatrix) outChannels = dims*dims;
	
	std::vector<int64_t> outputSize = {1, outChannels};
	for(index_t dim=dims-1;dim>=0;--dim){
		outputSize.push_back(grid.size.a[dim]); //
	}
	
	auto valueOptions = torch::TensorOptions().dtype(vectorMatrixA.scalar_type()).layout(torch::kStrided).device(vectorMatrixA.device().type(), vectorMatrixA.device().index());
	torch::Tensor output = torch::zeros(outputSize, valueOptions);
	
	DISPATCH_FTYPES_DIMS(vectorMatrixA.scalar_type(), dims, "matmul",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
		
		if(isAmatrix && isBmatrix){
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_mul_matrix_matrix<scalar_t, dim>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			
			k_mul_matrix_matrix<scalar_t, dim><<<gridSize, blockSize>>>(
				grid,
				vectorMatrixA.data_ptr<scalar_t>(), transposeA, invertA,
				vectorMatrixB.data_ptr<scalar_t>(), transposeB, invertB,
				output.data_ptr<scalar_t>(), transposeOutput, invertOutput
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		} else if (isAmatrix && !isBmatrix){
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_mul_matrix_vector<scalar_t, dim>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			
			k_mul_matrix_vector<scalar_t, dim><<<gridSize, blockSize>>>(
				grid,
				vectorMatrixA.data_ptr<scalar_t>(), transposeA, invertA,
				vectorMatrixB.data_ptr<scalar_t>(),
				output.data_ptr<scalar_t>()
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		} else if (!isAmatrix && isBmatrix){
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_mul_vector_matrix<scalar_t, dim>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			
			k_mul_vector_matrix<scalar_t, dim><<<gridSize, blockSize>>>(
				grid,
				vectorMatrixA.data_ptr<scalar_t>(),
				vectorMatrixB.data_ptr<scalar_t>(), transposeB, invertB,
				output.data_ptr<scalar_t>()
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		} else {
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_mul_vector_vector<scalar_t, dim>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			
			k_mul_vector_vector<scalar_t, dim><<<gridSize, blockSize>>>(
				grid,
				vectorMatrixA.data_ptr<scalar_t>(),
				vectorMatrixB.data_ptr<scalar_t>(),
				output.data_ptr<scalar_t>()
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		}
	);
	
	return output;
}



template<typename scalar_t, int DIMS>
__global__
void k_vectorToDiagMatrix(
		const GridInfo grid,
		const scalar_t* __restrict__ vectors,
		scalar_t* __restrict__ matricesOut
	){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		const Vector<scalar_t, DIMS> v = loadVectorFromChannels<scalar_t, DIMS>(vectors, flatIdx, totalSize);
		
		/* MatrixSquare<scalar_t, DIMS> m = {0};
		for(index_t i=0; i<DIMS; ++i){
			m.a[i][i] = v.a[i];
		}
		
		writeMatrixToChannels<scalar_t, DIMS>(m, matricesOut, flatIdx, totalSize); */
		
		// matricesOut is initialized zero, so only write new diagonal values
		for(index_t i=0; i<DIMS; ++i){
			matricesOut[flatIdx + (i*DIMS + i)*totalSize] = v.a[i];
		}
	}
}

torch::Tensor VectorToDiagMatrix(const torch::Tensor &vectors){
	CHECK_INPUT_CUDA(vectors);
	TORCH_CHECK(2<vectors.dim() && vectors.dim()<6, "vectors must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(vectors.size(0)==1, "vectors batch dimension must be 1.");
	index_t dims = vectors.dim()-2;
	TORCH_CHECK(vectors.size(1)==dims, "vectors channel dimension must match spatial dimensionality.");
	
	const GridInfo grid = MakeGridInfo(vectors.size(-1), dims>1?vectors.size(-2):1, dims>2?vectors.size(-3):1, vectors.size(1));
	
	std::vector<int64_t> outputSize = {1, dims*dims};
	for(index_t dim=dims-1;dim>=0;--dim){
		outputSize.push_back(grid.size.a[dim]); //
	}
	
	auto valueOptions = torch::TensorOptions().dtype(vectors.scalar_type()).layout(torch::kStrided).device(vectors.device().type(), vectors.device().index());
	torch::Tensor matricesOut = torch::zeros(outputSize, valueOptions);
	
	DISPATCH_FTYPES_DIMS(vectors.scalar_type(), dims, "InvertMatrix",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_vectorToDiagMatrix<scalar_t, dim>, 0, 0);
		gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		
		k_vectorToDiagMatrix<scalar_t, dim><<<gridSize, blockSize>>>(
			grid,
			vectors.data_ptr<scalar_t>(),
			matricesOut.data_ptr<scalar_t>()
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return matricesOut;
}



template<typename scalar_t, int DIMS>
__global__
void k_inverseMatrix(const GridInfo grid, const scalar_t* matricesIn, scalar_t* matricesOut){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		MatrixSquare<scalar_t, DIMS> m = loadMatrixFromChannels<scalar_t, DIMS>(matricesIn, flatIdx, totalSize);
		
		m = inverse(m);
		
		writeMatrixToChannels<scalar_t, DIMS>(m, matricesOut, flatIdx, totalSize);
	}
}

torch::Tensor InvertMatrix(const torch::Tensor &matrices, const bool inPlace){
	CHECK_INPUT_CUDA(matrices);
	TORCH_CHECK(2<matrices.dim() && matrices.dim()<6, "matrices must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(matrices.size(0)==1, "matrices batch dimension must be 1.");
	index_t dims = matrices.dim()-2;
	TORCH_CHECK(matrices.size(1)==dims*dims, "matrices channel dimension must match spatial dimensionality.");
	
	const GridInfo grid = MakeGridInfo(matrices.size(-1), dims>1?matrices.size(-2):1, dims>2?matrices.size(-3):1, matrices.size(1));
	
	torch::Tensor matricesOut = inPlace ? matrices : torch::zeros_like(matrices);
	
	DISPATCH_FTYPES_DIMS(matrices.scalar_type(), dims, "InvertMatrix",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_inverseMatrix<scalar_t, dim>, 0, 0);
		gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		
		k_inverseMatrix<scalar_t, dim><<<gridSize, blockSize>>>(
			grid,
			matrices.data_ptr<scalar_t>(),
			matricesOut.data_ptr<scalar_t>()
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return matricesOut;
}