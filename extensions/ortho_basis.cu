#include "ortho_basis.h"
#include "dispatch.h"
#include "matrix_vector_ops.h"

#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__device__
Vector<index_t, 1> getOrderDescending(const Vector<scalar_t, 1> &v){
	return {0};
}

template<typename scalar_t>
__device__
Vector<index_t, 2> getOrderDescending(const Vector<scalar_t, 2> &v){
	Vector<index_t, 2> order = {.a={0,1}};
	if(v.a[0] < v.a[1]){
		order.a[0] = 1;
		order.a[1] = 0;
	}
	return order;
}

template<typename scalar_t>
__device__
Vector<index_t, 3> getOrderDescending(const Vector<scalar_t, 3> &v){
	Vector<index_t, 3> order = {.a={0,1,2}};
	if(v.a[0] > v.a[1]){
		if(v.a[1] < v.a[2]){
			if(v.a[0] > v.a[2]){
				// {0,2,1};
				order.a[0] = 0;
				order.a[1] = 2;
				order.a[2] = 1;
			} else { // 0<2
				// {2,0,1};
				order.a[0] = 2;
				order.a[1] = 0;
				order.a[2] = 1;
			}
		}
	} else { // 0<1
		if(v.a[1] > v.a[2]){
			if(v.a[0] > v.a[2]){
				// {1,0,2};
				order.a[0] = 1;
				order.a[1] = 0;
				order.a[2] = 2;
			} else { // 0<2
				// {1,2,0};
				order.a[0] = 1;
				order.a[1] = 2;
				order.a[2] = 0;
			}
		} else { // 1<2
			// {2,1,0};
			order.a[0] = 2;
			order.a[1] = 1;
			order.a[2] = 0;
		}
	}
	return order;
}

template<typename scalar_t>
__device__
scalar_t RHSmeasure(const MatrixSquare<scalar_t, 1> &m){
	return 1;
}

template<typename scalar_t>
__device__
scalar_t RHSmeasure(const MatrixSquare<scalar_t, 2> &m){
	return cross(m.v[0], m.v[1]);
}

template<typename scalar_t>
__device__
scalar_t RHSmeasure(const MatrixSquare<scalar_t, 3> &m){
	return dot(cross(m.v[0], m.v[1]), m.v[2]);
}



template<typename scalar_t, int DIMS>
__global__
void k_makeBasisUnique(const GridInfo grid, const scalar_t* basisMatrixIn, const scalar_t* sortingVector, scalar_t* basisMatrixOut){
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		Vector<scalar_t, DIMS> measure;
		MatrixSquare<scalar_t, DIMS> basis;
		{
			// basis matrices are stored as row-major matrix with basis vectors as columns.
			// load basis vectors as rows for easier access via matrix.v[row]
			MatrixSquare<scalar_t, DIMS> m = loadMatrixFromChannelsTransposed<scalar_t, DIMS>(basisMatrixIn, flatIdx, totalSize);
			Vector<scalar_t, DIMS> u = loadVectorFromChannels<scalar_t, DIMS>(sortingVector, flatIdx, totalSize);
			
			// Feature alignment: sort basis vectors
			measure = abs(matmul(m, u));
			
			const Vector<index_t, DIMS> order = getOrderDescending(measure);
			
			for(index_t i=0; i<DIMS; ++i){
				const index_t o = order.a[i];
				
				basis.v[i] = m.v[o];
			}
			
			measure = matmul(basis, u);
		}
		
		const scalar_t mOne = -1; // to have the correct type
		
		// Positive features:
		for(index_t i=0; i<(DIMS-1); ++i){
			if(measure.a[i] < 0){
				basis.v[i] *= mOne;
			}
		}
		
		// Right hand side system:
		if(RHSmeasure(basis) < 0){
			basis.v[DIMS-1] *= mOne;
		}
		
		writeMatrixToChannelsTransposed<scalar_t, DIMS>(basis, basisMatrixOut, flatIdx, totalSize);
	}
}



torch::Tensor MakeBasisUnique(const torch::Tensor &basisMatrices, const torch::Tensor &sortingVectors, const bool inPlace){
	CHECK_INPUT_CUDA(basisMatrices);
	TORCH_CHECK(2<basisMatrices.dim() && basisMatrices.dim()<6, "basisMatrices must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(basisMatrices.size(0)==1, "basisMatrices batch dimension must be 1.");
	index_t dims = basisMatrices.dim()-2;
	TORCH_CHECK(basisMatrices.size(1)==dims*dims, "basisMatrices channel dimension must match spatial dimensionality.");
	
	CHECK_INPUT_CUDA(sortingVectors);
	TORCH_CHECK(sortingVectors.dim()==(dims+2), "sortingVectors dimensionality must match basisMatrices.");
	TORCH_CHECK(sortingVectors.size(0)==1, "sortingVectors batch dimension must be 1.");
	TORCH_CHECK(sortingVectors.size(1)==dims, "sortingVectors channel dimension must match spatial dimensionality.");
	
	const GridInfo grid = MakeGridInfo(basisMatrices.size(-1), dims>1?basisMatrices.size(-2):1, dims>2?basisMatrices.size(-3):1, basisMatrices.size(1));
	
	torch::Tensor basisMatricesOut = inPlace ? basisMatrices : torch::zeros_like(basisMatrices);
	
	DISPATCH_FTYPES_DIMS(basisMatrices.scalar_type(), dims, "MakeBasisUnique",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_makeBasisUnique<scalar_t, dim>, 0, 0);
		gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		
		k_makeBasisUnique<scalar_t, dim><<<gridSize, blockSize>>>(
			grid,
			basisMatrices.data_ptr<scalar_t>(),
			sortingVectors.data_ptr<scalar_t>(),
			basisMatricesOut.data_ptr<scalar_t>()
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return basisMatricesOut;
}