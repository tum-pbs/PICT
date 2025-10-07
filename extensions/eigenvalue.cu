#include "eigenvalue.h"
#include "dispatch.h"

#include <cuda.h>
#include <cuda_runtime.h>

/*
Charles-Alban Deledalle, Loic Denis, Sonia Tabti, Florence Tupin.
Closed-form expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian matrices.
[Research Report] Universit� de Lyon.
2017. ffhal-01501221ff
*/

/* eigen decomposition of hermitian 2x2 matrices (only real-valued)
  matrices: NCDHW with C=4 (row-major matrix, assumed symmetric, only lower triangular values are used)
  eigenvalues: NCDHW with C=2
  eigenvectors: NCDHW with C=4 (row-major matrix with eigenvectors as columns)
*/
template<typename scalar_t>
__global__
void k_EigenDecomposition_2(const GridInfo grid, const scalar_t* matrices, scalar_t* eigenvalues, scalar_t* eigenvectors, const bool normalizeEigenvectors) {
	
	if (eigenvalues==nullptr && eigenvectors==nullptr) return;
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		// a c*
		// c b
		// only lower triangular values needed
		const scalar_t a = matrices[flatIdx]; // a
		//const scalar_t c_star = matrices[flatIdx + totalSize]; // c*
		const scalar_t c = matrices[flatIdx + totalSize*2]; // c
		const scalar_t b = matrices[flatIdx + totalSize*3]; // b
		
		
		// eigenvalues
		scalar_t lambda_1, lambda_2;
		if (c==0) {
			lambda_1 = a;
			lambda_2 = b;
		} else {
			const scalar_t delta = sqrt(4*c*c + (a - b)*(a - b));
			lambda_1 = (a + b - delta)*0.5;
			lambda_2 = (a + b + delta)*0.5;
		}
		
		if (eigenvalues!=nullptr){
			// write in inverse order, otherwise the eigenvalues don't match the oder of the eigenvectors when testing with A= Q Lambda Q^-1
			eigenvalues[flatIdx] = lambda_2;
			eigenvalues[flatIdx + totalSize] = lambda_1;
		}
		
		// eigenvectors
		if (eigenvectors!=nullptr){
			Vector<scalar_t, 2> v_1, v_2;
			if (c==0) {
				v_1.a[0] = 1;
				v_1.a[1] = 0;
				v_2.a[0] = 0;
				v_2.a[1] = 1;
			} else {
				v_1.a[0] = (lambda_2 - b) / c;
				v_1.a[1] = 1;
				v_2.a[0] = (lambda_1 - b) / c;
				v_2.a[1] = 1;

				if(normalizeEigenvectors){
					const scalar_t one = 1;
					v_1 *= one / norm(v_1);
					v_2 *= one / norm(v_2);
				}
			}
			
			// column vectors into flat row-major matrix
			eigenvectors[flatIdx]               = v_1.a[0];
			eigenvectors[flatIdx + totalSize]   = v_2.a[0];
			eigenvectors[flatIdx + totalSize*2] = v_1.a[1];
			eigenvectors[flatIdx + totalSize*3] = v_2.a[1];
		}
	}
}


/* eigen decomposition of hermitian 3x3 matrices (only real-valued)
  matrices: NCDHW with C=9 (row-major matrix, assumed symmetric, only lower triangular values are used)
  eigenvalues: NCDHW with C=3
  eigenvectors: NCDHW with C=9 (row-major matrix with eigenvectors as columns)
*/
template<typename scalar_t>
__global__
void k_EigenDecomposition_3(const GridInfo grid, const scalar_t* matrices, scalar_t* eigenvalues, scalar_t* eigenvectors, const bool normalizeEigenvectors) {
	
	if (eigenvalues==nullptr && eigenvectors==nullptr) return;
	
	const scalar_t pi = 3.1415926536;
	
	const index_t totalSize = grid.stride.w;
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
		
		// a  d* f*
		// d  b  e*
		// f  e  c
		// only lower triangular values needed
		const scalar_t a = matrices[flatIdx];
		const scalar_t d = matrices[flatIdx + totalSize*3];
		const scalar_t b = matrices[flatIdx + totalSize*4];
		const scalar_t f = matrices[flatIdx + totalSize*6];
		const scalar_t e = matrices[flatIdx + totalSize*7];
		const scalar_t c = matrices[flatIdx + totalSize*8];

		const bool isDiagonal = (d==0) && (e==0) && (f==0);
		
		// eigenvalues
		scalar_t lambda_1, lambda_2, lambda_3;
		if (isDiagonal) {
			lambda_1 = a;
			lambda_2 = b;
			lambda_3 = c;
		} else {
			const scalar_t x_1 = a*a + b*b + c*c - a*b - a*c - b*c + 3*(d*d + f*f + e*e);
		
			const scalar_t a_2 = 2*a - b - c;
			const scalar_t b_2 = 2*b - a - c;
			const scalar_t c_2 = 2*c - a - b;
			const scalar_t x_2 = - a_2 * b_2 * c_2 + 9*( c_2 * d*d + b_2 * f*f + a_2 * e*e ) - 54*d*e*f;
			
			scalar_t phi;
			if (x_2==0){
				phi = pi * 0.5;
			} else {
				phi = atan( sqrt(4*x_1*x_1*x_1 - x_2*x_2) / x_2 );
				if (x_2 < 0){
					phi += pi;
				}
			}
			
			const scalar_t sqrt_x_1 = sqrt(x_1);
			const scalar_t third = 1.0 / 3.0;
			
			lambda_1 = ( a + b + c - 2 * sqrt_x_1 * cos(phi * third) ) * third;
			lambda_2 = ( a + b + c + 2 * sqrt_x_1 * cos( (phi - pi) * third ) ) * third;
			lambda_3 = ( a + b + c + 2 * sqrt_x_1 * cos( (phi + pi) * third ) ) * third;
		}
		
		
		if (eigenvalues!=nullptr){
			eigenvalues[flatIdx] = lambda_1;
			eigenvalues[flatIdx + totalSize] = lambda_2;
			eigenvalues[flatIdx + totalSize*2] = lambda_3;
		}
		
		// eigenvectors
		if (eigenvectors!=nullptr){
			if (isDiagonal) {
				eigenvectors[flatIdx]               = 1;
				eigenvectors[flatIdx + totalSize*4] = 1;
				eigenvectors[flatIdx + totalSize*8] = 1;
			} else {
				// TODO: handle m-denominator or f zero
				const scalar_t m_1 = (d*(c - lambda_1) - e*f) / (f*(b - lambda_1) - d*e);
				const scalar_t m_2 = (d*(c - lambda_2) - e*f) / (f*(b - lambda_2) - d*e);
				const scalar_t m_3 = (d*(c - lambda_3) - e*f) / (f*(b - lambda_3) - d*e);
				
				const scalar_t v_1_1 = (lambda_1 - c - e*m_1) / f;
				const scalar_t v_2_1 = (lambda_2 - c - e*m_2) / f;
				const scalar_t v_3_1 = (lambda_3 - c - e*m_3) / f;
				
				if(normalizeEigenvectors) {
					const scalar_t one = 1;
					Vector<scalar_t, 3> v;

					v.a[0] = v_1_1;
					v.a[1] = m_1;
					v.a[2] = 1;
					v *= one / norm(v);
					eigenvectors[flatIdx]               = v.a[0];
					eigenvectors[flatIdx + totalSize*3] = v.a[1];
					eigenvectors[flatIdx + totalSize*6] = v.a[2];

					v.a[0] = v_2_1;
					v.a[1] = m_2;
					v.a[2] = 1;
					v *= one / norm(v);
					eigenvectors[flatIdx + totalSize]   = v.a[0];
					eigenvectors[flatIdx + totalSize*4] = v.a[1];
					eigenvectors[flatIdx + totalSize*7] = v.a[2];

					v.a[0] = v_3_1;
					v.a[1] = m_3;
					v.a[2] = 1;
					v *= one / norm(v);
					eigenvectors[flatIdx + totalSize*2] = v.a[0];
					eigenvectors[flatIdx + totalSize*5] = v.a[1];
					eigenvectors[flatIdx + totalSize*8] = v.a[2];

				} else {
					eigenvectors[flatIdx]               = v_1_1;
					eigenvectors[flatIdx + totalSize]   = v_2_1;
					eigenvectors[flatIdx + totalSize*2] = v_3_1;
					eigenvectors[flatIdx + totalSize*3] = m_1; // v_1_2
					eigenvectors[flatIdx + totalSize*4] = m_2; // v_2_2
					eigenvectors[flatIdx + totalSize*5] = m_3; // v_3_2
					eigenvectors[flatIdx + totalSize*6] = 1; // v_1_3
					eigenvectors[flatIdx + totalSize*7] = 1; // v_2_3
					eigenvectors[flatIdx + totalSize*8] = 1; // v_3_3
				}
			}
		}
	}
}

std::vector<optional<torch::Tensor>> EigenDecomposition(const torch::Tensor &matrices, const bool outputEigenvalues, const bool outputEigenvectors, const bool normalizeEigenvectors){
    CHECK_INPUT_CUDA(matrices);
	TORCH_CHECK(2<matrices.dim() && matrices.dim()<6, "matrices must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(matrices.size(0)==1, "matrices batch dimension must be 1.");
	index_t dims = matrices.dim()-2;
	TORCH_CHECK(matrices.size(1)==dims*dims, "matrices channel dimension must match spatial dimensionality.");
	
	
	const GridInfo grid = MakeGridInfo(matrices.size(-1), dims>1?matrices.size(-2):1, dims>2?matrices.size(-3):1, matrices.size(1));
	
    auto valueOptions = torch::TensorOptions().dtype(matrices.scalar_type()).layout(torch::kStrided).device(matrices.device().type(), matrices.device().index());
	
	std::vector<int64_t> eigenvalueSize;
	eigenvalueSize.push_back(1); // batch size
	eigenvalueSize.push_back(dims); // staggered grid
	for(index_t dim=dims-1;dim>=0;--dim){
		eigenvalueSize.push_back(grid.size.a[dim]); //
	}
	
	optional<torch::Tensor> eigenvalues = nullopt;
	if (outputEigenvalues) {
		eigenvalues = torch::zeros(eigenvalueSize, valueOptions);
	}

	optional<torch::Tensor> eigenvectors = nullopt;
	if (outputEigenvectors) {
		eigenvectors = torch::zeros_like(matrices);
	}

    AT_DISPATCH_FLOATING_TYPES(matrices.scalar_type(), "EigenDecomposition", ([&] {
		int minGridSize = 0, blockSize = 0, gridSize = 0;
		switch (dims)
		{
		case 2:
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_EigenDecomposition_2<scalar_t>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			k_EigenDecomposition_2<scalar_t><<<gridSize, blockSize>>>(
				grid, matrices.data_ptr<scalar_t>(),
				eigenvalues.has_value() ? eigenvalues.value().data_ptr<scalar_t>() : nullptr,
				eigenvectors.has_value() ? eigenvectors.value().data_ptr<scalar_t>() : nullptr,
				normalizeEigenvectors
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			break;
		case 3:
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_EigenDecomposition_3<scalar_t>, 0, 0);
			gridSize = (grid.stride.w + blockSize - 1) / blockSize;
			k_EigenDecomposition_3<scalar_t><<<gridSize, blockSize>>>(
				grid, matrices.data_ptr<scalar_t>(),
				eigenvalues.has_value() ? eigenvalues.value().data_ptr<scalar_t>() : nullptr,
				eigenvectors.has_value() ? eigenvectors.value().data_ptr<scalar_t>() : nullptr,
				normalizeEigenvectors
			);
			CUDA_CHECK_RETURN(cudaDeviceSynchronize());
			break;
		default:
			break;
		}
	}));
	
	return {eigenvalues, eigenvectors};
}