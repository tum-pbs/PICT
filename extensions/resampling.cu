#include "resampling.h"
#include "dispatch.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>



template<typename scalar_t, int DIMS>
__global__
void k_SampleTransformedGridGlobalToLocal(const scalar_t *d_globalData, const MatrixSquare<scalar_t, DIMS+1> t, const MatrixSquare<scalar_t, DIMS+1> tInv, const GridInfo globalGrid,
										 scalar_t *d_localData, const scalar_t *d_localCoords, const GridInfo localGrid,
										 const index_t channels, const BoundarySampling boundaryMode, const scalar_t constantValue){
	
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < localGrid.stride.w; flatIdx += blockDim.x * gridDim.x){
		const I4 pos = unflattenIndex(flatIdx, localGrid);
		// load global/world coordinates of current cell
		Vector<scalar_t, DIMS+1> coords = {.a={0}};
		for(index_t dim=0; dim<DIMS; ++dim){
			coords.a[dim] = d_localCoords[flatIdx + localGrid.stride.w*dim];
		}
		coords.a[DIMS] = 1;
		
		// get coords in object space of globalGrid
		const Vector<scalar_t, DIMS+1> sampleCoords = matmul(tInv, coords);
		
		// sample from d_globalData, for each channel
		
		for(index_t channel=0; channel<channels; ++channel){
			scalar_t data = 0;
			
			// 2^DIMS sample positions (neighbor cells around sampling location)
			for(index_t idx=0; idx<(DIMS<<1); ++idx){ // (-x,-y,-z) 000, (+x,-y,-z) 001, (-x,+y,-z) 010, ...
				I4 samplePos = {.a={0}};
				scalar_t weight = 1;
				bool isOOB = false;
				// DIMS position components 
				for(index_t c=0; c<DIMS; ++c){ // x,y,z
					bool isUpper = (idx&(1<<c))!=0;
					samplePos.a[c] = isUpper ? ceil(sampleCoords.a[c]) : floor(sampleCoords.a[c]);
					isOOB = isOOB || (samplePos.a[c]<0 || samplePos.a[c]>=globalGrid.size.a[c]);
					weight *= isUpper ? frac(sampleCoords.a[c]) : 1 - frac(sampleCoords.a[c]);
				}
				
				// TODO: boundary handling
				if(isOOB && boundaryMode==BoundarySampling::CLAMP){
					for(index_t c=0; c<DIMS; ++c){
						if(samplePos.a[c]<0) samplePos.a[c]=0;
						else if(samplePos.a[c]>=globalGrid.size.a[c]) samplePos.a[c]=globalGrid.size.a[c]-1;
					}
				}
				
				if(isOOB && boundaryMode==BoundarySampling::CONSTANT){
					data += constantValue * weight;
				} else{
					const index_t samplePosFlat = flattenIndex(samplePos, globalGrid);
					data += d_globalData[samplePosFlat + globalGrid.stride.w*channel] * weight;
				}
				
			}
			
			d_localData[flatIdx + localGrid.stride.w*channel] = data; //data;
		}
	
	}
}

template<typename scalar_t, int DIMS>
void _SampleTransformedGridGlobalToLocal(const scalar_t *d_globalData, const scalar_t *h_globalTransform, const GridInfo &globalGrid,
										 scalar_t *d_localData, const scalar_t *d_localCoords, const GridInfo &localGrid,
										 const index_t channels, const BoundarySampling boundaryMode, const scalar_t constantValue){
	
	// make transform from tensor to matrix struct
	MatrixSquare<scalar_t, DIMS+1> t;
	for(index_t row=0;row<DIMS+1;++row){
		for(index_t col=0; col<DIMS+1; ++col){
			t.a[row][col] = h_globalTransform[row*(DIMS+1) + col];
		}
	}
	// get inverse transform
	MatrixSquare<scalar_t, DIMS+1> tInv = inverse(t);
	
	// kernel over all points in localCoords
	int minGridSize = 0, blockSize = 0, gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_SampleTransformedGridGlobalToLocal<scalar_t, DIMS>, 0, 0);
	gridSize = (localGrid.stride.w + blockSize - 1) / blockSize;
	
	k_SampleTransformedGridGlobalToLocal<scalar_t, DIMS><<<gridSize, blockSize>>>(
		d_globalData, t, tInv, globalGrid,
		d_localData, d_localCoords, localGrid,
		channels, boundaryMode, constantValue);
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}


// transformation: OS to global/WS
// inverse: WS to OS

torch::Tensor SampleTransformedGridGlobalToLocal(const torch::Tensor &globalData, const torch::Tensor &globalTransform, const torch::Tensor &localCoords,
												 const BoundarySampling boundaryMode, const torch::Tensor &constantValue){
	// input checks
    CHECK_INPUT_CUDA(globalData);
	TORCH_CHECK(2<globalData.dim() && globalData.dim()<6, "globalData must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(globalData.size(0)==1, "globalData batch dimension must be 1.");
	const index_t batchSize = globalData.size(0);
	const index_t dataChannels = globalData.size(1);
	const index_t dims = globalData.dim()-2;
	
	CHECK_INPUT_HOST(globalTransform);
	TORCH_CHECK(globalTransform.dim() == 3, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(0)==batchSize, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(1)==dims+1, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(2)==dims+1, "globalTransform must have dimension 3D.");
	
    CHECK_INPUT_CUDA(localCoords);
	TORCH_CHECK(localCoords.dim() == dims+2, "localCoords dimensionality must match globalData.");
	TORCH_CHECK(localCoords.size(0)==batchSize, "localCoords batch dimension must be 1.");
	TORCH_CHECK(localCoords.size(1)==dims, "localCoords channel dimension must match spatial dimensionality.");
	
	CHECK_INPUT_HOST(constantValue);
	TORCH_CHECK(constantValue.dim() == 1, "constantValue must have shape [1].");
	TORCH_CHECK(constantValue.size(0)==1, "constantValue must have shape [1].");
	
	
	const GridInfo globalGrid = MakeGridInfo(globalData.size(-1), dims>1?globalData.size(-2):1, dims>2?globalData.size(-3):1, globalData.size(1));
	const GridInfo localGrid = MakeGridInfo(localCoords.size(-1), dims>1?localCoords.size(-2):1, dims>2?localCoords.size(-3):1, localCoords.size(1));
	
	// make output tensor
	auto valueOptions = torch::TensorOptions().dtype(globalData.scalar_type()).layout(torch::kStrided).device(globalData.device().type(), globalData.device().index());
	std::vector<int64_t> localShape;
	localShape.push_back(batchSize); // batch
	localShape.push_back(dataChannels);
	for(index_t dim=dims-1;dim>=0;--dim){
		localShape.push_back(localGrid.size.a[dim]); //
	}
	torch::Tensor localData = torch::zeros(localShape, valueOptions);
	
	DISPATCH_FTYPES_DIMS(globalData.scalar_type(), dims, "CoordsToTransforms",
		_SampleTransformedGridGlobalToLocal<scalar_t, dim>(
			globalData.data_ptr<scalar_t>(), globalTransform.data_ptr<scalar_t>(), globalGrid,
			localData.data_ptr<scalar_t>(), localCoords.data_ptr<scalar_t>(), localGrid,
			dataChannels, boundaryMode, constantValue.data_ptr<scalar_t>()[0]
		);
	);
	
	return localData;
}

// Local to Global

template<typename scalar_t>
scalar_t __device__ constexpr getEps();
template<>
float __device__ constexpr getEps<float>() { return 1e-8f; }
template<>
double __device__ constexpr getEps<double>() { return 1e-12; }

template<typename scalar_t>
__global__
void k_MakeIndicatorFromWeights(const scalar_t *d_weightNorm, const GridInfo grid, bool *d_indicator_out){
	
	const scalar_t eps = getEps<scalar_t>();
	
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < grid.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const scalar_t normWeight = d_weightNorm[flatIdx];
		d_indicator_out[flatIdx] = normWeight > eps;
	}
}

template<typename scalar_t, int DIMS>
__global__
void k_FillFromNeighbors(scalar_t *d_globalData, const bool *d_indicator_in, const GridInfo grid, const index_t channels, bool *d_indicator_out){
	
	
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < grid.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		bool isValid = d_indicator_in[flatIdx];
			
		if(!isValid){
			const I4 pos = unflattenIndex(flatIdx, grid);
			
			bool isValidNeighbor[DIMS*2] = {false};
			index_t validNeighbors = 0;
			for(index_t face=0; face<(DIMS*2); ++face){
				const index_t axis = face>>1;
				I4 posN = pos;
				posN.a[axis] += (face&1)*2 -1;
				if(0<=posN.a[axis] && posN.a[axis]<grid.size.a[axis]){
					const index_t flatIdxN = flattenIndex(posN, grid);
					isValidNeighbor[face] = d_indicator_in[flatIdxN];
				} else {
					isValidNeighbor[face] = false;
				}
				if(isValidNeighbor[face]){ ++validNeighbors; }
			}
			
			if(validNeighbors>0){
				for(index_t channel=0; channel<channels; ++channel){
					scalar_t value = 0;
					
					for(index_t face=0; face<(DIMS*2); ++face){
						if(isValidNeighbor[face]){
							const index_t axis = face>>1;
							I4 posN = pos;
							posN.a[axis] += (face&1)*2 -1;
							//if(0<=posN.a[axis] && posN.a[axis]<grid.size.a[axis]){ not needed since this neighbor would not be valid in isValidNeighbor
							posN.w = channel;
							const index_t flatIdxNchannel = flattenIndex(posN, grid);
							value += d_globalData[flatIdxNchannel];
						}
					}
					
					value /= validNeighbors;
					d_globalData[flatIdx + grid.stride.w*channel] = value;
				}
				isValid = true;
			}
		}
		
		d_indicator_out[flatIdx] = isValid;
	}
}

template<typename scalar_t, int DIMS>
void _FillEmptyCells(torch::Tensor &globalData, const scalar_t *d_weightNorm, const GridInfo &grid,
					 const index_t channels, const index_t fillMaxSteps){
	
	if(fillMaxSteps<1){ return; }
	
	const index_t checkAllFilledSteps = 4;
	
	//py::print("_FillEmptyCells");
	
	// make indicator grids
	auto tensorOptions = torch::TensorOptions().dtype(torch::kBool).layout(torch::kStrided).device(globalData.device().type(), globalData.device().index());
	torch::Tensor indicators[2] = {torch::zeros(grid.stride.w, tensorOptions), torch::zeros(grid.stride.w, tensorOptions)};
	
	bool *pp_indicator[2] = {indicators[0].data_ptr<bool>(), indicators[1].data_ptr<bool>()};
	
	// fill first indicator grid from weightNorm
	{
		//py::print("k_MakeIndicatorFromWeights");
		int minGridSize = 0, blockSize = 0, gridSize = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_MakeIndicatorFromWeights<scalar_t>, 0, 0);
		gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		k_MakeIndicatorFromWeights<scalar_t><<<gridSize, blockSize>>>(
			d_weightNorm, grid, pp_indicator[0]
		);
	}
	
	{
		int minGridSize = 0, blockSize = 0, gridSize = 0;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_FillFromNeighbors<scalar_t, DIMS>, 0, 0);
		gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		for(index_t step=0; step<fillMaxSteps; ++step){
			//py::print("step", step);
			if((step%checkAllFilledSteps)==0){ //8
				if(torch::all(indicators[step%2]).cpu().data_ptr<bool>()[0]){
					//py::print("break");
					break;
				}
			}
			
			bool *p_indicator_in = pp_indicator[step%2];
			bool *p_indicator_out = pp_indicator[(step+1)%2];
			
			k_FillFromNeighbors<scalar_t, DIMS><<<gridSize, blockSize>>>(
				globalData.data_ptr<scalar_t>(), p_indicator_in, grid, channels, p_indicator_out
			);
		}
	}
}


template<typename scalar_t, int DIMS>
__global__
void k_SampleTransformedGridLocalToGlobal(const scalar_t *d_localData, const scalar_t *d_localCoords, const GridInfo localGrid,
										 scalar_t *d_globalData, scalar_t *d_weightNorm, const MatrixSquare<scalar_t, DIMS+1> t, const MatrixSquare<scalar_t, DIMS+1> tInv, const GridInfo globalGrid,
										 const index_t channels){
	
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < localGrid.stride.w; flatIdx += blockDim.x * gridDim.x){
		const I4 pos = unflattenIndex(flatIdx, localGrid);
		// load global/world coordinates of current cell
		Vector<scalar_t, DIMS+1> coords = {.a={0}};
		for(index_t dim=0; dim<DIMS; ++dim){
			coords.a[dim] = d_localCoords[flatIdx + localGrid.stride.w*dim];
		}
		coords.a[DIMS] = 1;
		
		// get coords in object space of globalGrid
		const Vector<scalar_t, DIMS+1> sampleCoords = matmul(tInv, coords);
		
		// sample from d_globalData, for each channel
		
		for(index_t channel=0; channel<channels; ++channel){
			scalar_t data = d_localData[flatIdx + localGrid.stride.w*channel];
			
			// 2^DIMS sample positions (neighbor cells around sampling location)
			for(index_t idx=0; idx<(DIMS<<1); ++idx){ // (-x,-y,-z) 000, (+x,-y,-z) 001, (-x,+y,-z) 010, ...
				I4 samplePos = {.a={0}};
				scalar_t weight = 1;
				bool isOOB = false;
				// DIMS position components 
				for(index_t c=0; c<DIMS; ++c){ // x,y,z
					bool isUpper = (idx&(1<<c))!=0;
					samplePos.a[c] = isUpper ? ceil(sampleCoords.a[c]) : floor(sampleCoords.a[c]);
					isOOB = isOOB || (samplePos.a[c]<0 || samplePos.a[c]>=globalGrid.size.a[c]);
					weight *= isUpper ? frac(sampleCoords.a[c]) : 1 - frac(sampleCoords.a[c]);
				}
				if(!isOOB){
					const index_t samplePosFlat = flattenIndex(samplePos, globalGrid);
					atomicAdd(d_globalData + (samplePosFlat + globalGrid.stride.w*channel), data*weight);
					if(channel==0){
						atomicAdd(d_weightNorm + samplePosFlat, weight);
					}
				}
			
			}
		
		}
	
	}
}

template<typename scalar_t>
__global__
void k_NormScatteredWithWeight(scalar_t *d_data, const scalar_t *d_weightNorm, const GridInfo grid, const index_t channels){
	
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < grid.stride.w; flatIdx += blockDim.x * gridDim.x){
		//const I4 pos = unflattenIndex(flatIdx, grid);
		
		const scalar_t normWeight = d_weightNorm[flatIdx];
		const scalar_t eps = getEps<scalar_t>();
		
		if(normWeight > eps){
			for(index_t channel=0; channel<channels; ++channel){
				scalar_t data = d_data[flatIdx + grid.stride.w*channel];
				data /= normWeight;
				d_data[flatIdx + grid.stride.w*channel] = data;
			}
		}
	}
}

template<typename scalar_t, int DIMS>
void _SampleTransformedGridLocalToGlobal(const scalar_t *d_localData, const scalar_t *d_localCoords, const GridInfo &localGrid,
										 torch::Tensor &globalData, scalar_t *d_weightNorm, const scalar_t *h_globalTransform, const GridInfo &globalGrid,
										 const index_t channels, const index_t fillMaxSteps){
	
	// make transform from tensor to matrix struct
	MatrixSquare<scalar_t, DIMS+1> t;
	for(index_t row=0;row<DIMS+1;++row){
		for(index_t col=0; col<DIMS+1; ++col){
			t.a[row][col] = h_globalTransform[row*(DIMS+1) + col];
		}
	}
	// get inverse transform
	MatrixSquare<scalar_t, DIMS+1> tInv = inverse(t);
	
	// kernel over all points in localData
	int minGridSize = 0, blockSize = 0, gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_SampleTransformedGridLocalToGlobal<scalar_t, DIMS>, 0, 0);
	gridSize = (localGrid.stride.w + blockSize - 1) / blockSize;
	
	k_SampleTransformedGridLocalToGlobal<scalar_t, DIMS><<<gridSize, blockSize>>>(
		d_localData, d_localCoords, localGrid,
		globalData.data_ptr<scalar_t>(), d_weightNorm, t, tInv, globalGrid,
		channels);
	
	// kernel over all points in globalData
	minGridSize = 0, blockSize = 0, gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_NormScatteredWithWeight<scalar_t>, 0, 0);
	gridSize = (globalGrid.stride.w + blockSize - 1) / blockSize;
	k_NormScatteredWithWeight<scalar_t><<<gridSize, blockSize>>>(
		globalData.data_ptr<scalar_t>(), d_weightNorm, globalGrid, channels
	);
	
	_FillEmptyCells<scalar_t, DIMS>(globalData, d_weightNorm, globalGrid, channels, fillMaxSteps);
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

std::vector<torch::Tensor> SampleTransformedGridLocalToGlobal(const torch::Tensor &localData, const torch::Tensor &localCoords, const torch::Tensor &globalTransform, const torch::Tensor &t_globalShape, const index_t fillMaxSteps){
	// globalShape is x,y,z
	// input checks
    CHECK_INPUT_CUDA(localData);
	TORCH_CHECK(2<localData.dim() && localData.dim()<6, "localData must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(localData.size(0)==1, "localData batch dimension must be 1.");
	const index_t batchSize = localData.size(0);
	const index_t dataChannels = localData.size(1);
	const index_t dims = localData.dim()-2;
	
    CHECK_INPUT_CUDA(localCoords);
	TORCH_CHECK(localCoords.dim() == dims+2, "localCoords dimensionality must match localData.");
	TORCH_CHECK(localCoords.size(0)==batchSize, "localCoords batch dimension must be 1.");
	TORCH_CHECK(localCoords.size(1)==dims, "localCoords channel dimension must match spatial dimensionality.");
	for(index_t dim=0; dim<dims; ++dim){
		TORCH_CHECK(localCoords.size(dim+2)==localData.size(dim+2), "localCoords channel spatial dimensions must match localData.");
	}
	
	CHECK_INPUT_HOST(globalTransform);
	TORCH_CHECK(globalTransform.dim() == 3, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(0)==batchSize, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(1)==dims+1, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(2)==dims+1, "globalTransform must have dimension 3D.");
	
	CHECK_INPUT_HOST(t_globalShape); // x,y,z
	TORCH_CHECK(t_globalShape.dim() == 1, "globalShape must have flat shape.");
	TORCH_CHECK(t_globalShape.size(0)==dims, "globalShape must match spatial dimensionality.");
	
	// make output tensor
	auto valueOptions = torch::TensorOptions().dtype(localData.scalar_type()).layout(torch::kStrided).device(localData.device().type(), localData.device().index());
	std::vector<int64_t> globalShape;
	globalShape.push_back(batchSize); // batch
	globalShape.push_back(dataChannels);
	for(index_t dim=dims-1;dim>=0;--dim){
		globalShape.push_back(t_globalShape.data_ptr<index_t>()[dim]); //
	}
	torch::Tensor globalData = torch::zeros(globalShape, valueOptions);
	
	std::vector<int64_t> normShape;
	normShape.push_back(1); // batch
	normShape.push_back(1);
	for(index_t dim=dims-1;dim>=0;--dim){
		normShape.push_back(t_globalShape.data_ptr<index_t>()[dim]); //
	}
	torch::Tensor weightNorm = torch::zeros(normShape, valueOptions);
	
	const GridInfo globalGrid = MakeGridInfo(globalData.size(-1), dims>1?globalData.size(-2):1, dims>2?globalData.size(-3):1, globalData.size(1));
	const GridInfo localGrid = MakeGridInfo(localCoords.size(-1), dims>1?localCoords.size(-2):1, dims>2?localCoords.size(-3):1, localCoords.size(1));
	
	DISPATCH_FTYPES_DIMS(localData.scalar_type(), dims, "SampleTransformedGridLocalToGlobal",
		_SampleTransformedGridLocalToGlobal<scalar_t, dim>(
			localData.data_ptr<scalar_t>(), localCoords.data_ptr<scalar_t>(), localGrid,
			globalData, weightNorm.data_ptr<scalar_t>(), globalTransform.data_ptr<scalar_t>(), globalGrid,
			dataChannels, fillMaxSteps
		);
	);
	
	return {globalData, weightNorm};
}

template<typename scalar_t, int DIMS>
__host__ 
void PrintMatrix(const MatrixSquare<scalar_t, DIMS> m){
	std::cout << "Matrix:" << std::endl;
	std::cout << std::fixed << std::setprecision(4) << std::setfill(' ');
	for(index_t row=0;row<DIMS;++row){
		for(index_t col=0; col<DIMS; ++col){ 
			std::cout << std::setw(7) << m.a[row][col] << ' ';
		}
		std::cout << std::endl;
	}
}

template<typename scalar_t, int DIMS>
void _SampleTransformedGridLocalToGlobalMulti(const std::vector<torch::Tensor> &localDataList, const std::vector<torch::Tensor> &localCoordsList,
										 torch::Tensor &globalData, scalar_t *d_weightNorm, const scalar_t *h_globalTransform, const GridInfo &globalGrid,
										 const index_t channels, const index_t fillMaxSteps){
	
	// make transform from tensor to matrix struct
	MatrixSquare<scalar_t, DIMS+1> t;
	for(index_t row=0;row<DIMS+1;++row){
		for(index_t col=0; col<DIMS+1; ++col){
			t.a[row][col] = h_globalTransform[row*(DIMS+1) + col];
		}
	}
	// get inverse transform
	MatrixSquare<scalar_t, DIMS+1> tInv = inverse(t);

	//PrintMatrix(t);
	//PrintMatrix(tInv);
	
	// kernel over all points in localData
	int minGridSize = 0, blockSize = 0, gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_SampleTransformedGridLocalToGlobal<scalar_t, DIMS>, 0, 0);
	
	// scatter all grids to the global grid. The normalization should blend overlaps.
	for(index_t i=0; i<localDataList.size(); ++i){
		const GridInfo localGrid = MakeGridInfo(localCoordsList[i].size(-1), DIMS>1?localCoordsList[i].size(-2):1, DIMS>2?localCoordsList[i].size(-3):1, localCoordsList[i].size(1));
		const scalar_t *d_localData = localDataList[i].data_ptr<scalar_t>();
		const scalar_t *d_localCoords = localCoordsList[i].data_ptr<scalar_t>();
		
		gridSize = (localGrid.stride.w + blockSize - 1) / blockSize;
		
		k_SampleTransformedGridLocalToGlobal<scalar_t, DIMS><<<gridSize, blockSize>>>(
			d_localData, d_localCoords, localGrid,
			globalData.data_ptr<scalar_t>(), d_weightNorm, t, tInv, globalGrid,
			channels);
	}
	
	// kernel over all points in globalData
	minGridSize = 0, blockSize = 0, gridSize = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_NormScatteredWithWeight<scalar_t>, 0, 0);
	gridSize = (globalGrid.stride.w + blockSize - 1) / blockSize;
	k_NormScatteredWithWeight<scalar_t><<<gridSize, blockSize>>>(
		globalData.data_ptr<scalar_t>(), d_weightNorm, globalGrid, channels
	);
	
	_FillEmptyCells<scalar_t, DIMS>(globalData, d_weightNorm, globalGrid, channels, fillMaxSteps);
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

std::vector<torch::Tensor> SampleTransformedGridLocalToGlobalMulti(const std::vector<torch::Tensor> &localDataList, const std::vector<torch::Tensor> &localCoordsList, const torch::Tensor &globalTransform, const torch::Tensor &t_globalShape, const index_t fillMaxSteps){
	// globalShape is x,y,z
	// input checks
	TORCH_CHECK(localDataList.size()==localCoordsList.size(), "Size of localData and localCoords must match.");
	const index_t numLocal = localDataList.size();
	TORCH_CHECK(numLocal>0, "Empty input list.")
	
	CHECK_INPUT_CUDA(localDataList[0]);
	TORCH_CHECK(2<localDataList[0].dim() && localDataList[0].dim()<6, "localData must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(localDataList[0].size(0)==1, "localData batch dimension must be 1.");
	const index_t batchSize = localDataList[0].size(0);
	const index_t dataChannels = localDataList[0].size(1);
	const index_t dims = localDataList[0].dim()-2;
	const auto dtype = localDataList[0].dtype();
	
	CHECK_INPUT_CUDA(localCoordsList[0]);
	TORCH_CHECK(localCoordsList[0].dim() == dims+2, "localCoords dimensionality must match localData.");
	TORCH_CHECK(localCoordsList[0].size(0)==batchSize, "localCoords batch dimension must be 1.");
	TORCH_CHECK(localCoordsList[0].size(1)==dims, "localCoords channel dimension must match spatial dimensionality.");
	for(index_t dim=0; dim<dims; ++dim){
		TORCH_CHECK(localCoordsList[0].size(dim+2)==localDataList[0].size(dim+2), "localCoords channel spatial dimensions must match localData.");
	}
	TORCH_CHECK(localCoordsList[0].dtype()==dtype, "localCoords must have matching data type.");
	
	for(index_t i=1; i<numLocal; ++i){
		const torch::Tensor localData = localDataList[i];
		const torch::Tensor localCoords = localCoordsList[i];
		
		CHECK_INPUT_CUDA(localData);
		TORCH_CHECK(localData.dim()==(dims+2), "Additional localData must have matching dimensionality.");
		TORCH_CHECK(localData.size(0)==batchSize, "Additional localData batch dimension must match.");
		TORCH_CHECK(localData.dtype()==dtype, "Additional localData must have matching data type.");
		
		CHECK_INPUT_CUDA(localCoords);
		TORCH_CHECK(localCoords.dim() == dims+2, "Additional localCoords dimensionality must match localData.");
		TORCH_CHECK(localCoords.size(0)==batchSize, "Additional localCoords batch dimension must be 1.");
		TORCH_CHECK(localCoords.size(1)==dims, "Additional localCoords channel dimension must match spatial dimensionality.");
		for(index_t dim=0; dim<dims; ++dim){
			TORCH_CHECK(localCoords.size(dim+2)==localData.size(dim+2), "Additional localCoords channel spatial dimensions must match additional localData.");
		}
		TORCH_CHECK(localCoords.dtype()==dtype, "Additional localCoords must have matching data type.");
	}
	
	// output checks
	CHECK_INPUT_HOST(globalTransform);
	TORCH_CHECK(globalTransform.dim() == 3, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(0)==batchSize, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(1)==dims+1, "globalTransform must have dimension 3D.");
	TORCH_CHECK(globalTransform.size(2)==dims+1, "globalTransform must have dimension 3D.");
	
	CHECK_INPUT_HOST(t_globalShape); // x,y,z
	TORCH_CHECK(t_globalShape.dim() == 1, "globalShape must have flat shape.");
	TORCH_CHECK(t_globalShape.size(0)==dims, "globalShape must match spatial dimensionality.");
	
	// make output tensor
	auto valueOptions = torch::TensorOptions().dtype(localDataList[0].scalar_type()).layout(torch::kStrided).device(localDataList[0].device().type(), localDataList[0].device().index());
	std::vector<int64_t> globalShape;
	globalShape.push_back(batchSize); // batch
	globalShape.push_back(dataChannels);
	for(index_t dim=dims-1;dim>=0;--dim){
		globalShape.push_back(t_globalShape.data_ptr<index_t>()[dim]); //
	}
	torch::Tensor globalData = torch::zeros(globalShape, valueOptions);
	
	std::vector<int64_t> normShape;
	normShape.push_back(1); // batch
	normShape.push_back(1);
	for(index_t dim=dims-1;dim>=0;--dim){
		normShape.push_back(t_globalShape.data_ptr<index_t>()[dim]); //
	}
	torch::Tensor weightNorm = torch::zeros(normShape, valueOptions);
	
	const GridInfo globalGrid = MakeGridInfo(globalData.size(-1), dims>1?globalData.size(-2):1, dims>2?globalData.size(-3):1, globalData.size(1));
	
	
	DISPATCH_FTYPES_DIMS(localDataList[0].scalar_type(), dims, "SampleTransformedGridLocalToGlobalMulti",
		_SampleTransformedGridLocalToGlobalMulti<scalar_t, dim>(
			localDataList, localCoordsList,
			globalData, weightNorm.data_ptr<scalar_t>(), globalTransform.data_ptr<scalar_t>(), globalGrid,
			dataChannels, fillMaxSteps
		);
	);
	
	return {globalData, weightNorm};
}