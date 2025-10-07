
#include "grid_gen.h"
#include "dispatch.h"

#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void k_MakeGrid2DNonUniformScale(const index_t sizeX, const index_t sizeY, TransformGPU<scalar_t,2> *transforms){
    const index_t totalSize = sizeX*sizeY;
    for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
        const I4 pos = {.a={flatIdx%sizeX, flatIdx/sizeX,0,0}}; //unflattenIndex(flatIdx);
        TransformGPU<scalar_t,2> *p_T = transforms + flatIdx;
        scalar_t posXnorm = static_cast<scalar_t>(pos.x)/static_cast<scalar_t>(sizeX);
        scalar_t scalX = 0.5 +posXnorm*posXnorm;
        //scalar_t posYnorm = static_cast<scalar_t>(pos.x)/static_cast<scalar_t>(sizeX);
        scalar_t scalY = 1;
        p_T->M.a[0][0] = scalX;
        p_T->M.a[1][1] = scalY;
        p_T->Minv.a[0][0] = 1/scalX;
        p_T->Minv.a[1][1] = 1/scalY;
        p_T->det = scalX*scalY;
    }
}

torch::Tensor MakeGrid2DNonUniformScale(const index_t sizeX, const index_t sizeY, const torch::Tensor &scaleStrength){
    CHECK_INPUT_CUDA(scaleStrength);
    auto valueOptions = torch::TensorOptions().dtype(scaleStrength.scalar_type()).layout(torch::kStrided).device(scaleStrength.device().type(), scaleStrength.device().index());
    torch::Tensor transforms = torch::zeros({sizeY, sizeX, 9}, valueOptions);
    const index_t totalSize = sizeX*sizeY;
    
	AT_DISPATCH_FLOATING_TYPES(scaleStrength.scalar_type(), "k_MakeGrid2DNonUniformScale", ([&] {
	    int minGridSize = 0, blockSize = 0, gridSize = 0;
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_MakeGrid2DNonUniformScale<scalar_t>, 0, 0);
	    gridSize = (totalSize + blockSize - 1) / blockSize;
		k_MakeGrid2DNonUniformScale<scalar_t><<<gridSize, blockSize>>>(
			sizeX, sizeY, reinterpret_cast<TransformGPU<scalar_t,2>*>(transforms.data_ptr<scalar_t>())
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}));
	
	return transforms;
}



template<typename scalar_t, int DIMS>
__global__ void k_MakeGridNDNonUniformScaleNormalized(const GridInfo grid, const scalar_t *scaleStrength, TransformGPU<scalar_t,DIMS> *transforms){
	//total size equals untransformed grid/resolution
	// scaling at borders matches for periodic boundaries
	// scaleStrength is the ratio between the largest and smallest cell
    const index_t totalSize = grid.stride.w;
    for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
        const I4 pos = unflattenIndex(flatIdx, grid);
        TransformGPU<scalar_t,DIMS> *p_T = transforms + flatIdx;
		
		scalar_t det = 1;
		for(index_t dim=0;dim<DIMS;++dim){
			//linear "hat", rising from border to center, discretely integrates to "size"; (assume size%2==0)
			const scalar_t s = max(min(scaleStrength[dim], static_cast<scalar_t>(1)), static_cast<scalar_t>(0));
			const scalar_t halfRes = static_cast<scalar_t>(grid.size.a[dim]) * 0.5;
			const scalar_t offset = 1 - s;
			const scalar_t x = -abs(static_cast<scalar_t>(pos.a[dim]) + 0.5 - halfRes) + halfRes;
			const scalar_t slope = 2*s/halfRes;
			scalar_t scale = slope*x+offset;
			if((grid.size.a[dim]%2)!=0){
				//correction if size is odd
				scale -= s/(grid.size.a[dim]*grid.size.a[dim]);
			}
			
			det *= scale;
			p_T->M.a[dim][dim] = scale;
			p_T->Minv.a[dim][dim] = 1/scale;
		}
        p_T->det = det;
    }
}

torch::Tensor MakeGridNDNonUniformScaleNormalized(const index_t sizeX, const index_t sizeY, const index_t sizeZ, const torch::Tensor &scaleStrength){
    CHECK_INPUT_CUDA(scaleStrength);
	TORCH_CHECK(scaleStrength.dim()==1, "scaleStrength must be a vector.");
	
	index_t dims = 1;
	if(sizeY>0){
		dims = 2;
		if(sizeZ>0){
			dims = 3;
		}
	}else if(sizeZ>0){
		TORCH_CHECK(sizeZ<=0, "sizeZ is not used and must be <=0 if sizeY<=0.");
	}
	TORCH_CHECK(scaleStrength.size(0)==dims, "Size of scaleStrength must match output dimensionality.");
	
	
	const GridInfo grid = MakeGridInfo(sizeX, dims>1?sizeY:1, dims>2?sizeZ:1);
	
    auto valueOptions = torch::TensorOptions().dtype(scaleStrength.scalar_type()).layout(torch::kStrided).device(scaleStrength.device().type(), scaleStrength.device().index());
	
	std::vector<int64_t> transformSize;
	for(index_t dim=dims-1;dim>=0;--dim){
		transformSize.push_back(grid.size.a[dim]); //
	}
	transformSize.push_back(dims*dims*2+1); //size of one transform struct
	
    torch::Tensor transforms = torch::zeros(transformSize, valueOptions);
    //const index_t totalSize = sizeX*sizeY; ==grid.strides.w
    
	DISPATCH_FTYPES_DIMS(scaleStrength.scalar_type(), dims, "MakeGridNDNonUniformScaleNormalized",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_MakeGridNDNonUniformScaleNormalized<scalar_t, dim>, 0, 0);
	    gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		k_MakeGridNDNonUniformScaleNormalized<scalar_t, dim><<<gridSize, blockSize>>>(
			grid, scaleStrength.data_ptr<scalar_t>(), reinterpret_cast<TransformGPU<scalar_t,dim>*>(transforms.data_ptr<scalar_t>())
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return transforms;
}


template<typename scalar_t, int DIMS>
__global__ void k_MakeGridNDExpScaleNormalized(const GridInfo grid, const scalar_t *scaleStrength, TransformGPU<scalar_t,DIMS> *transforms){
	//total size equals untransformed grid/resolution
	// scaling at borders matches for periodic boundaries
	// scaleStrength is the ratio between the largest and smallest cell
    const index_t totalSize = grid.stride.w;
    for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
        const I4 pos = unflattenIndex(flatIdx, grid);
        TransformGPU<scalar_t,DIMS> *p_T = transforms + flatIdx;
		
		scalar_t det = 1;
		for(index_t dim=0;dim<DIMS;++dim){
			//linear "hat", rising from border to center, discretely integrates to "size"; (assume size%2==0)
			const scalar_t s = max(scaleStrength[dim], static_cast<scalar_t>(1));
			scalar_t scale = 1;
			if(s>1){
				const index_t i = min(pos.a[dim], grid.size.a[dim] -1 -pos.a[dim]);
				const scalar_t halfRes = static_cast<scalar_t>(grid.size.a[dim]) * 0.5;
				const scalar_t sPowHalf = pow(s, halfRes);
				const scalar_t sPowPos = pow(s, i);
				scale = halfRes * (1-s) * sPowPos / (1 - sPowHalf);
			}
			
			
			det *= scale;
			p_T->M.a[dim][dim] = scale;
			p_T->Minv.a[dim][dim] = 1/scale;
		}
        p_T->det = det;
    }
}

torch::Tensor MakeGridNDExpScaleNormalized(const index_t sizeX, const index_t sizeY, const index_t sizeZ, const torch::Tensor &scaleStrength){
    CHECK_INPUT_CUDA(scaleStrength);
	TORCH_CHECK(scaleStrength.dim()==1, "scaleStrength must be a vector.");
	
	index_t dims = 1;
	if(sizeY>0){
		dims = 2;
		if(sizeZ>0){
			dims = 3;
		}
	}else if(sizeZ>0){
		TORCH_CHECK(sizeZ<=0, "sizeZ is not used and must be <=0 if sizeY<=0.");
	}
	TORCH_CHECK(scaleStrength.size(0)==dims, "Size of scaleStrength must match output dimensionality.");
	
	
	const GridInfo grid = MakeGridInfo(sizeX, dims>1?sizeY:1, dims>2?sizeZ:1);
	
    auto valueOptions = torch::TensorOptions().dtype(scaleStrength.scalar_type()).layout(torch::kStrided).device(scaleStrength.device().type(), scaleStrength.device().index());
	
	std::vector<int64_t> transformSize;
	for(index_t dim=dims-1;dim>=0;--dim){
		transformSize.push_back(grid.size.a[dim]); //
	}
	transformSize.push_back(dims*dims*2+1); //size of one transform struct
	
    torch::Tensor transforms = torch::zeros(transformSize, valueOptions);
    //const index_t totalSize = sizeX*sizeY; ==grid.strides.w
    
	DISPATCH_FTYPES_DIMS(scaleStrength.scalar_type(), dims, "MakeGridNDExpScaleNormalized",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_MakeGridNDExpScaleNormalized<scalar_t, dim>, 0, 0);
	    gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		k_MakeGridNDExpScaleNormalized<scalar_t, dim><<<gridSize, blockSize>>>(
			grid, scaleStrength.data_ptr<scalar_t>(), reinterpret_cast<TransformGPU<scalar_t,dim>*>(transforms.data_ptr<scalar_t>())
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return transforms;
}

__device__ constexpr index_t gaussSum(const index_t n){
	//return n*(n+1)/2;
	return (n*(n+1))>>1;
}

template<typename scalar_t, int DIMS>
__global__ void k_MakeCoordsNDNonUniformScaleNormalized(const GridInfo grid, const scalar_t *scaleStrength, scalar_t *p_coords){
	//total size equals untransformed grid/resolution
	// scaling at borders matches for periodic boundaries
	// scaleStrength is the ratio between the largest and smallest cell
    const index_t totalSize = grid.stride.w;
    for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < totalSize; flatIdx += blockDim.x * gridDim.x){
        const I4 pos = unflattenIndex(flatIdx, grid);
		
		scalar_t det = 1;
		for(index_t dim=0;dim<DIMS;++dim){
			//linear "hat", rising from border to center, discretely integrates to "size"; (assume size%2==0)
			const index_t gridRes = grid.size.a[dim]-1;
			const scalar_t s = max(min(scaleStrength[dim], static_cast<scalar_t>(1)), static_cast<scalar_t>(0));
			const scalar_t halfRes = static_cast<scalar_t>(gridRes) * 0.5;
			const scalar_t pos_s = static_cast<scalar_t>(pos.a[dim]);
			const scalar_t offset = pos_s*(1 - s - (gridRes%2)*s/(gridRes*gridRes));
			const scalar_t slope = 2*s/halfRes;
			//const scalar_t x = -abs(static_cast<scalar_t>(pos.a[dim]) + 0.5 - halfRes) + halfRes;
			const index_t numCellsHalf1 = min(pos.a[dim], gridRes/2); //pos_s<=halfRes ? pos.a[dim] : gridRes/2;
			scalar_t xsum = static_cast<scalar_t>(gaussSum(numCellsHalf1)) - static_cast<scalar_t>(numCellsHalf1)*0.5;

			const index_t numCellsHalf2 = pos.a[dim] - numCellsHalf1;//max(pos.a[dim] - gridRes/2, 0);
			xsum += static_cast<scalar_t>(numCellsHalf2)*(static_cast<scalar_t>(gridRes) + 0.5);
			//xsum += -static_cast<scalar_t>(gaussSum(pos.a[dim]) - gaussSum(gridRes/2)) + static_cast<scalar_t>(numVertHalf2)*(static_cast<scalar_t>(gridRes) - 0.5);
			if(0 < numCellsHalf2){
				xsum -= static_cast<scalar_t>(gaussSum(pos.a[dim]) - gaussSum(numCellsHalf1));
			}
			
			const scalar_t coord = slope*xsum+offset;

			I4 tempPos = pos;
			tempPos.w = dim;
			p_coords[flattenIndex(tempPos, grid)] = coord;
		}
    }
}

torch::Tensor MakeCoordsNDNonUniformScaleNormalized(const index_t sizeX, const index_t sizeY, const index_t sizeZ, const torch::Tensor &scaleStrength){
    CHECK_INPUT_CUDA(scaleStrength);
	TORCH_CHECK(scaleStrength.dim()==1, "scaleStrength must be a vector.");
	
	index_t dims = 1;
	if(sizeY>0){
		dims = 2;
		if(sizeZ>0){
			dims = 3;
		}
	}else if(sizeZ>0){
		TORCH_CHECK(sizeZ<=0, "sizeZ is not used and must be <=0 if sizeY<=0.");
	}
	TORCH_CHECK(scaleStrength.size(0)==dims, "Size of scaleStrength must match output dimensionality.");
	
	
	const GridInfo grid = MakeGridInfo(sizeX+1, dims>1?sizeY+1:1, dims>2?sizeZ+1:1);
	
    auto valueOptions = torch::TensorOptions().dtype(scaleStrength.scalar_type()).layout(torch::kStrided).device(scaleStrength.device().type(), scaleStrength.device().index());
	
	std::vector<int64_t> coordsSize;
	coordsSize.push_back(1); // batch
	coordsSize.push_back(dims);
	for(index_t dim=dims-1;dim>=0;--dim){
		coordsSize.push_back(grid.size.a[dim]); //
	}
	
    torch::Tensor coords = torch::zeros(coordsSize, valueOptions);
    //const index_t totalSize = sizeX*sizeY; ==grid.strides.w
    
	DISPATCH_FTYPES_DIMS(scaleStrength.scalar_type(), dims, "MakeGridNDNonUniformScaleNormalized",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_MakeCoordsNDNonUniformScaleNormalized<scalar_t, dim>, 0, 0);
	    gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		k_MakeCoordsNDNonUniformScaleNormalized<scalar_t, dim><<<gridSize, blockSize>>>(
			grid, scaleStrength.data_ptr<scalar_t>(), coords.data_ptr<scalar_t>()
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return coords;
}


template<typename scalar_t, int DIMS>
__global__ void k_CoordsToTransforms(const scalar_t *p_coords, const GridInfo coordsGrid, TransformGPU<scalar_t,DIMS> *p_transforms, const GridInfo transformsGrid){
    for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < transformsGrid.stride.w; flatIdx += blockDim.x * gridDim.x){
        const I4 pos = unflattenIndex(flatIdx, transformsGrid);
        TransformGPU<scalar_t,DIMS> *p_T = p_transforms + flatIdx;

		const index_t numVertices = 1<<DIMS; // vertices per cell

		const index_t numFaces = 2*DIMS;
		Vector<scalar_t, DIMS> faceCoords[numFaces] = {0}; //-x,+x,-y,+y,-z,+z
		
		for(index_t vertIdx=0;vertIdx<numVertices;++vertIdx){ // iterate vertices per cell
			// get vertex position/index in coordinates grid
			I4 vertPos = pos; // lower corner
			vertPos.x += vertIdx & 1;
			vertPos.y += (vertIdx & 2)!=0;
			vertPos.z += (vertIdx & 4)!=0;

			// load vertex
			Vector<scalar_t, DIMS> coord;
			for(index_t compIdx=0;compIdx<DIMS;++compIdx){
				vertPos.w = compIdx;
				const index_t vertFlatPos = flattenIndex(vertPos, coordsGrid);
				coord.a[compIdx] = p_coords[vertFlatPos];
			}
			
			// add vertex to all faces that share it
			for(index_t faceIdx=0;faceIdx<DIMS;++faceIdx){ // a vertex is used by DIMS faces
				// normal axis of face + is it upper or lower
				const index_t isUpper = (vertIdx & (1<<faceIdx))!=0;
				faceCoords[faceIdx*2 + isUpper] += coord;
			}
		}
		
		const index_t verticesPerFace = 1<<(DIMS-1); // == numVertices/2
		const scalar_t faceNorm = static_cast<scalar_t>(1.0)/static_cast<scalar_t>(verticesPerFace);
		for(index_t faceIdx=0;faceIdx<numFaces;++faceIdx){
			faceCoords[faceIdx] *= faceNorm;
		}


		MatrixSquare<scalar_t,DIMS> t;
		for(index_t i=0; i<DIMS ;++i){
			for(index_t k=0; k<DIMS ;++k){
				//i: coordiante component
				//k: direction
				t.a[i][k] = faceCoords[k*2+1].a[i] - faceCoords[k*2].a[i];
			}
		}
		p_T->M = t;

		const scalar_t determinant = det(t);
		const MatrixSquare<scalar_t,DIMS> t_inv = inverse(t, determinant);
		p_T->Minv = t_inv;
		p_T->det = determinant;
	}
}

torch::Tensor CoordsToTransforms(const torch::Tensor &coords){
    CHECK_INPUT_CUDA(coords);
	TORCH_CHECK(2<coords.dim() && coords.dim()<6, "coords must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(coords.size(0)==1, "coords batch dimension must be 1.");
	index_t dims = coords.dim()-2;
	TORCH_CHECK(coords.size(1)==dims, "coords channel dimension must match spatial dimensionality.");
	
	
	const GridInfo coordsGrid = MakeGridInfo(coords.size(-1), dims>1?coords.size(-2):1, dims>2?coords.size(-3):1, coords.size(1));
	const GridInfo transformsGrid = MakeGridInfo(coords.size(-1)-1, dims>1?coords.size(-2)-1:1, dims>2?coords.size(-3)-1:1, coords.size(1));
	
    auto valueOptions = torch::TensorOptions().dtype(coords.scalar_type()).layout(torch::kStrided).device(coords.device().type(), coords.device().index());
	
	std::vector<int64_t> transformSize;
	transformSize.push_back(1); // batch size
	for(index_t dim=dims-1;dim>=0;--dim){
		transformSize.push_back(transformsGrid.size.a[dim]); //
	}
	transformSize.push_back(dims*dims*2+1); //size of one transform struct
	
    torch::Tensor transforms = torch::zeros(transformSize, valueOptions);
    //const index_t totalSize = sizeX*sizeY; ==grid.strides.w
    
	DISPATCH_FTYPES_DIMS(coords.scalar_type(), dims, "CoordsToTransforms",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_CoordsToTransforms<scalar_t, dim>, 0, 0);
	    gridSize = (transformsGrid.stride.w + blockSize - 1) / blockSize;
		k_CoordsToTransforms<scalar_t, dim><<<gridSize, blockSize>>>(
			coords.data_ptr<scalar_t>(), coordsGrid, reinterpret_cast<TransformGPU<scalar_t,dim>*>(transforms.data_ptr<scalar_t>()), transformsGrid
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return transforms;
}

/**
 * Compute the transformation metrics directly at the cell faces
 * transforms now have a staggered grid layout NCDHWT
 * Each cell/thread handles its lower faces, if valid
 */
template<typename scalar_t, int DIMS>
__global__ void k_CoordsToFaceTransforms(const scalar_t *p_coords, TransformGPU<scalar_t,DIMS> *p_transforms, const GridInfo grid){
	
	const index_t numFaceVertices = 1<<(DIMS-1);
	const index_t numEdgeVertices = 1<<(DIMS-2);
    
	for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < grid.stride.w; flatIdx += blockDim.x * gridDim.x){
        const I4 pos = unflattenIndex(flatIdx, grid);
		for(index_t dim=0; dim<DIMS; ++dim){
			// staggered grid upper cut-off
			for(index_t i=1; i<DIMS; ++i){
				const index_t tDim = (dim+i)%DIMS;
				if(pos.a[tDim]==grid.size.a[tDim]-1){
					goto dimLoopEnd;
				}
			}

			{ // if no staggered grid upper cut-off
				MatrixSquare<scalar_t,DIMS> t = {.a={0}};
				// face normal direction, center coords of upper and lower opposite faces
				// extrapolated if necessary/at boundary -> linear extrapolation == one-sided differencing
				// accumulate upper and lower face vertex coordiantes with correct sign, then normalize
				{
					//Vector<scalar_t, DIMS> grad = {.a={0}}; 
					//Vector<scalar_t, DIMS> &grad = t.v[dim]; // this leads to a transposed result
					//grad = {.a={0}};
					scalar_t normWeight = 0.5;

					for(index_t isUpper=0; isUpper<2; ++isUpper){
						const index_t faceSign = isUpper*2-1;
						const bool atBound = isUpper ? pos.a[dim] == grid.size.a[dim]-1 : pos.a[dim]==0;
						I4 facePos = pos;
						if(!atBound) {
							facePos.a[dim] += faceSign;
						}else{
							// can only happen once due to minimum grid resolution requirements
							normWeight = 1.0;
						}

						for(index_t vertIdx=0;vertIdx<numFaceVertices;++vertIdx){
							I4 vertPos = facePos;
							for(index_t i=1; i<DIMS; ++i){
								vertPos.a[(dim+i)%DIMS] += (vertIdx & i)!=0;
							}
							for(index_t compIdx=0;compIdx<DIMS;++compIdx){
								vertPos.w = compIdx;
								const index_t vertFlatPos = flattenIndex(vertPos, grid);
								t.a[compIdx][dim] += p_coords[vertFlatPos] * faceSign;
							}
						}
					}
					normWeight /= static_cast<scalar_t>(numFaceVertices);
					for(index_t compIdx=0;compIdx<DIMS;++compIdx){
						t.a[compIdx][dim] *= normWeight;
					}
				}

				// face tangential directions, center coords of edges (cell corners in 2D)
				// always available
				const scalar_t tNormWeight = static_cast<scalar_t>(1.0)/static_cast<scalar_t>(numEdgeVertices);
				for(index_t k=1; k<DIMS; ++k){
					const index_t tDim = (dim+k)%DIMS;
					//Vector<scalar_t, DIMS> &grad = t.v[tDim]; 
					//grad = {.a={0}};
					for(index_t vertIdx=0;vertIdx<numFaceVertices;++vertIdx){
						I4 vertPos = pos;
						for(index_t i=1; i<DIMS; ++i){
							vertPos.a[(dim+i)%DIMS] += (vertIdx & i)!=0;
						}
						const bool tIsUpper = vertIdx & k;
						const index_t tFaceSign = tIsUpper ? 1 : -1;
						for(index_t compIdx=0;compIdx<DIMS;++compIdx){
							vertPos.w = compIdx;
							const index_t vertFlatPos = flattenIndex(vertPos, grid);
							t.a[compIdx][tDim] += p_coords[vertFlatPos] * tFaceSign;
						}
					}
					for(index_t compIdx=0;compIdx<DIMS;++compIdx){
						t.a[compIdx][tDim] *= tNormWeight;
					}
				}

				// write transform
				I4 matPos = pos;
				matPos.w = dim;
				TransformGPU<scalar_t,DIMS> *p_T = p_transforms + flattenIndex(matPos, grid);
				p_T->M = t;

				const scalar_t determinant = det(t);
				const MatrixSquare<scalar_t,DIMS> t_inv = inverse(t, determinant);
				p_T->Minv = t_inv;
				p_T->det = determinant;
			}
			
			dimLoopEnd:
		}
	}
}

torch::Tensor CoordsToFaceTransforms(const torch::Tensor &coords){
    CHECK_INPUT_CUDA(coords);
	TORCH_CHECK(2<coords.dim() && coords.dim()<6, "coords must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(coords.size(0)==1, "coords batch dimension must be 1.");
	index_t dims = coords.dim()-2;
	TORCH_CHECK(coords.size(1)==dims, "coords channel dimension must match spatial dimensionality.");
	
	
	const GridInfo grid = MakeGridInfo(coords.size(-1), dims>1?coords.size(-2):1, dims>2?coords.size(-3):1, coords.size(1));
	
    auto valueOptions = torch::TensorOptions().dtype(coords.scalar_type()).layout(torch::kStrided).device(coords.device().type(), coords.device().index());
	
	std::vector<int64_t> transformSize;
	transformSize.push_back(1); // batch size
	transformSize.push_back(dims); // staggered grid
	for(index_t dim=dims-1;dim>=0;--dim){
		transformSize.push_back(grid.size.a[dim]); //
	}
	transformSize.push_back(dims*dims*2+1); //size of one transform struct
	
    torch::Tensor transforms = torch::zeros(transformSize, valueOptions);
    //const index_t totalSize = sizeX*sizeY; ==grid.strides.w
    
	DISPATCH_FTYPES_DIMS(coords.scalar_type(), dims, "CoordsToFaceTransforms",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_CoordsToFaceTransforms<scalar_t, dim>, 0, 0);
	    gridSize = (grid.stride.w + blockSize - 1) / blockSize;
		k_CoordsToFaceTransforms<scalar_t, dim><<<gridSize, blockSize>>>(
			coords.data_ptr<scalar_t>(), reinterpret_cast<TransformGPU<scalar_t,dim>*>(transforms.data_ptr<scalar_t>()), grid
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return transforms;
}