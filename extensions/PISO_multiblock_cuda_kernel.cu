
#include "PISO_multiblock_cuda.h"

#include "bicgstab_solver.h"

#include <cuda.h>
#include <cuda_runtime.h>



//#include <vector>
#include <limits>


#define LOGGING
#ifdef LOGGING
//#define PROFILING
#endif
#include "logging.h"

const size_t WARP_SIZE = 32;
const size_t MAX_BLOCK_SIZE = 512; //1024;


static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(10);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

//const int WARP_SIZE = 32;

__device__ constexpr
int divCeil(const int a, const int b){
	return (a + b - 1)/b;
}

template<typename I, typename M>
__device__ constexpr
I posMod(const I a, const M m){
	I mod = a % m;
	if(mod<0){
		mod += m;
	}
	return mod;
}

void CopyToGPU(void *p_dst, const void *p_src, const size_t bytes){
	//copy to GPU
	CUDA_CHECK_RETURN(cudaMemcpy(p_dst, p_src, bytes, cudaMemcpyHostToDevice)); //dst, src, bytes, kind
}
void CopyDomainToGPU(const DomainAtlasSet &domainAtlas){
	//copy atlas to GPU
	CUDA_CHECK_RETURN(cudaMemcpy(domainAtlas.p_device, domainAtlas.p_host, domainAtlas.sizeBytes, cudaMemcpyHostToDevice)); //dst, src, bytes, kind
}

__host__ void ComputeThreadBlocks(std::shared_ptr<Domain> domain, int32_t &threads, dim3 &blocks, std::vector<index_t> &blockIdxByThreadBlock, std::vector<index_t> &threadBlockOffsetInBlock) {
	index_t numThreadBlocks = 0;
	const index_t maxBlockSize = domain->getMaxBlockSize();
	const index_t threadBlockSize = maxBlockSize>=MAX_BLOCK_SIZE ? MAX_BLOCK_SIZE : ALIGN_UP(maxBlockSize, WARP_SIZE);
	index_t blockIdx = 0;
	blockIdxByThreadBlock.clear();
	threadBlockOffsetInBlock.clear();
	for(std::shared_ptr<const Block> block : domain->blocks){
		const index_t blockSize = block->getStrides().w;
		const index_t threadBlocksPerBlock = divCeil(blockSize, threadBlockSize);
		//py::print("block", blockIdx, ", size", blockSize, ", blocks", threadBlocksPerBlock);
		numThreadBlocks += threadBlocksPerBlock;
		for(index_t i=0; i<threadBlocksPerBlock; ++i){
			blockIdxByThreadBlock.push_back(blockIdx);
			threadBlockOffsetInBlock.push_back(i);
			//py::print("push", blockIdx, i);
		}
		++blockIdx;
	}

	threads = threadBlockSize;
	blocks = dim3(numThreadBlocks,1);
}

__host__ torch::Tensor CopyBlockIndices(std::shared_ptr<Domain> domain,
		std::vector<index_t> &blockIdxByThreadBlock, std::vector<index_t> &threadBlockOffsetInBlock,
		index_t *&p_blockIdxByThreadBlock, index_t *&p_threadBlockOffsetInBlock){
	auto byteOptions = torch::TensorOptions().dtype(torch_kIndex).layout(torch::kStrided).device(domain->getDevice().type(), domain->getDevice().index());
	torch::Tensor allocTensor = torch::zeros(blockIdxByThreadBlock.size() + threadBlockOffsetInBlock.size(), byteOptions);
	p_blockIdxByThreadBlock = allocTensor.data_ptr<index_t>();
	p_threadBlockOffsetInBlock = p_blockIdxByThreadBlock + blockIdxByThreadBlock.size();
	CUDA_CHECK_RETURN(cudaMemcpy(p_blockIdxByThreadBlock, blockIdxByThreadBlock.data(), blockIdxByThreadBlock.size()*sizeof(index_t), cudaMemcpyHostToDevice)); //dst, src, bytes, kind
	CUDA_CHECK_RETURN(cudaMemcpy(p_threadBlockOffsetInBlock, threadBlockOffsetInBlock.data(), threadBlockOffsetInBlock.size()*sizeof(index_t), cudaMemcpyHostToDevice)); //dst, src, bytes, kind
	return allocTensor;
}

__device__ index_t GetBlockIdxForThreadBlock(const index_t threadBlockIdx){
	//TODO
	//blockIdxByThreadBlock[threadBlockIdx];
	return 0;
}



#define SWITCH_DIMS_CASE(DIM, ...) \
	case DIM: { \
		const index_t dim = DIM; \
		__VA_ARGS__; \
		break; \
	}

#define SWITCH_DIMS_SWITCH(DIMS, ...) \
	switch(DIMS) { \
		__VA_ARGS__ \
		default: \
			TORCH_CHECK(false, "Unknown dimension."); \
	}

#define SWITCH_DIMS(DIM, ...) \
	SWITCH_DIMS_SWITCH(DIM, SWITCH_DIMS_CASE(1, __VA_ARGS__) SWITCH_DIMS_CASE(2, __VA_ARGS__) SWITCH_DIMS_CASE(3, __VA_ARGS__))

#define DISPATCH_FTYPES_DIMS(DOMAIN, NAME, ...) \
	AT_DISPATCH_FLOATING_TYPES(DOMAIN->getDtype(), NAME, ([&] { \
		SWITCH_DIMS(DOMAIN->getSpatialDims(), \
			__VA_ARGS__; \
		); \
	}));

template <typename scalar_t>
__device__ constexpr
S4<scalar_t> makeS4GPU(const scalar_t x = 0, const scalar_t y = 0, const scalar_t z = 0, const scalar_t w = 0){
	//return {{.x=x, .y=y, .z=z, .w=w}};
	return {{x, y, z, w}};
}
const auto makeI4GPU = makeS4GPU<int32_t>;
const auto makeU4GPU = makeS4GPU<dim_t>;
const auto makeF4GPU = makeS4GPU<float>;
const auto makeD4GPU = makeS4GPU<double>;

__device__ inline
index_t flattenIndex(const I4 &pos, const I4 &stride){
	return pos.x + stride.y*pos.y + stride.z*pos.z + stride.w*pos.w;
}

template<typename scalar_t>
__device__ inline
index_t flattenIndex(const I4 &pos, const BlockGPU<scalar_t> &block){
	//return pos.x + block.stride.y*pos.y + block.stride.z*pos.z + block.stride.w*pos.w;
	return flattenIndex(pos, block.stride);
}

template<typename scalar_t>
__device__ inline
index_t flattenIndex(const I4 &pos, const BlockGPU<scalar_t> *block){
	//return pos.x + block->stride.y*pos.y + block->stride.z*pos.z + block->stride.w*pos.w;
	return flattenIndex(pos, block->stride);
}

__device__ inline
I4 unflattenIndex(const index_t idx, const I4 &size, const I4 &stride){
	return makeI4GPU(idx%size.x, (idx/stride.y)%size.y, (idx/stride.z)%size.z, (idx/stride.w)%size.w);
} 

template<typename scalar_t>
__device__ inline
I4 unflattenIndex(const index_t idx, const BlockGPU<scalar_t> &block){
	//return makeI4GPU(idx%block.size.x, (idx/block.stride.y)%block.size.y, (idx/block.stride.z)%block.size.z, (idx/block.stride.w)%block.size.w);
	return unflattenIndex(idx, block.size, block.stride);
} 

template<typename scalar_t>
__device__ inline
index_t flattenIndexGlobal(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	const index_t dim = pos.w;
	I4 tempPos = pos;
	tempPos.w = 0;
	return flattenIndex(tempPos, block.stride) + block.globalOffset + domain.numCells*dim;
}

template<typename scalar_t>
__device__ inline
index_t flattenIndexGlobal(const I4 &pos, const BlockGPU<scalar_t> *block, const DomainGPU<scalar_t> &domain){
	const index_t dim = pos.w;
	I4 tempPos = pos;
	tempPos.w = 0;
	return flattenIndex(tempPos, block->stride) + block->globalOffset + domain.numCells*dim;
}

template<typename scalar_t>
 __host__ __device__ inline
 bool isEmptyBound(const index_t idx, const BoundaryGPU<scalar_t> *bounds){
	return
		bounds[idx].type==BoundaryType::DIRICHLET
		|| bounds[idx].type==BoundaryType::DIRICHLET_VARYING
		|| bounds[idx].type==BoundaryType::GRADIENT
		|| bounds[idx].type==BoundaryType::FIXED;
}

// boundary: [0,dims*2) = [-x,+x,-y,+y,-z,+z]
// lowest bit is the direction (0=lower, 1= upper), next 2 bits are the axis (00=x, 01=y, 10=z)
__host__ __device__ constexpr
index_t axisFromBound(const index_t bound){
	return bound>>1; // remove the direction bit
}
/** axis must be non-negative, isUpper 0 or 1. */
__host__ __device__ constexpr
index_t axisToBound(const index_t axis, const index_t isUpper){
	return (axis<<1) | isUpper;
}

__host__ __device__ constexpr
index_t boundIsUpper(const index_t bound){
	return bound&1; // check the direction bit
}

__host__ __device__ constexpr
index_t invertBound(const index_t bound){
	return bound^1; // flip the direction bit
}

__host__ __device__ constexpr
index_t faceSignFromBound(const index_t bound){
	return boundIsUpper(bound)*2 - 1;
}

/**
 * check if the cell of block "block" at position "pos" has a boundary (fixed or connected) in the direction "bound".
 */
template<typename scalar_t>
__device__ inline
bool isAtBound(const I4 &pos, const index_t bound, const BlockGPU<scalar_t> &block){
	const index_t axis = axisFromBound(bound);
	return boundIsUpper(bound) ? pos.a[axis]==(block.size.a[axis]-1) : pos.a[axis]==0;
}
template<typename scalar_t>
__device__ inline
bool isAtBound(const I4 &pos, const index_t bound, const BlockGPU<scalar_t> *p_block){
	const index_t axis = axisFromBound(bound);
	return boundIsUpper(bound) ? pos.a[axis]==(p_block->size.a[axis]-1) : pos.a[axis]==0;
}

/**
 * returns: positive a s.t. (otherAxis + a)%dims == axis
 */
__host__ __device__ constexpr
index_t getAxisRelativeToOther(const index_t axis, const index_t otherAxis, const index_t dims){
	return posMod(axis - otherAxis, dims);
}

struct RowMeta{
	int endOffset;
	int size;
};
template<typename scalar_t>
__device__ RowMeta getCSRMatrixRowEndOffsetFromBlockBoundaries3D(const int flatPos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	const I4 pos = unflattenIndex(flatPos, block);
		
	int rowSize = 2*domain.numDims+1;
	int rowEndOffset=(flatPos+1)*rowSize;
	
	//subtract for open/closed bounds
	// number of cells with boundary at ? before and including current cell:
	//X
	// -x
	if(isEmptyBound(0,block.boundaries)){
		rowEndOffset -= (flatPos/block.size.x)+1;
		if(pos.x==0){
			--rowSize;
		}
	}
	// +x
	if(isEmptyBound(1,block.boundaries)){
		rowEndOffset -= (flatPos + 1)/block.size.x;
		if(pos.x==block.size.x-1){
			--rowSize;
		}
	}
	//Y
	if(domain.numDims>1){
		// -y
		if(isEmptyBound(2,block.boundaries)){
			rowEndOffset -= block.size.x*pos.z //previous slices
				+ (pos.y==0 ? pos.x+1 : block.size.x); // current slice
			if(pos.y==0){
				--rowSize;
			}
		}
		// +y
		if(isEmptyBound(3,block.boundaries)){
			rowEndOffset -= block.size.x*pos.z //previous slices
				+ (pos.y==(block.size.y-1) ? pos.x+1 : 0); // current slice
			if(pos.y==block.size.y-1){
				--rowSize;
			}
		}
	}
	//Z
	if(domain.numDims>2){
		// -z
		if(isEmptyBound(4,block.boundaries)){
			rowEndOffset -= (pos.z==0 ? flatPos+1 : block.stride.z); // flatPos=pos.x+pos.y*size.y, stride.z=size.x*size.y
			if(pos.z==0){
				--rowSize;
			}
		}
		// +z
		if(isEmptyBound(5,block.boundaries)){
			rowEndOffset -= (pos.z==(block.size.z-1) ? flatPos+1 - (pos.z*block.stride.z): 0); // stride.z=size.x*size.y
			if(pos.z==block.size.z-1){
				--rowSize;
			}
		}
	}
	return {rowEndOffset, rowSize};
}

template<typename scalar_t>
__device__ inline
index_t computeConnectedDir(const index_t dir, const index_t boundaryDim, const ConnectedBoundaryGPU<scalar_t> *p_cb, const DomainGPU<scalar_t> &domain){
	
	const index_t dirAxis = axisFromBound(dir);
	const index_t dirRelativeToConnection = getAxisRelativeToOther(dirAxis, boundaryDim, domain.numDims);
	
	// p_cb->axes uses the same format as face indexing, but the isUpper-bit means that the connection is inverted.
	return p_cb->axes.a[dirRelativeToConnection] ^ boundIsUpper(dir);
}

template<typename scalar_t>
__device__ inline
I4 computeConnectedPos(const I4 pos, const index_t boundaryDim, const ConnectedBoundaryGPU<scalar_t> *p_cb, const DomainGPU<scalar_t> &domain, const index_t borderOffset=0){
	const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_cb->connectedGridIndex;
	I4 connectedPos = pos; //sets w
	
	
	index_t connectedAxis = p_cb->axes.a[0]>>1;
	//connectedPos.w = connectedAxis;
	connectedPos.a[connectedAxis] = (p_cb->axes.a[0] & 1) ? p_connectedBlock->size.a[connectedAxis] - 1 -borderOffset: borderOffset;
	if(domain.numDims>1){
		index_t axis = (boundaryDim+1)%domain.numDims;
		connectedAxis = p_cb->axes.a[1]>>1;
		connectedPos.a[connectedAxis] = (p_cb->axes.a[1] & 1) ? p_connectedBlock->size.a[connectedAxis]-1 - pos.a[axis] : pos.a[axis];
		if(domain.numDims>2){
			axis = (boundaryDim+2)%domain.numDims;
			connectedAxis = p_cb->axes.a[2]>>1;
			connectedPos.a[connectedAxis] = (p_cb->axes.a[2] & 1) ? p_connectedBlock->size.a[connectedAxis]-1 - pos.a[axis] : pos.a[axis];
		}
	}
	return connectedPos;
}


template<typename scalar_t>
__device__ inline I4 computeConnectedPosWithChannel(const I4 pos, const index_t boundaryDim, const ConnectedBoundaryGPU<scalar_t> *p_cb, const DomainGPU<scalar_t> &domain, const index_t borderOffset=0){
	// computeConnectedPos() does not change pos.w
	I4 connectedPos = computeConnectedPos(pos, boundaryDim, p_cb, domain, borderOffset);
	// the requested component is not necessarily the boundary axis here
	// boundaryAxis = bound>>1; connected to axes[0]
	// requestedAxis = pos.w; connected to axes[?]
	// pos.W == bA -> axes[0], pos.w == (bA+1)%dims -> axes[1], pos.w == (bA+2)%dims -> axes[2]
	const index_t connectionIndex = posMod(pos.w-boundaryDim, domain.numDims); //needs positive mod
	connectedPos.w = p_cb->axes.a[connectionIndex]>>1;
	return connectedPos;
}

template<typename scalar_t>
__device__ inline bool isGlobalIndexInBlock(const index_t idx, const BlockGPU<scalar_t> *p_block){
	const index_t blockGlobalIndexStart = p_block->globalOffset;
	const index_t blockGlobalIndexEnd = blockGlobalIndexStart + p_block->stride.w;
	return (blockGlobalIndexStart<=idx && idx<blockGlobalIndexEnd);
}
template<typename scalar_t>
__device__ inline index_t getBlockIndexFromGlobalIndex(const index_t idx, const DomainGPU<scalar_t> &domain){
	for(index_t blockIdx=0; blockIdx<domain.numBlocks; ++blockIdx){
		const BlockGPU<scalar_t> *p_block = domain.blocks + blockIdx;
		if(isGlobalIndexInBlock(idx, p_block)){
			return blockIdx;
		}
	}
	return -1;
}

template<typename scalar_t>
__device__ inline index_t getBoundaryIndexToBlock(const index_t otherBlockIdx, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		if(block.boundaries[bound].type==BoundaryType::CONNECTED_GRID && block.boundaries[bound].cb.connectedGridIndex==otherBlockIdx){
			return bound;
		}
	}
	return -1;
}


template<typename scalar_t>
struct CellInfo{
	I4 pos;
	bool isBlock;
	union{
		//struct blockInfo{
			const BlockGPU<scalar_t> *p_block;
		//};
		//struct boundInfo{
			const FixedBoundaryGPU<scalar_t> *p_bound;
		//};
	};
};


template<typename scalar_t>
struct NeighborCellInfo{
	CellInfo<scalar_t> cell;
	Vector<index_t, 3> axisMapping;
};

template<typename scalar_t>
__device__ inline
void initDefaultAxisMapping(NeighborCellInfo<scalar_t> &info){
	info.axisMapping.a[0] = 0;
	info.axisMapping.a[1] = 2;
	info.axisMapping.a[2] = 4;
}

template<typename scalar_t>
__device__ inline
index_t flattenIndex(const CellInfo<scalar_t> &info){
	return info.isBlock ? flattenIndex(info.pos, info.p_block->stride) : flattenIndex(info.pos, info.p_bound->stride);
}
template<typename scalar_t>
__device__ inline
index_t flattenIndexGlobal(const CellInfo<scalar_t> &info, const DomainGPU<scalar_t> &domain){
	return info.isBlock ? flattenIndexGlobal(info.pos, info.p_block, domain) : 0;
}

template<typename scalar_t>
__device__ inline
NeighborCellInfo<scalar_t> resolveNeighborCell(const I4 &pos, const index_t dir, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain){
	
	NeighborCellInfo<scalar_t> info;
	initDefaultAxisMapping(info);
	const index_t axis = axisFromBound(dir);
	
	if(isAtBound(pos, dir, p_block)){
		switch(p_block->boundaries[dir].type){
			case BoundaryType::FIXED:
			{
				info.cell.isBlock = false;
				info.cell.pos = pos;
				info.cell.pos.a[axis] = 0;
				info.cell.p_bound = &(p_block->boundaries[dir].fb);
				break;
			}
			case BoundaryType::CONNECTED_GRID:
			{
				const ConnectedBoundaryGPU<scalar_t> *p_cb = &(p_block->boundaries[dir].cb);
				info.cell.isBlock = true;
				info.cell.p_block = domain.blocks + p_cb->connectedGridIndex;
				info.cell.pos = computeConnectedPos(pos, axis, p_cb, domain);
				for(index_t dim=0; dim<domain.numDims; ++dim){
					info.axisMapping.a[dim]=computeConnectedDir(info.axisMapping.a[dim], axis, p_cb, domain);
				}
				break;
			}
			case BoundaryType::PERIODIC:
			{
				info.cell.isBlock = true;
				info.cell.pos = pos;
				info.cell.pos.a[axis] = boundIsUpper(dir) ? 0 : p_block->size.a[axis]-1;
				info.cell.p_block = p_block;
				break;
			}
			default:
				break;
		}
	} else {
		info.cell.isBlock = true;
		info.cell.pos = pos;
		info.cell.pos.a[axis] += faceSignFromBound(dir);
		info.cell.p_block = p_block;
	}
	return info;
}


template <typename scalar_t, int DIMS>
__device__ scalar_t VelocityToContravariantComponent(const Vector<scalar_t,DIMS> &vel, const BlockGPU<scalar_t> *p_block, I4 pos){
	if(p_block->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		//load more globally as it is needed for all sides
		//const MatrixSquare<scalar_t, 3> mInv = T->Minv;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		//Minv[pos.w] is needed for 2 faces
		return det*dot(mInvRow, vel);
	} else {
		return vel.a[pos.w];
	}
}


template <typename scalar_t, int DIMS>
__device__ scalar_t VelocityToContravariantComponentBoundaryVarying(const Vector<scalar_t,DIMS> &vel, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound, I4 pos){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, vel);
	} else {
		return vel.a[pos.w];
	}
}
template <typename scalar_t, int DIMS>
__device__ scalar_t VelocityToContravariantComponentBoundaryFixed(const Vector<scalar_t,DIMS> &vel, const FixedBoundaryGPU<scalar_t> *p_bound, I4 pos){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, vel);
	} else {
		return vel.a[pos.w];
	}
}


#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> VelocityGradFromContravariantComponentGrad(const scalar_t &velGrad, const BlockGPU<scalar_t> *p_block, I4 pos){
	if(p_block->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		
		return mInvRow*(det*velGrad);
	} else {
		Vector<scalar_t,DIMS> grad = {.a={0}};
		grad.a[pos.w] = velGrad;
		return grad;
	}
}

template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> VelocityGradFromContravariantComponentGradBoundaryFixed(const scalar_t &velGrad, const FixedBoundaryGPU<scalar_t> *p_bound, I4 pos){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		
		return mInvRow*(det*velGrad);
	} else {
		Vector<scalar_t,DIMS> grad = {.a={0}};
		grad.a[pos.w] = velGrad;
		return grad;
	}
}
#endif //WITH_GRAD


template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> getVelocityFromBlock(I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	Vector<scalar_t,DIMS> vel = {.a={0}};
	//pos.w = 0;
	//const index_t blockStride = p_block->stride.w;
	//const index_t blockOffset = p_block->globalOffset;
	//index_t flatPos = flattenIndex(pos, p_block->stride);
	for(index_t dim=0; dim<DIMS; ++dim){
		pos.w = dim;
		if(velocityGlobal==nullptr){
			vel.a[dim] = p_block->velocity[flattenIndex(pos, p_block)];
		}else{
			vel.a[dim] = velocityGlobal[flattenIndexGlobal(pos, p_block, domain)];
		}
		//flatPos += blockStride;
	}
	return vel;
}


template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> getVelocityFromBoundaryVarying(I4 pos, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound){
	Vector<scalar_t,DIMS> vel = {.a={0}};
	for(index_t dim=0; dim<DIMS; ++dim){
		pos.w = dim;
		vel.a[dim] = p_bound->velocity[flattenIndex(pos, p_bound->stride)];
	}
	return vel;
}
template <typename scalar_t, int DIMS>
__device__ Vector<scalar_t,DIMS> getVelocityFromBoundaryFixed(I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound){
	const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->velocity);
	Vector<scalar_t,DIMS> vel = {.a={0}};
	if(p_data->boundaryType==BoundaryConditionType::DIRICHLET){
		if(p_data->isStatic){
			for(index_t dim=0; dim<DIMS; ++dim){
				pos.w = dim;
				vel.a[dim] = p_data->data[pos.w];
			}
		} else {
			for(index_t dim=0; dim<DIMS; ++dim){
				pos.w = dim;
				vel.a[dim] = p_data->data[flattenIndex(pos, p_bound->stride)];
			}
		}
	}
	// else TODO
	return vel;
}


#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__device__ void scatterVelocityGradToGrid(const Vector<scalar_t,DIMS> &velGrad, I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, scalar_t *velocityGlobalGrad){
	for(index_t dim=0; dim<DIMS; ++dim){
		pos.w = dim;
		if(velocityGlobalGrad==nullptr){
			atomicAdd(p_block->velocity_grad + flattenIndex(pos, p_block), velGrad.a[dim]);
		}else{
			atomicAdd(velocityGlobalGrad + flattenIndexGlobal(pos, p_block, domain), velGrad.a[dim]);
		}
	}
}

template <typename scalar_t, int DIMS>
__device__ void scatterVelocityGradToGridBoundaryFixed(const Vector<scalar_t,DIMS> &velGrad, I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound){
	for(index_t dim=0; dim<DIMS; ++dim){
		const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->velocity);
		if(p_data->isStatic){
			atomicAdd(p_data->grad + dim, velGrad.a[dim]);
		} else {
			pos.w = dim;
			atomicAdd(p_data->grad + flattenIndex(pos, p_bound->stride), velGrad.a[dim]);
		}
	}
}
#endif //WITH_GRAD


template <typename scalar_t, int DIMS>
__device__ inline scalar_t getContravariantComponent(const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	if(p_block->hasTransform){
		Vector<scalar_t,DIMS> vel = getVelocityFromBlock<scalar_t, DIMS>(pos, p_block, domain, velocityGlobal);
		return VelocityToContravariantComponent<scalar_t, DIMS>(vel, p_block, pos);
	} else {
		if(velocityGlobal==nullptr){
			return p_block->velocity[flattenIndex(pos, p_block)];
		}else{
			return velocityGlobal[flattenIndexGlobal(pos, p_block, domain)];
		}
	}
}


template <typename scalar_t, int DIMS>
__device__ inline scalar_t getContravariantComponentBoundaryVarying(const I4 pos, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound){
	if(p_bound->hasTransform){
		Vector<scalar_t,DIMS> vel = getVelocityFromBoundaryVarying<scalar_t, DIMS>(pos, p_bound);
		return VelocityToContravariantComponentBoundaryVarying<scalar_t, DIMS>(vel, p_bound, pos);
	} else {
		return p_bound->velocity[flattenIndex(pos, p_bound->stride)];
	}
}

template <typename scalar_t, int DIMS>
__device__ inline scalar_t getContravariantComponentBoundaryFixed(const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound){
	if(p_bound->hasTransform){
		Vector<scalar_t,DIMS> vel = getVelocityFromBoundaryFixed<scalar_t, DIMS>(pos, p_bound);
		return VelocityToContravariantComponentBoundaryFixed<scalar_t, DIMS>(vel, p_bound, pos);
	} else {
		//return getFixedBoundaryData();//p_bound->velocity[flattenIndex(pos, p_bound->stride)];
		const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->velocity);
		if(p_data->isStatic){
			return p_data->data[pos.w];
		} else {
			//const index_t dim = axisFromBound(bound);
			//I4 boundPos = pos;
			//if(isScalarDataType(type)) { boundPos.w = 0; }
			//boundPos.a[dim] = 0;
			const index_t flatBoundPos = flattenIndex(pos, p_bound->stride);
			return p_data->data[flatBoundPos];
		}
	}
}

#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__device__ void scatterContravariantComponentGrad(const scalar_t &velGrad, const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, scalar_t *velocityGlobalGrad){
	if(p_block->hasTransform){
		Vector<scalar_t,DIMS> grad = VelocityGradFromContravariantComponentGrad<scalar_t, DIMS>(velGrad, p_block, pos);
		scatterVelocityGradToGrid<scalar_t, DIMS>(grad, pos, p_block, domain, velocityGlobalGrad);
	} else {
		if(velocityGlobalGrad==nullptr){
			atomicAdd(p_block->velocity_grad + flattenIndex(pos, p_block), velGrad);
		}else{
			atomicAdd(velocityGlobalGrad + flattenIndexGlobal(pos, p_block, domain), velGrad);
		}
	}
}

template <typename scalar_t, int DIMS>
__device__ void scatterContravariantComponentGradBoundaryFixed(const scalar_t &velGrad, const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound){
	if(p_bound->hasTransform){
		Vector<scalar_t,DIMS> grad = VelocityGradFromContravariantComponentGradBoundaryFixed<scalar_t, DIMS>(velGrad, p_bound, pos);
		scatterVelocityGradToGridBoundaryFixed<scalar_t, DIMS>(grad, pos, p_bound);
	} else {
		const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->velocity);
		if(p_data->isStatic){
			atomicAdd(p_data->grad + pos.w, velGrad);
		}else{
			const index_t flatBoundPos = flattenIndex(pos, p_bound->stride);
			atomicAdd(p_data->grad + flatBoundPos, velGrad);
		}
	}
}
#endif //WITH_GRAD


template <typename scalar_t>
__device__ inline scalar_t getContravariantComponentDimSwitch(const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	switch(domain.numDims){
	case 1:
		return getContravariantComponent<scalar_t, 1>(pos, p_block, domain, velocityGlobal);
	case 2:
		return getContravariantComponent<scalar_t, 2>(pos, p_block, domain, velocityGlobal);
	case 3:
		return getContravariantComponent<scalar_t, 3>(pos, p_block, domain, velocityGlobal);
	default:
		return 0;
	}
}


template <typename scalar_t>
__device__ inline scalar_t getContravariantComponentBoundaryVaryingDimSwitch(const I4 pos, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound, const DomainGPU<scalar_t> &domain){
	switch(domain.numDims){
	case 1:
		return getContravariantComponentBoundaryVarying<scalar_t, 1>(pos, p_bound);
	case 2:
		return getContravariantComponentBoundaryVarying<scalar_t, 2>(pos, p_bound);
	case 3:
		return getContravariantComponentBoundaryVarying<scalar_t, 3>(pos, p_bound);
	default:
		return 0;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getContravariantComponentBoundaryFixedDimSwitch(const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound, const DomainGPU<scalar_t> &domain){
	switch(domain.numDims){
	case 1:
		return getContravariantComponentBoundaryFixed<scalar_t, 1>(pos, p_bound);
	case 2:
		return getContravariantComponentBoundaryFixed<scalar_t, 2>(pos, p_bound);
	case 3:
		return getContravariantComponentBoundaryFixed<scalar_t, 3>(pos, p_bound);
	default:
		return 0;
	}
}

#ifdef WITH_GRAD
template <typename scalar_t>
__device__ void scatterContravariantComponentDimSwitch(const scalar_t &velGrad, const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, scalar_t *velocityGlobalGrad){
	switch(domain.numDims){
	case 1:
		scatterContravariantComponentGrad<scalar_t, 1>(velGrad, pos, p_block, domain, velocityGlobalGrad);
		break;
	case 2:
		scatterContravariantComponentGrad<scalar_t, 2>(velGrad, pos, p_block, domain, velocityGlobalGrad);
		break;
	case 3:
		scatterContravariantComponentGrad<scalar_t, 3>(velGrad, pos, p_block, domain, velocityGlobalGrad);
		break;
	default:
		break;
	}
}

template <typename scalar_t>
__device__ void scatterContravariantComponentBoundaryFixedDimSwitch(const scalar_t &velGrad, const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound, const DomainGPU<scalar_t> &domain){
	switch(domain.numDims){
	case 1:
		scatterContravariantComponentGradBoundaryFixed<scalar_t, 1>(velGrad, pos, p_bound);
		break;
	case 2:
		scatterContravariantComponentGradBoundaryFixed<scalar_t, 2>(velGrad, pos, p_bound);
		break;
	case 3:
		scatterContravariantComponentGradBoundaryFixed<scalar_t, 3>(velGrad, pos, p_bound);
		break;
	default:
		break;
	}
}
#endif //WITH_GRAD




template <typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t,DIMS> getPressureGradient(const BlockGPU<scalar_t> &block, const I4 pos, const DomainGPU<scalar_t> &domain){
	Vector<scalar_t,DIMS> pressureGrad;
	for(index_t dim=0; dim<DIMS; ++dim){
		I4 tempPos = pos;
		tempPos.w = 0;
		//boundary handling, correct extrapolation distance norm
		scalar_t fac = 0.5;
		if(!(pos.a[dim]==0 && isEmptyBound(dim*2,block.boundaries))){
			tempPos.a[dim] = pos.a[dim]-1;
		}else{
			fac = 1;
		}
		const scalar_t valN = getPressureAtWithBounds(tempPos, block, domain, false);
		if(!(pos.a[dim]==block.size.a[dim]-1 && isEmptyBound(dim*2+1,block.boundaries))){
			tempPos.a[dim] = pos.a[dim]+1;
		}else{
			tempPos.a[dim] = pos.a[dim];
			fac = 1;
		}
		const scalar_t valP = getPressureAtWithBounds(tempPos, block, domain, false);
		pressureGrad.a[dim] = (valP - valN)*fac;
	}
	
	if(block.hasTransform){
		I4 tempPos = pos;
		tempPos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flattenIndex(tempPos, block);
		//pressureGrad = matmul(T->Minv, pressureGrad);
		pressureGrad = matmul(pressureGrad, T->Minv); // = matmul(transpose(T->Minv), pressureGrad)
	}
	
	return pressureGrad;
}


#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__device__ inline void scatterPressureGradientGrad(Vector<scalar_t,DIMS> pressureGradGrad, const BlockGPU<scalar_t> &block, const I4 pos, const DomainGPU<scalar_t> &domain){
	
	if(block.hasTransform){
		I4 tempPos = pos;
		tempPos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flattenIndex(tempPos, block);
		pressureGradGrad = matmul(T->Minv, pressureGradGrad);
	}
	
	for(index_t dim=0; dim<DIMS; ++dim){
		I4 posP = pos;
		posP.w = 0;
		I4 posN = pos;
		posN.w = 0;
		//boundary handling, correct extrapolation distance norm
		scalar_t fac = 0.5;
		if(!(pos.a[dim]==0 && isEmptyBound(dim*2,block.boundaries))){
			posN.a[dim] = pos.a[dim]-1;
		}else{
			fac = 1;
		}
		if(!(pos.a[dim]==block.size.a[dim]-1 && isEmptyBound(dim*2+1,block.boundaries))){
			posP.a[dim] = pos.a[dim]+1;
		}else{
			fac = 1;
		}
		const scalar_t gradP = pressureGradGrad.a[dim] * fac;
		const scalar_t gradN = pressureGradGrad.a[dim] * -fac;
		scatterPressureGradToWithBounds(gradN, posN, block, domain);
		scatterPressureGradToWithBounds(gradP, posP, block, domain);
	}
}

#endif //WITH_GRAD

template <typename scalar_t, int DIMS>
__device__ inline
const TransformGPU<scalar_t, DIMS>* getFaceTransformPtr(I4 facePos, const index_t face, const BlockGPU<scalar_t> *p_block){
	if(!p_block->hasFaceTransform){ return nullptr; }
	const index_t dim = axisFromBound(face);
	const index_t isUpper = boundIsUpper(face);
	//I4 facePos = pos;
	facePos.a[dim] += isUpper;
	facePos.w = dim;
	I4 faceSize = p_block->size;
	for(index_t i=0; i<DIMS; ++i) { faceSize.a[i] += 1; }
	I4 faceStride = {{.x=1, .y=faceSize.x, .z=faceSize.x*faceSize.y, .w=faceSize.x*faceSize.y*faceSize.z}};
	const index_t flatFacePos = flattenIndex(facePos, faceStride);
	const TransformGPU<scalar_t, DIMS>* p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->faceTransform ) + flatFacePos;
	return p_T;
}

template <typename scalar_t, int DIMS>
__device__ inline
const TransformGPU<scalar_t, DIMS>* getFaceTransformPtr(const I4 &facePos, const index_t face, const BlockGPU<scalar_t> &block){
	return getFaceTransformPtr<scalar_t, DIMS>(facePos, face, &block);
}

template <typename scalar_t, int DIMS>
__device__ inline
Vector<scalar_t,DIMS> lerpHalfpoint(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2){
	return (v1 + v2)*static_cast<scalar_t>(0.5);
}

template <typename scalar_t, int DIMS>
__device__ inline
Vector<scalar_t,DIMS> slerp1Halfpoint(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2, const scalar_t eps){
	// normalize vectors
	const scalar_t sqlen1 = dot(v1,v1);
	const scalar_t sqlen2 = dot(v2,v2);
	if(sqlen1<eps || sqlen2<eps){
		// vector to short to normalize, default to lerp
		return lerpHalfpoint(v1,v2);
	}
	const Vector<scalar_t,DIMS> n1 = v1*rsqrt(sqlen1);
	const Vector<scalar_t,DIMS> n2 = v2*rsqrt(sqlen1);
	
	// get angle between vectors
	const scalar_t d = dot(n1, n2);
	if(d<(eps-1) || (1-eps)<d){
		return lerpHalfpoint(v1,v2);
	}
	const scalar_t rad = acos(d);
	if(rad<eps){
		return lerpHalfpoint(v1,v2);
	}
	
	// slerp direction and magnitude
	const scalar_t w = sin(0.5*rad) / sin(rad);
	const Vector<scalar_t,DIMS> v = (v1 + v2)*w;
	
	return v;
}

template <typename scalar_t, int DIMS>
__device__ inline
Vector<scalar_t,DIMS> slerp2Halfpoint(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2, const scalar_t eps){
	// normalize vectors
	const scalar_t sqlen1 = dot(v1,v1);
	const scalar_t sqlen2 = dot(v2,v2);
	if(sqlen1<eps || sqlen2<eps){
		// vector to short to normalize, default to lerp
		return lerpHalfpoint(v1,v2);
	}
	const Vector<scalar_t,DIMS> n1 = v1*rsqrt(sqlen1);
	const Vector<scalar_t,DIMS> n2 = v2*rsqrt(sqlen1);
	
	// get angle between vectors
	// halfpoint simplification
	Vector<scalar_t,DIMS> n = lerpHalfpoint(n1, n2);
	const scalar_t sqlen = dot(n,n);
	if(sqlen<eps){
		// vector to short to normalize, default to lerp
		return lerpHalfpoint(v1,v2);
	}
	n *= rsqrt(sqlen);
	
	// lerp magnitude
	const scalar_t m1 = sqrt(sqlen1);
	const scalar_t m2 = sqrt(sqlen2);
	const scalar_t m = (m1 + m2)*0.5;
	
	const Vector<scalar_t,DIMS> v = n*m;
	
	return v;
}

template <typename scalar_t, int DIMS>
__device__ inline
scalar_t getSlerpWeight(const Vector<scalar_t,DIMS> &v1, const Vector<scalar_t,DIMS> &v2, const scalar_t eps){
	
	// normalize vectors
	const scalar_t sqlen1 = dot(v1,v1);
	const scalar_t sqlen2 = dot(v2,v2);
	if(sqlen1<eps || sqlen2<eps){
		// vector to short to normalize, default to lerp
		return 0.5;
	}
	const Vector<scalar_t,DIMS> n1 = v1*rsqrt(sqlen1);
	const Vector<scalar_t,DIMS> n2 = v2*rsqrt(sqlen1);
	
	// get angle between vectors
	const scalar_t rad = acos(dot(n1, n2));
	if(rad<eps){
		return 0.5;
	}
	
	const scalar_t weight = sin(0.5*rad) / sin(rad);
	
	return weight;
}

template <typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t,DIMS> getPressureGradientFVM(const BlockGPU<scalar_t> &block, I4 pos, const DomainGPU<scalar_t> &domain,
		const index_t gradientInterpolation){
	// use a finite volume approach (instead of finite differencing)
	// interpolate pressure at the faces, multiply with face area and face normal
	// face area * face normal = row of inverse transform
	// identical to finite difference gradient for orthogonal grids/transforms
	if(!block.hasTransform){ return getPressureGradient<scalar_t, DIMS>(block, pos, domain);}
	

	Vector<scalar_t,DIMS> pressureGrad = {.a={0}};

	pos.w = 0;
	const index_t flatIndexP = flattenIndex(pos, block);
	const TransformGPU<scalar_t, DIMS> *p_TP = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndexP;
	const scalar_t pP = block.pressure[flatIndexP];
	

	for(index_t face=0; face<(DIMS*2); ++face){
		const index_t dim = axisFromBound(face); //face>>1;
		const bool isUpper = boundIsUpper(face); // face&1;
		const index_t faceSign = faceSignFromBound(face); //(face&1) * 2 - 1; // -1 if lower, +1 if upper
		const bool atBound = isAtBound(pos, face, &block);
		
		const Vector<scalar_t, DIMS> fluxP = p_TP->Minv.v[dim] * (p_TP->det * pP);
		
		
		I4 tempPos = pos;
		const TransformGPU<scalar_t, DIMS> *p_T = nullptr;
		index_t axisNeighbor = dim; // needed for possible axis shuffling over CONNECTED_GRID boundaries.
		scalar_t transformSign = 1;
		scalar_t p = 0;
		if(atBound){ //((pos.a[dim]==0 && !isUpper) || (pos.a[dim]==(block.size.a[dim]-1)) && isUpper)){
			switch(block.boundaries[face].type){
				case BoundaryType::VALUE:
					// static boundary with transforms should never happen, but p_T = p_TP is a reasonable fallback.
					p_T = p_TP;
					tempPos.a[dim] = pos.a[dim] - faceSign;
					p = 2*pP - block.pressure[flattenIndex(tempPos, block)];
					break;
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				{
					// boundary is closed, so pressure gradient is 0.
					tempPos.a[dim] = 0;
					if(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING){
						p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.boundaries[face].vdb.transform) + flattenIndex(tempPos, block.boundaries[face].vdb.stride);
					} else {
						p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.boundaries[face].fb.transform) + flattenIndex(tempPos, block.boundaries[face].fb.stride);
					}
					//p = pP;  //using center pressure (extrapolation with p-grad=0) leads to oscillations at the boundary.
					// instead, extrapolate using the pressure gradient at the opposite side. Minimum block resolution requirements mean that there can't be a boundary.
					tempPos.a[dim] = pos.a[dim] - faceSign;
					if(gradientInterpolation!=3){
						p = pP + (pP - block.pressure[flattenIndex(tempPos, block)]) *0.5;
						const Vector<scalar_t, DIMS> fluxN = p_T->Minv.v[dim] * (p_T->det * p);
						pressureGrad += fluxN * static_cast<scalar_t>(faceSign);// * boundaryWeights[dim]);
						
						continue;
					}else{
						p = pP + (pP - block.pressure[flattenIndex(tempPos, block)]);
					}
					break;
				}
				case BoundaryType::GRADIENT:
					// unsupported
					continue;
				case BoundaryType::CONNECTED_GRID:
				{
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[face].cb.connectedGridIndex;
					tempPos.w = dim;
					I4 otherPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[face].cb, domain);
					axisNeighbor = otherPos.w; // for axis shuffling
					const bool otherIsUpper = boundIsUpper(block.boundaries[face].cb.axes.a[0]);
					if(otherIsUpper==isUpper) {
						transformSign = -1;
					}
					otherPos.w = 0;
					const index_t flatIndex = flattenIndex(otherPos, p_connectedBlock);
					p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_connectedBlock->transform) + flatIndex;
					p = p_connectedBlock->pressure[flatIndex];
					break;
				}
				case BoundaryType::PERIODIC:
				{
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
					const index_t flatIndex = flattenIndex(tempPos, block);
					p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndex;
					p = block.pressure[flatIndex];
					break;
				}
				default:
					continue;
			}
		} else {
			tempPos.a[dim] += faceSign;
			const index_t flatIndex = flattenIndex(tempPos, block);
			p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flatIndex;
			p = block.pressure[flatIndex];
		}
		
		
		// like the flux computation. computes the fluxes of the individual cells first before interpolating to the face.
		const Vector<scalar_t, DIMS> fluxN = p_T->Minv.v[axisNeighbor] * (p_T->det * p * transformSign);
		switch(gradientInterpolation){
			case 0:
			{ // lerp
				pressureGrad += (fluxP + fluxN) * (static_cast<scalar_t>(0.5) * faceSign);
				break;
			}
			case 1:
			{ // slerp direction and magnitude
				pressureGrad += slerp1Halfpoint(fluxP, fluxN, static_cast<scalar_t>(1e-5)) * static_cast<scalar_t>(faceSign);
				break;
			}
			case 2:
			{ // slerp direction, lerp magnitude
				pressureGrad += slerp2Halfpoint(fluxP, fluxN, static_cast<scalar_t>(1e-5)) * static_cast<scalar_t>(faceSign);
				break;
			}
			case 3:
			{
				p_T = getFaceTransformPtr<scalar_t, DIMS>(pos, face, block);
				pressureGrad += p_T->Minv.v[dim] * (p_T->det * (p + pP) * static_cast<scalar_t>(0.5) * static_cast<scalar_t>(faceSign));
				break;
			}
			default:
				break;
		}
		//pressureGrad += (fluxP + fluxN) * (static_cast<scalar_t>(1e-5)0.5) * faceSign);// * boundaryWeights[dim]);
	}
	
	pressureGrad *= static_cast<scalar_t>(1.0) / (p_TP->det);// * detWeight);
	
	return pressureGrad;
}

template <typename scalar_t, int DIMS>
__device__ inline scalar_t getDeterminant(const BlockGPU<scalar_t> &block, I4 pos){
	if(block.hasTransform){
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flattenIndex(pos, block);
		const scalar_t det = T->det;
		return det;
	} else {
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getDeterminantDimSwitch(const BlockGPU<scalar_t> &block, const I4 pos, const index_t nDims){
	switch(nDims){
	case 1:
		return getDeterminant<scalar_t, 1>(block, pos);
	case 2:
		return getDeterminant<scalar_t, 2>(block, pos);
	case 3:
		return getDeterminant<scalar_t, 3>(block, pos);
	default:
		return 1;
	}
}

template <typename scalar_t, int DIMS>
__device__ inline scalar_t getDeterminant(const BlockGPU<scalar_t> *p_block, I4 pos){
	if(p_block->hasTransform){
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		return det;
	} else {
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getDeterminantDimSwitch(const BlockGPU<scalar_t> *p_block, const I4 pos, const index_t nDims){
	switch(nDims){
	case 1:
		return getDeterminant<scalar_t, 1>(p_block, pos);
	case 2:
		return getDeterminant<scalar_t, 2>(p_block, pos);
	case 3:
		return getDeterminant<scalar_t, 3>(p_block, pos);
	default:
		return 1;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline scalar_t getTransformMetricOrthogonal(I4 pos, const BlockGPU<scalar_t> *p_block){
	if(p_block->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		return T->Minv.a[component][component];
		
	}else{
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getTransformMetricOrthogonalDimSwitch(const I4 pos, const BlockGPU<scalar_t> *p_block, const index_t nDims){
	switch(nDims){
	case 1:
		return getTransformMetricOrthogonal<scalar_t, 1>(pos, p_block);
	case 2:
		return getTransformMetricOrthogonal<scalar_t, 2>(pos, p_block);
	case 3:
		return getTransformMetricOrthogonal<scalar_t, 3>(pos, p_block);
	default:
		return 1;
	}
}


template<typename scalar_t, int DIMS>
__device__ inline scalar_t getLaplaceCoefficientOrthogonal(I4 pos, const BlockGPU<scalar_t> *p_block){
	if(p_block->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		//load more globally as it is needed for all sides?
		//const MatrixSquare<scalar_t, 3> mInv = T->Minv;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, mInvRow);
		
	}else{
		return 1;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalFace(I4 pos, const index_t face, const BlockGPU<scalar_t> *p_block){
	if(p_block->hasFaceTransform){
		const index_t component = axisFromBound(face);//pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = getFaceTransformPtr<scalar_t, DIMS>(pos, face, p_block);
		const scalar_t det = T->det;
		//load more globally as it is needed for all sides?
		//const MatrixSquare<scalar_t, 3> mInv = T->Minv;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, mInvRow);
		
	}else{
		return 1;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline scalar_t getLaplaceCoefficient(I4 pos, const index_t component1, const index_t component2, const BlockGPU<scalar_t> *p_block){
	if(p_block->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		//load more globally as it is needed for all sides?
		//const MatrixSquare<scalar_t, 3> mInv = T->Minv;
		const Vector<scalar_t, DIMS> mInvRow1 = T->Minv.v[component1];
		const Vector<scalar_t, DIMS> mInvRow2 = T->Minv.v[component2];
		return det*dot(mInvRow1, mInvRow2);
		
	}else{
		return 1;
	}
}

__device__ constexpr index_t getLaplaceCoefficientCenterIndex(const index_t component1, const index_t component2){
	return component1 + component2 + ((component1>>1) | (component2>>1));
}

__device__ constexpr index_t getLaplaceCoefficientsFullLength(const index_t dims){
	return dims * (dims+1) / 2; // gauss sum
}

template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> getLaplaceCoefficientsFull(I4 pos, const BlockGPU<scalar_t> *p_block){
	// 3D: all 6 coefficients
	if(p_block->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		
		Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			for(index_t c2=0; c2<=c1; ++c2){
				coefficients.a[getLaplaceCoefficientCenterIndex(c1,c2)] = det*dot(T->Minv.v[c1], T->Minv.v[c2]);
			}
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS * (DIMS+1) / 2> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			for(index_t c2=0; c2<=c1; ++c2){
				coefficients.a[getLaplaceCoefficientCenterIndex(c1,c2)] = c1==c2 ? 1 : 0;
			}
		}
		return coefficients;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsSingleAxis(I4 pos, const index_t axis, const BlockGPU<scalar_t> *p_block){
	// 3D: 3 coefficients: neighbour direction x all
	if(p_block->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_block->transform) + flattenIndex(pos, p_block);
		const scalar_t det = T->det;
		
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = det*dot(T->Minv.v[c1], T->Minv.v[axis]);
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = c1==axis ? 1 : 0;
		}
		return coefficients;
	}
}

template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsSingleFace(I4 pos, const index_t face, const BlockGPU<scalar_t> *p_block){
	// 3D: 3 coefficients: neighbour direction x all
	const index_t axis = axisFromBound(face);
	if(p_block->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = getFaceTransformPtr<scalar_t, DIMS>(pos, face, p_block);
		const scalar_t det = T->det;
		
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = det*dot(T->Minv.v[c1], T->Minv.v[axis]);
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = c1==axis ? 1 : 0;
		}
		return coefficients;
	}
}

/** go from current position 'pos' in 'block' to the neighbor cell in direction/face 'dir'.
 * Compute the laplace coefficients that contain the axis of 'dir'.
 * If there is a prescribed boundary in direction 'dir', its transformation is returned.
 * If there is a connected block with shuffled axes, the connection is resolved s.t. the laplace coefficients are valid for 'block'.
 * If the connection (block or boundary) has no transformation, laplace coefficients based on the identity transformation are returned.
 * 
 */
template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsSingleAxisNeighbor(I4 pos, const index_t dir, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	
	const index_t axis = axisFromBound(dir);
	pos.w = 0;
	NeighborCellInfo<scalar_t> cellInfo;
	const TransformGPU<scalar_t, DIMS> *p_T = nullptr;
	
	if(isAtBound(pos, dir, block) && isEmptyBound(dir, block.boundaries) && block.boundaries[dir].type!=BoundaryType::FIXED){
		switch(block.boundaries[dir].type){
			case BoundaryType::DIRICHLET_VARYING:
			{
				const VaryingDirichletBoundaryGPU<scalar_t> *p_bound = &(block.boundaries[dir].vdb);
				if(p_bound->hasTransform){
					cellInfo.cell.isBlock = false;
					initDefaultAxisMapping(cellInfo);
					pos.a[axis] = 0;
					p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
				}
			}
			default:
				p_T = nullptr;
				break;
		}
	} else {
		cellInfo = resolveNeighborCell<scalar_t>(pos, dir, &block, domain);
		cellInfo.cell.pos.w = 0; // might have been changed by resolveNeighborCell
		if(cellInfo.cell.isBlock){
			if(cellInfo.cell.p_block->hasTransform){
				p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(cellInfo.cell.p_block->transform)
					+ flattenIndex(cellInfo.cell.pos, cellInfo.cell.p_block);
			}
		} else {
			if(cellInfo.cell.p_bound->hasTransform){
				p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(cellInfo.cell.p_bound->transform)
					+ flattenIndex(cellInfo.cell.pos, cellInfo.cell.p_bound->stride);
			}
		}
		
	}
	
	if(p_T!=nullptr){
		const scalar_t det = p_T->det;
		
		Vector<scalar_t, DIMS> coefficients;
		const index_t axisMapped = axisFromBound(cellInfo.axisMapping.a[axis]); // identity in most cases
		const Vector<scalar_t, DIMS> metricMapped = p_T->Minv.v[axisMapped];
		for(index_t c1=0; c1<DIMS; ++c1){
			const index_t c1Mapped = axisFromBound(cellInfo.axisMapping.a[c1]);
			coefficients.a[c1] = det*dot(p_T->Minv.v[c1Mapped], metricMapped);
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS> coefficients;
		// TODO: shuffle for connected grid?
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = c1==axis ? 1 : 0;
		}
		return coefficients;
	}
}
template<typename scalar_t, int DIMS>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsSingleAxisNeighbor(const NeighborCellInfo<scalar_t> &cellInfo, const index_t dir, const DomainGPU<scalar_t> &domain){
	
	const index_t axis = axisFromBound(dir);
	const TransformGPU<scalar_t, DIMS> *p_T = nullptr;
	
	I4 pos = cellInfo.cell.pos;
	pos.w = 0;
	if(cellInfo.cell.isBlock){
		if(cellInfo.cell.p_block->hasTransform){
			p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(cellInfo.cell.p_block->transform)
				+ flattenIndex(pos, cellInfo.cell.p_block);
		}
	} else {
		if(cellInfo.cell.p_bound->hasTransform){
			p_T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(cellInfo.cell.p_bound->transform)
				+ flattenIndex(pos, cellInfo.cell.p_bound->stride);
		}
	}
		
	
	
	Vector<scalar_t, DIMS> coefficients = {.a={0}};
	if(p_T!=nullptr){
		const scalar_t det = p_T->det;
		const index_t axisMapped = axisFromBound(cellInfo.axisMapping.a[axis]); // identity in most cases
		const Vector<scalar_t, DIMS> metricMapped = p_T->Minv.v[axisMapped];
		for(index_t c1=0; c1<DIMS; ++c1){
			const index_t c1Mapped = axisFromBound(cellInfo.axisMapping.a[c1]);
			coefficients.a[c1] = det*dot(p_T->Minv.v[c1Mapped], metricMapped);
		}
		
	}else{
		coefficients.a[axis] = 1;
	}
	return coefficients;
}

template<typename scalar_t, int DIMS, typename BOUNDARY_T>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalBoundary(I4 pos, const BOUNDARY_T *p_bound){
	if(p_bound->hasTransform){
		const index_t component = pos.w;
		pos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		const Vector<scalar_t, DIMS> mInvRow = T->Minv.v[component];
		return det*dot(mInvRow, mInvRow);
		
	}else{
		return 1;
	}
}

template<typename scalar_t, int DIMS, typename BOUNDARY_T>
__device__ inline Vector<scalar_t, DIMS> getLaplaceCoefficientsNeighbourBoundary(I4 pos, const index_t axis, const BOUNDARY_T *p_bound){
	// 3D: 3 coefficients: neighbour direction x all
	if(p_bound->hasTransform){
		//const index_t component = pos.w;
		pos.w = 0;
		pos.a[axis] = 0; //assuming p_bound is on axis
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(p_bound->transform) + flattenIndex(pos, p_bound->stride);
		const scalar_t det = T->det;
		
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = det*dot(T->Minv.v[c1], T->Minv.v[axis]);
		}
		return coefficients;
		
	}else{
		Vector<scalar_t, DIMS> coefficients;
		for(index_t c1=0; c1<DIMS; ++c1){
			coefficients.a[c1] = c1==axis ? 1 : 0;
		}
		return coefficients;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalDimSwitch(const I4 pos, const BlockGPU<scalar_t> *p_block, const index_t nDims){
	switch(nDims){
	case 1:
		return getLaplaceCoefficientOrthogonal<scalar_t, 1>(pos, p_block);
	case 2:
		return getLaplaceCoefficientOrthogonal<scalar_t, 2>(pos, p_block);
	case 3:
		return getLaplaceCoefficientOrthogonal<scalar_t, 3>(pos, p_block);
	default:
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalBoundaryVaryingDimSwitch(const I4 pos, const VaryingDirichletBoundaryGPU<scalar_t> *p_bound, const index_t nDims){
	switch(nDims){
	case 1:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 1, VaryingDirichletBoundaryGPU<scalar_t>>(pos, p_bound);
	case 2:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 2, VaryingDirichletBoundaryGPU<scalar_t>>(pos, p_bound);
	case 3:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 3, VaryingDirichletBoundaryGPU<scalar_t>>(pos, p_bound);
	default:
		return 1;
	}
}
template <typename scalar_t>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(const I4 pos, const FixedBoundaryGPU<scalar_t> *p_bound, const index_t nDims){
	switch(nDims){
	case 1:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 1, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	case 2:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 2, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	case 3:
		return getLaplaceCoefficientOrthogonalBoundary<scalar_t, 3, FixedBoundaryGPU<scalar_t>>(pos, p_bound);
	default:
		return 1;
	}
}

template <typename scalar_t>
__device__ inline scalar_t getLaplaceCoefficientOrthogonalFaceDimSwitch(const I4 &pos, const index_t face, const BlockGPU<scalar_t> *p_block, const index_t nDims){
	switch(nDims){
	case 1:
		return getLaplaceCoefficientOrthogonalFace<scalar_t, 1>(pos, face, p_block);
	case 2:
		return getLaplaceCoefficientOrthogonalFace<scalar_t, 2>(pos, face, p_block);
	case 3:
		return getLaplaceCoefficientOrthogonalFace<scalar_t, 3>(pos, face, p_block);
	default:
		return 1;
	}
}


/**
 * Compute the advective fluxes for all faces of the cell at 'pos'. The fluxes are NOT multiplied by the face sign.
 */
template <typename scalar_t>
__device__ void computeFluxesNDLoop(const I4 pos, scalar_t* fluxes, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	
	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const bool atBound = isAtBound(pos, bound, &block);
		
		I4 tempPos = pos;
		tempPos.w = dim;
		
		const scalar_t velC = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
		
		if(atBound){// lower boundary
			switch(block.boundaries[bound].type){
			case BoundaryType::DIRICHLET:
				// enforce flux
				fluxes[bound] = block.boundaries[bound].sdb.velocity.a[dim];
				break;
			case BoundaryType::DIRICHLET_VARYING:
				// enforce flux
			{
				tempPos.a[dim] = 0;
				fluxes[bound] = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &block.boundaries[bound].vdb, domain);
				break;
			}
			case BoundaryType::FIXED:
				// enforce flux
			{
				tempPos.a[dim] = 0;
				fluxes[bound] = getContravariantComponentBoundaryFixedDimSwitch(tempPos, &block.boundaries[bound].fb, domain);
				break;
			}
			case BoundaryType::GRADIENT:
				// compute flux only from center cell?
				// velN = velC - grad*distance*2 & flux = (velN+velC)*0.5 ->
				fluxes[bound] = 0; //velC + block.boundaries[bound].snb.boundaryGradient.a[dim] * 0.5; //* distance
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				// handle multi-block grids, load from correct cell of the connected grid
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain);
				// tempPos.w is always dim, so connectedAxis==0. otherwise use computeConnectedPosWithChannel()
				otherPos.w = block.boundaries[bound].cb.axes.a[0]>>1;
				scalar_t velN = getContravariantComponentDimSwitch(otherPos, p_connectedBlock, domain, velocityGlobal);
				
				// if the connection goes upper to upper or lower to lower, the velocity has to be inverted
				// (if it goes upper->lower or lower->upper it should be fine)
				//const bool isUpper = boundIsUpper(bound);
				const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
				if(otherIsUpper==isUpper) {
					// connected to a boundary of the same side (lower/upper), need to invert its flux to be consistent
					velN = -velN;
				}
				fluxes[bound] = (velN + velC) * 0.5f;
				break;
			}
			case BoundaryType::PERIODIC:
			{
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
				const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
				fluxes[bound] = (velN + velC) * 0.5f;
				break;
			}
			default:
				fluxes[bound] = 0;
				break;
			}
		}else{
			tempPos.a[dim] = pos.a[dim] + faceSignFromBound(bound);
			const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
			fluxes[bound] = (velN + velC) * 0.5f;
		}
	}
	
}

#ifdef WITH_GRAD


template <typename scalar_t>
__device__ void ScatterFluxesGradNDLoop(const I4 pos, const scalar_t* fluxesGrad, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, scalar_t *velocityGlobalGrad){
	
	
	for(index_t dim=0; dim<domain.numDims; ++dim) {
		I4 tempPos = pos;
		tempPos.w = dim;
		
		scalar_t velCGrad = 0;
		
		for(index_t isUpper=0; isUpper<2; ++isUpper){
		
			int bound = dim*2 + isUpper;
			const bool atBound = isAtBound(pos, bound, &block);
			//index_t faceSign = -1 + (bound&1)*2;
		
			if(atBound){// lower boundary
				switch(block.boundaries[bound].type){
				case BoundaryType::FIXED:
				{
					tempPos.a[dim] = 0;
					//fluxes[bound] = getContravariantComponentBoundaryFixedDimSwitch(tempPos, &block.boundaries[bound].fb, domain);
					scatterContravariantComponentBoundaryFixedDimSwitch(fluxesGrad[bound], tempPos, &block.boundaries[bound].fb, domain);
					break;
				}
				case BoundaryType::CONNECTED_GRID:
				{
					const bool otherIsUpper = boundIsUpper(block.boundaries[bound].cb.axes.a[0]);
					const scalar_t otherFactor = (otherIsUpper==isUpper) ? -0.5 : 0.5;
					
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain);
					otherPos.w = block.boundaries[bound].cb.axes.a[0]>>1;
					
					scatterContravariantComponentDimSwitch(otherFactor *fluxesGrad[bound], otherPos, p_connectedBlock, domain, velocityGlobalGrad);
					velCGrad += 0.5f*fluxesGrad[bound];
					break;
				}
				case BoundaryType::PERIODIC:
				{
					tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
					//block.velocity_grad[flattenIndex(tempPos, block)] = 0.5f*fluxesGrad[bound];
					//atomicAdd(block.velocity_grad + flattenIndex(tempPos, block), 0.5f*fluxesGrad[bound]);
					scatterContravariantComponentDimSwitch(0.5f*fluxesGrad[bound], tempPos, &block, domain, velocityGlobalGrad);
					velCGrad += 0.5f*fluxesGrad[bound];
					break;
				}
				default:
					break;
				}
			}else{
				tempPos.a[dim] = pos.a[dim] + faceSignFromBound(bound);
				//block.velocity_grad[flattenIndex(tempPos, block)] = 0.5f*fluxesGrad[bound];
				//atomicAdd(block.velocity_grad + flattenIndex(tempPos, block), 0.5f*fluxesGrad[bound]);
				scatterContravariantComponentDimSwitch(0.5f*fluxesGrad[bound], tempPos, &block, domain, velocityGlobalGrad);
				velCGrad += 0.5f*fluxesGrad[bound];
			}
		}
		
		tempPos.a[dim] = pos.a[dim];
		//block.velocity_grad[flattenIndex(tempPos, block)] = velCGrad;
		//atomicAdd(block.velocity_grad + flattenIndex(tempPos, block), velCGrad);
		scatterContravariantComponentDimSwitch(velCGrad, tempPos, &block, domain, velocityGlobalGrad);
	}
}

#endif //WITH_GRAD

template <typename scalar_t, index_t DIMS>
__device__
void computeFluxesWithFaceTransforms(const I4 pos, scalar_t* fluxes, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const scalar_t *velocityGlobal){
	// UNTESTED!
	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const bool atBound = isAtBound(pos, bound, &block);
		
		I4 tempPos = pos;
		tempPos.w = dim;
		
		//const scalar_t velC = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);

		bool isBoundVel = false;
		Vector<scalar_t,DIMS> velN = {.a={0}};
		
		if(atBound){// lower boundary
			switch(block.boundaries[bound].type){
			case BoundaryType::DIRICHLET:
				// enforce flux
				for(index_t compIdx=0; compIdx<DIMS; ++compIdx){
					velN.a[compIdx] = block.boundaries[bound].sdb.velocity.a[compIdx]; // sdb.velocity always has DIMS==3
				}
				break;
			case BoundaryType::DIRICHLET_VARYING:
				// enforce flux
			{
				tempPos.a[dim] = 0;
				//fluxes[bound] = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &block.boundaries[bound].vdb, domain);
				velN = getVelocityFromBoundaryVarying<scalar_t, DIMS>(tempPos, &block.boundaries[bound].vdb);
				break;
			}
			case BoundaryType::FIXED:
				// enforce flux
			{
				tempPos.a[dim] = 0;
				velN = getVelocityFromBoundaryFixed<scalar_t, DIMS>(tempPos, &block.boundaries[bound].fb);
				break;
			}
			case BoundaryType::GRADIENT:
				// compute flux only from center cell?
				// velN = velC - grad*distance*2 & flux = (velN+velC)*0.5 ->
				fluxes[bound] = 0; //velC + block.boundaries[bound].snb.boundaryGradient.a[dim] * 0.5; //* distance
				continue;
				break;
			case BoundaryType::CONNECTED_GRID:
			{
				// handle multi-block grids, load from correct cell of the connected grid
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain);
				//scalar_t velN = getContravariantComponentDimSwitch(otherPos, p_connectedBlock, domain, velocityGlobal);
				velN = getVelocityFromBlock<scalar_t, DIMS>(otherPos, p_connectedBlock, domain, velocityGlobal);
				break;
			}
			case BoundaryType::PERIODIC:
			{
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = isUpper ? 0 : block.size.a[dim] - 1;
				//const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
				velN = getVelocityFromBlock<scalar_t, DIMS>(tempPos, &block, domain, velocityGlobal);
				break;
			}
			default:
				fluxes[bound] = 0;
				break;
			}
		}else{
			tempPos.a[dim] = pos.a[dim] + faceSignFromBound(bound);
			//const scalar_t velN = getContravariantComponentDimSwitch(tempPos, &block, domain, velocityGlobal);
			velN = getVelocityFromBlock<scalar_t, DIMS>(tempPos, &block, domain, velocityGlobal);
		}

		if(!isBoundVel){
			const Vector<scalar_t,DIMS> velC = getVelocityFromBlock<scalar_t, DIMS>(pos, &block, domain, velocityGlobal);
			velN = slerp2Halfpoint(velC, velN, static_cast<scalar_t>(1e-5));
		}

		const TransformGPU<scalar_t, DIMS> *p_T = getFaceTransformPtr<scalar_t, DIMS>(pos, bound, block);
		fluxes[bound] = p_T->det * dot(p_T->Minv.v[dim], velN);
	}
}


template <typename scalar_t>
__device__
scalar_t getViscosity(const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	if(forPassiveScalar && (domain.scalarViscosity!=nullptr)){
		if(domain.scalarViscosityStatic){
			return domain.scalarViscosity[0];
		}else{
			return domain.scalarViscosity[passiveScalarChannel];
		}
	}else{
		return domain.viscosity;
	}
}
template <typename scalar_t>
__device__
scalar_t getViscosityBlock(const I4 &pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	if(forPassiveScalar){
		// TODO: per-block passive scalar
		return getViscosity<scalar_t>(domain, true, passiveScalarChannel);
	}else {
		if(p_block->viscosity==nullptr){
			// Alternative: set p_block->viscosity to global domain viscosity on copy to gpu
			return getViscosity<scalar_t>(domain, false, 0);
		} else {
			if(p_block->isViscosityStatic){
				return p_block->viscosity[0];
			} else {
				I4 tempPos = pos;
				tempPos.w = 0; // velocity viscosity can't be per channel
				const index_t flatPos = flattenIndex(tempPos, p_block);
				return p_block->viscosity[flatPos];
			}
		}
	}
}
template <typename scalar_t>
__device__
scalar_t getViscosityFixedBoundary(const I4 &posBlock, const FixedBoundaryGPU<scalar_t> *p_fbound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	// TODO: per-boundary viscosity
	return getViscosityBlock<scalar_t>(posBlock, p_block, domain, forPassiveScalar, passiveScalarChannel);
}

#ifdef WITH_GRAD

template <typename scalar_t>
__device__
void scatterViscosity_GRAD(const scalar_t viscosity_grad, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	scalar_t *p_viscosity_grad = nullptr;
	if(forPassiveScalar && (domain.scalarViscosity_grad!=nullptr)){
		if(domain.scalarViscosityStatic){
			p_viscosity_grad = domain.scalarViscosity_grad;
		}else{
			p_viscosity_grad = domain.scalarViscosity_grad + passiveScalarChannel;
		}
	}else{
		p_viscosity_grad = domain.viscosity_grad;
	}
	
	// TODO: this is one address per channel, use better reduction scheme?
	if(p_viscosity_grad){ atomicAdd(p_viscosity_grad, viscosity_grad); }
}
template <typename scalar_t>
__device__
void scatterViscosityBlock_GRAD(const scalar_t viscosity_grad, const I4 &pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	if(forPassiveScalar){
		// TODO: per-block passive scalar
		scatterViscosity_GRAD(viscosity_grad, domain, true, passiveScalarChannel);
	}else {
		if(p_block->viscosity==nullptr){ // forward value was taken from global, so we scatter to global grad
			// Alternative: set p_block->viscosity_grad to global domain viscosity on copy to gpu
			scatterViscosity_GRAD(viscosity_grad, domain, false, 0);
		} else {
			scalar_t *p_viscosity_grad = nullptr;
			if(p_block->isViscosityStatic){
				p_viscosity_grad = p_block->viscosity_grad;
			} else {
				I4 tempPos = pos;
				tempPos.w = 0;
				const index_t flatPos = flattenIndex(tempPos, p_block);
				p_viscosity_grad = p_block->viscosity_grad + flatPos;
			}
			if(p_viscosity_grad){ atomicAdd(p_viscosity_grad, viscosity_grad); }
		}
	}
}
template <typename scalar_t>
__device__
void scatterViscosityBoundary_GRAD(const scalar_t viscosity_grad, const I4 &posBlock, const FixedBoundaryGPU<scalar_t> *p_fbound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const bool forPassiveScalar, const index_t passiveScalarChannel){
	// TODO: per-boundary viscosity
	scatterViscosityBlock_GRAD(viscosity_grad, posBlock, p_block, domain, forPassiveScalar, passiveScalarChannel);
}

#endif //WITH_GRAD


// --- Laplace kernel coefficients for transformed grids. Used for pressure solve and diffusion. ---

/**
 * to resolve cyclic axis indices from interpolateNonOrthoLaplaceComponents
 */
__host__ __device__
constexpr index_t getAxisRelativeToOtherNonOrtho(const index_t axis, const index_t otherAxis, const index_t dims){
	return (dims - 1 - otherAxis + axis)%dims; 
}

template <typename scalar_t>
__device__
scalar_t inline getInterpolatedNonOrthoLaplaceComponent(const scalar_t *alphaInterp, const index_t face, const index_t otherAxis, const index_t dims){
	const index_t otherAxisRelativeToFace = getAxisRelativeToOtherNonOrtho(otherAxis, axisFromBound(face), dims);
	return alphaInterp[face*(dims-1) + otherAxisRelativeToFace]; // always =alphaInterp[face] for 2D
}

#ifdef WITH_GRAD
template <typename scalar_t>
__device__
void inline addInterpolatedNonOrthoLaplaceComponent_GRAD(scalar_t value, scalar_t *alphaInterp_grad, const index_t face, const index_t otherAxis, const index_t dims){
	const index_t otherAxisRelativeToFace = getAxisRelativeToOtherNonOrtho(otherAxis, axisFromBound(face), dims);
	alphaInterp_grad[face*(dims-1) + otherAxisRelativeToFace] += value; // always =alphaInterp[face] for 2D
}

#endif //WITH_GRAD

template <typename scalar_t, index_t DIMS>
__device__
void interpolateNonOrthoLaplaceComponents(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, scalar_t *alphaInterpOut,
	const bool withViscosity, const bool withA, const bool useFaceTransform){
	// useFaceTransform is experimental
	// needed length for alphaInterpOut: 2*DIMS*(DIMS-1). 2D: 4, 3D: 12
	// 2D: -x01, +x01, -y10, +y10. (note: 01=10)
	// 3D: -x01, -x02, +x01, +x02, -y12, -y 10, +y12, +y10, -z20, -z21, +z20, +z21

	// get center coefficients
	const Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> alphasP = getLaplaceCoefficientsFull<scalar_t, DIMS>(pos, &block);
	const index_t flatPosGlobal = block.globalOffset + flattenIndex(pos, block);
	
	const scalar_t one = static_cast<scalar_t>(1.0);
	const scalar_t raP = (withA ? one /domain.Adiag[flatPosGlobal] : one) * (withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, false, 0) : one);

	// loop neighbours to get interpolated coefficients
	for(index_t face=0; face<DIMS*2; ++face){
		const index_t axis = face>>1;
		// Using alias for copied code. TODO: unify.
		const index_t &dim = axis;
		const index_t &bound = face;
		const index_t faceSign = -1 + ((face&1)<<1);
		const bool isUpper = face&1;
		I4 posN = pos;
		posN.w = axis;
		Vector<scalar_t, DIMS> alphasN;
		index_t flatPosGlobalN = 0;
		if((0<pos.a[axis] && pos.a[axis]<(block.size.a[axis]-1)) || !isEmptyBound(bound, block.boundaries)){ // not at prescribed bound
			const BlockGPU<scalar_t> *p_block = &block;
			if(((pos.a[dim]==0 && !isUpper) || (pos.a[dim]==block.size.a[axis]-1 && isUpper)) && block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				posN = computeConnectedPosWithChannel(posN, dim, &block.boundaries[bound].cb, domain);
				alphasN = getLaplaceCoefficientsSingleAxis<scalar_t, DIMS>(posN, axis, p_connectedBlock);
				posN.w = 0;
				flatPosGlobalN = flattenIndex(posN, p_connectedBlock) + p_connectedBlock->globalOffset;
				p_block = p_connectedBlock;
			}else {
				if(!isUpper && pos.a[dim]==0 && block.boundaries[bound].type==BoundaryType::PERIODIC){
					posN.a[dim] = block.size.a[dim]-1;
				}else if(isUpper && pos.a[dim]==block.size.a[dim]-1 && block.boundaries[bound].type==BoundaryType::PERIODIC){
					posN.a[dim] = 0;
				}else{
					posN.a[dim] = pos.a[dim] + faceSign;
				}
				alphasN = getLaplaceCoefficientsSingleAxis<scalar_t, DIMS>(posN, axis, &block);
				posN.w = 0;
				flatPosGlobalN = flattenIndex(posN, block) + block.globalOffset;
				//rowValues[bound+1] = -1 / advectionMatrixDiagonal[tempFlatPos] * invLaplace;
			}
			const scalar_t raN = (withA ? one /domain.Adiag[flatPosGlobalN] : one) * (withViscosity ? getViscosityBlock<scalar_t>(posN, p_block, domain, false, 0) : one);
			if(useFaceTransform){
				//overwrite alphasN with face transform
				alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block); // <- ISSUE here? no
				//alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(posN, face, &block);
			}
			for(index_t i=1; i<DIMS; ++i){
				const index_t otherAxis = (axis + i)%DIMS;
				scalar_t alphaFace = 0;
				if(useFaceTransform){
					alphaFace = alphasN.a[otherAxis] * (raP + raN) * 0.5;
				}else{
					alphaFace = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, otherAxis)]*raP + alphasN.a[otherAxis]*raN )*0.5;
				}
				alphaInterpOut[face*(DIMS-1) + (i-1)] = alphaFace;
			}
		} else {
			//TODO: get boundary transform metrics directly? 
			
			//getLaplaceCoefficientsNeighbourBoundaryVarying<scalar_t, DIMS>(, axis, bound)
			for(index_t i=1; i<DIMS; ++i){
				alphaInterpOut[face*(DIMS-1) + (i-1)] = 0;
			}
		}
	}
}

#ifdef WITH_GRAD

template <typename scalar_t, index_t DIMS>
__device__
void scatterNonOrthoLaplaceComponents_GRAD(scalar_t *alphaInterp_grad, const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const bool withViscosity, const bool withA, const bool useFaceTransform){
	// useFaceTransform is experimental
	// needed length for alphaInterpOut: 2*DIMS*(DIMS-1). 2D: 4, 3D: 12
	// 2D: -x01, +x01, -y10, +y10. (note: 01=10)
	// 3D: -x01, -x02, +x01, +x02, -y12, -y 10, +y12, +y10, -z20, -z21, +z20, +z21

	// get center coefficients
	const Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> alphasP = getLaplaceCoefficientsFull<scalar_t, DIMS>(pos, &block);
	const index_t flatPosGlobal = block.globalOffset + flattenIndex(pos, block);
	
	const scalar_t one = static_cast<scalar_t>(1.0);
	//const scalar_t raP = (withA ? one /domain.Adiag[flatPosGlobal] : one) * (withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, false, 0) : one);
	scalar_t raP_grad = 0;

	// loop neighbours to get interpolated coefficients
	for(index_t face=0; face<DIMS*2; ++face){
		const index_t axis = face>>1;
		// Using alias for copied code. TODO: unify.
		const index_t &dim = axis;
		const index_t &bound = face;
		const index_t faceSign = -1 + ((face&1)<<1);
		const bool isUpper = face&1;
		I4 posN = pos;
		posN.w = axis;
		Vector<scalar_t, DIMS> alphasN;
		index_t flatPosGlobalN = 0;
		if((0<pos.a[axis] && pos.a[axis]<(block.size.a[axis]-1)) || !isEmptyBound(bound, block.boundaries)){ // not at prescribed bound
			const BlockGPU<scalar_t> *p_block = &block;
			if(((pos.a[dim]==0 && !isUpper) || (pos.a[dim]==block.size.a[axis]-1 && isUpper)) && block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				posN = computeConnectedPosWithChannel(posN, dim, &block.boundaries[bound].cb, domain);
				alphasN = getLaplaceCoefficientsSingleAxis<scalar_t, DIMS>(posN, axis, p_connectedBlock);
				posN.w = 0;
				flatPosGlobalN = flattenIndex(posN, p_connectedBlock) + p_connectedBlock->globalOffset;
				p_block = p_connectedBlock;
			}else {
				if(!isUpper && pos.a[dim]==0 && block.boundaries[bound].type==BoundaryType::PERIODIC){
					posN.a[dim] = block.size.a[dim]-1;
				}else if(isUpper && pos.a[dim]==block.size.a[dim]-1 && block.boundaries[bound].type==BoundaryType::PERIODIC){
					posN.a[dim] = 0;
				}else{
					posN.a[dim] = pos.a[dim] + faceSign;
				}
				alphasN = getLaplaceCoefficientsSingleAxis<scalar_t, DIMS>(posN, axis, &block);
				posN.w = 0;
				flatPosGlobalN = flattenIndex(posN, block) + block.globalOffset;
				//rowValues[bound+1] = -1 / advectionMatrixDiagonal[tempFlatPos] * invLaplace;
			}
			if(useFaceTransform){
				//overwrite alphasN with face transform
				alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block); // <- ISSUE here? no
				//alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(posN, face, &block);
			}

			//const scalar_t raN = (withA ? one /domain.Adiag[flatPosGlobalN] : one) * (withViscosity ? getViscosityBlock<scalar_t>(posN, p_block, domain, false, 0) : one);
			scalar_t raN_grad = 0;

			for(index_t i=1; i<DIMS; ++i){
				const index_t otherAxis = (axis + i)%DIMS;
				const scalar_t alphaFace_grad = alphaInterp_grad[face*(DIMS-1) + (i-1)];
				if(useFaceTransform){
					//alphaFace = alphasN.a[otherAxis] * (raP + raN) * 0.5;
					raP_grad += alphasN.a[otherAxis] * 0.5 * alphaFace_grad;
					raN_grad += alphasN.a[otherAxis] * 0.5 * alphaFace_grad;
				}else{
					//alphaFace = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, otherAxis)]*raP + alphasN.a[otherAxis]*raN )*0.5;
					raP_grad += alphasP.a[getLaplaceCoefficientCenterIndex(axis, otherAxis)] * 0.5 * alphaFace_grad;
					raN_grad += alphasN.a[otherAxis] * 0.5 * alphaFace_grad;

				}
				//alphaInterpOut[face*(DIMS-1) + (i-1)] = alphaFace;
			}
			
			const scalar_t raN = withA ? one /domain.Adiag[flatPosGlobalN] : one;
			
			if(withViscosity){
				scalar_t raN_gradTemp = raN_grad;
				if(withA){
					raN_gradTemp *= raN; 
				}
				scatterViscosityBlock_GRAD<scalar_t>(raN_gradTemp, posN, p_block, domain, false, 0);
			}
			if(withA){
				if(withViscosity){
					raN_grad *= getViscosityBlock<scalar_t>(posN, p_block, domain, false, 0);
				}
				atomicAdd(domain.Adiag_grad + flatPosGlobalN, -raN_grad*raN*raN);
			}
		}
	}
	
	const scalar_t raP = withA ? one /domain.Adiag[flatPosGlobal] : one;

	if(withViscosity){
		scalar_t raP_gradTemp = raP_grad;
		if(withA){
			raP_gradTemp *= raP; 
		}
		scatterViscosityBlock_GRAD<scalar_t>(raP_gradTemp, pos, &block, domain, false, 0);
	}
	if(withA){
		if(withViscosity){
			raP_grad *= getViscosityBlock<scalar_t>(pos, &block, domain, false, 0);
		}
		atomicAdd(domain.Adiag_grad + flatPosGlobal, -raP_grad*raP*raP);
	}
	
}

#endif //WITH_GRAD


template<typename scalar_t>
__device__ scalar_t getPressureNeighborDiagonal(const I4 pos, const index_t dir1, const index_t dir2, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/

	const bool dir1Empty = isEmptyBound(dir1, block.boundaries);
	// if dir1 leads to a prescibed boundary check dir2 first. if both are prescribed dir2 will be used
	const index_t dirs[2] = {dir1Empty ? dir2 : dir1, dir1Empty ? dir1 : dir2};
	//const index_t &d1 = dir1Empty ? dir2 : dir1;
	//const index_t &d2 = dir1Empty ? dir1 : dir2;
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	for(index_t i=0; i<2; ++i){
		const index_t bound = dirs[i];
		const index_t dim = axisFromBound(bound);
		const bool isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		if(isUpper ? tempPos.a[dim]==(p_block->size.a[dim]-1) : tempPos.a[dim]==0){ //check if there is a boundary in the direction we want to move
			switch(p_block->boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					return zeroBound ? 0 : p_block->pressure[flattenIndex(tempPos, p_block)];
					break;
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[bound].cb), domain, 1);
					p_block = p_connectedBlock;
					break;
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
					break;
				default:
					return 0;
					break;
			}
		} else {
			//same block, just update position
			tempPos.a[dim] += faceSign;
		}
	}
	tempPos.w = 0;
	return p_block->pressure[flattenIndex(tempPos, p_block)];
}

/** DEPRECATED, use getBlockData or getBlockDataNeighbor. */
template<typename scalar_t>
__device__ scalar_t getPressureAtWithBounds(const I4 pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const bool zeroBound){
	// read value at location with support for 1-cell ghost layer
	// any position outside the domain will be treated as being on the ghost layer
	// only one coordinate may be outside the domain (corner ghost cells are not supported)
	
	const int flatPos = flattenIndex(pos, block);
	I4 tempPos = pos;
	tempPos.w = 0; //pressure is scalar
	
	for(int dim=0; dim<domain.numDims; ++dim)
	{
		int bound = dim*2;
		
		if(pos.a[dim]<0){// lower boundary
			switch(block.boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					tempPos.a[dim] = 0;
					return zeroBound ? 0 : block.pressure[flattenIndex(tempPos, block)];
					// compute flux only from center cell?
					// velN = velC - grad*distance
					//return grid[flatPos] - domain.bounds[bound].prescribedValue; //* distance
					//break;
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					const I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain, -(pos.a[dim]+1));
					// pressure is scalar, so pos.w==0 always
					return p_connectedBlock->pressure[flattenIndex(otherPos, p_connectedBlock)];
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] += block.size.a[dim];
					return block.pressure[flattenIndex(tempPos, block)];
				default:
					return 0;
			}
		}
		
		bound = dim*2 + 1;
		
		if(pos.a[dim]>=block.size.a[dim]){// upper boundary
			switch(block.boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					tempPos.a[dim] = block.size.a[dim] - 1;
					return zeroBound ? 0 : block.pressure[flattenIndex(tempPos, block)];
				/* case BoundaryType::GRADIENT:
					// compute flux only from center cell?
					// velP = celC + grad*distance
					return grid[flatPos] + domain.bounds[bound].prescribedValue; //* distance
					break; */
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					const I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain, pos.a[dim]-block.size.a[dim]);
					return p_connectedBlock->pressure[flattenIndex(otherPos, p_connectedBlock)];
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] -= block.size.a[dim];
					return block.pressure[flattenIndex(tempPos, block)];
				default:
					return 0;
			}
		}
	}
	return block.pressure[flatPos];
}

template<typename scalar_t>
__device__
scalar_t getBlockVelocitySource(const I4 &pos, const BlockGPU<scalar_t> *p_block){
	if(p_block->velocitySource!=nullptr){
		if(p_block->isVelocitySourceStatic){
			return p_block->velocitySource[pos.w];
		} else {
			return p_block->velocitySource[flattenIndex(pos, p_block)];
		}
	}
	return 0;
}
template<typename scalar_t>
__device__
void scatterBlockVelocitySource_GRAD(const scalar_t vel_grad, const I4 &pos, const BlockGPU<scalar_t> *p_block){
	if(p_block->velocitySource_grad!=nullptr){
		if(p_block->isVelocitySourceStatic){
			atomicAdd(p_block->velocitySource_grad + pos.w, vel_grad);
		} else {
			atomicAdd(p_block->velocitySource_grad + flattenIndex(pos, p_block), vel_grad);
		}
	}
}

/** Returns the start pointer of the requested data tensor. */
template<typename scalar_t>
__device__
scalar_t *getBlockDataPtr(const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	if(type==GridDataType::IS_FIXED_BOUNDARY) { return nullptr; }
	
	const index_t dataIndex = gridDataTypeToIndex(type);
	const bool isGrad = isGradDataType(type);
	const bool isGlobal = isGlobalDataType(type);
	const bool isResult = isResultDataType(type);
	const bool isRHS = isRHSDataType(type);
	
	if(!isGrad){
		if(!isGlobal){ // block data
			return p_block->data[dataIndex];
		}
		if(isResult){
			return domain.results[dataIndex];
		}
		if(isRHS){
			return domain.RHS[dataIndex];
		}
	}
#ifdef WITH_GRAD
	else {
		if(!isGlobal){ // block data
			return p_block->grad[dataIndex];
		}
		if(isResult){
			return domain.results_grad[dataIndex];
		}
		if(isRHS){
			return domain.RHS_grad[dataIndex];
		}
	}
#endif //WITH_GRAD
	
	return nullptr;
}

/** Returns the pointer to the data of 'type' of cell 'pos' in a block. */
template<typename scalar_t>
__device__
scalar_t *getBlockCellDataPtr(I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	scalar_t* p_data = getBlockDataPtr<scalar_t>(p_block, domain, type);
	if(p_data==nullptr){ return nullptr;}
	
	//if(isScalarDataType(type)) { pos.w = 0; } // passive scalar can have multiple channels now
	const bool isGlobal = isGlobalDataType(type);
	const index_t flatPos = isGlobal ? flattenIndexGlobal(pos, p_block, domain) : flattenIndex(pos, p_block);
	
	return p_data + flatPos;
}


template<typename scalar_t>
__device__
scalar_t getBlockData(const I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	
	const scalar_t* p_data = getBlockCellDataPtr<scalar_t>(pos, p_block, domain, type);
	
	if(p_data){ 
		return *p_data;
	} else {
		return 0;
	}
}

template<typename scalar_t>
__device__
void writeBlockData(const scalar_t data, I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	scalar_t* p_data = getBlockCellDataPtr<scalar_t>(pos, p_block, domain, type);
	
	if(p_data){ *p_data = data; }
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__
void scatterBlockData(const scalar_t data, I4 pos, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	scalar_t* p_data = getBlockCellDataPtr<scalar_t>(pos, p_block, domain, type);
	
	if(p_data){ atomicAdd(p_data, data); }
}
#endif //WITH_GRAD

template<typename scalar_t>
__device__
BoundaryConditionType getFixedBoundaryType(const I4 &pos, const index_t bound, const BlockGPU<scalar_t> *p_block, const GridDataType type){ //, const DomainGPU<scalar_t> &domain
	switch(p_block->boundaries[bound].type){
		case BoundaryType::VALUE:
		case BoundaryType::DIRICHLET_VARYING:
			return BoundaryConditionType::DIRICHLET;
		case BoundaryType::GRADIENT:
			return BoundaryConditionType::NEUMANN;
		case BoundaryType::FIXED:
		{
			const index_t typeIndex = gridDataTypeToIndex(type);
			const FixedBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].fb);
			const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->data[typeIndex]);
			if(p_data->isStaticType){
				return p_data->boundaryType;
			} else {
				return p_data->p_boundaryTypes[pos.w];
			}
			break;
		}
		default:
			return BoundaryConditionType::DIRICHLET;
	}

}
template<typename scalar_t>
__device__
scalar_t getFixedBoundaryData(const I4 &pos, const index_t bound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	
	if(type==GridDataType::IS_FIXED_BOUNDARY) { return 1; }
	if(isGradDataType(type)) { return 0; } // TODO
	
	const GridDataType baseType = gridDataTypeToBaseType(type);
	
	switch(p_block->boundaries[bound].type){
		case BoundaryType::VALUE:
		{
			const StaticDirichletBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].sdb);
			switch(baseType){
				case GridDataType::VELOCITY:
					return p_bound->velocity.a[pos.w];
				case GridDataType::PRESSURE:
					return p_block->pressure[flattenIndex(pos, p_block)]; // assumes 0 pressure gradient at boundary
				case GridDataType::PASSIVE_SCALAR:
					return p_bound->scalar;
				default:
					return 0;
			}
		}
		case BoundaryType::DIRICHLET_VARYING:
		{
			const index_t dim = axisFromBound(bound);
			I4 boundPos = pos;
			if(isScalarDataType(type)) { boundPos.w = 0; }
			boundPos.a[dim] = 0;
			
			const VaryingDirichletBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].vdb);
			const index_t flatBoundPos = flattenIndex(boundPos, p_bound->stride);
			switch(baseType){
				case GridDataType::VELOCITY:
					return p_bound->velocity[flatBoundPos];
				case GridDataType::PRESSURE:
					return p_block->pressure[flattenIndex(pos, p_block)]; // assumes 0 pressure gradient at boundary
				case GridDataType::PASSIVE_SCALAR:
					return p_bound->scalar[flatBoundPos];
				default:
					return 0;
			}
		}
		case BoundaryType::GRADIENT: //TODO: how to handle this case here?
			// not implemented
			return 0;
		case BoundaryType::FIXED:
		{
			const index_t typeIndex = gridDataTypeToIndex(type);
			const FixedBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].fb);
			const FixedBoundaryDataGPU<scalar_t> *p_data = &(p_bound->data[typeIndex]);
			if(p_data->isStatic){
				//return (isScalarDataType(type) ? p_data->data[0] : p_data->data[pos.w]);
				return p_data->data[pos.w];
			} else {
				const index_t dim = axisFromBound(bound);
				I4 boundPos = pos;
				//if(isScalarDataType(type)) { boundPos.w = 0; } // passive scalar can have multiple channels now
				boundPos.a[dim] = 0;
				const index_t flatBoundPos = flattenIndex(boundPos, p_bound->stride);
				
				return p_data->data[flatBoundPos];
			}
			break;
		}
		default:
			return 0;
	}
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__
void scatterFixedBoundaryData(const scalar_t data, const I4 &pos, const index_t bound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	
	if(type==GridDataType::IS_FIXED_BOUNDARY || !(p_block->boundaries[bound].type==BoundaryType::FIXED)) {
		return;
	}
	
	const index_t typeIndex = gridDataTypeToIndex(type);
	const FixedBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].fb);
	const FixedBoundaryDataGPU<scalar_t> *p_dataType = &(p_bound->data[typeIndex]);
	
	index_t offset = 0;
	if(p_dataType->isStatic){
		//offset = isScalarDataType(type) ? 0 : pos.w; // passive scalar can have multiple channels now
		offset = pos.w;
	} else {
		const index_t dim = axisFromBound(bound);
		I4 boundPos = pos;
		//if(isScalarDataType(type)) { boundPos.w = 0; }
		boundPos.a[dim] = 0;
		offset = flattenIndex(boundPos, p_bound->stride);
		
	}
	
	scalar_t *p_data = isGradDataType(type) ? p_dataType->grad : p_dataType->data;
	atomicAdd(p_data + offset, data);
}

#endif //WITH_GRAD

/** 
  * If the requested data type is static the spatial position is ignored.
  */
template<typename scalar_t>
__device__
scalar_t *getFixedBoundaryCellDataPtr(I4 pos, const index_t bound, const BlockGPU<scalar_t> *p_block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	
	if(type==GridDataType::IS_FIXED_BOUNDARY) { return nullptr; }
	if(!(p_block->boundaries[bound].type==BoundaryType::FIXED)){ return nullptr; }
	
	/*if(isScalarDataType(type)){ // passive scalar can have multiple channels now
		pos.w = 0;
	}*/
	
	const bool isGrad = isGradDataType(type);
	const index_t typeIndex = gridDataTypeToIndex(type);
	const FixedBoundaryGPU<scalar_t> *p_bound = &(p_block->boundaries[bound].fb);
	const FixedBoundaryDataGPU<scalar_t> *p_dataType = &(p_bound->data[typeIndex]);
	
	index_t offset = pos.w;
	if(!(p_dataType->isStatic)){
		const index_t dim = axisFromBound(bound);
		pos.a[dim] = 0;
		offset = flattenIndex(pos, p_bound->stride);
	}
	
	if(!isGrad){
		return p_dataType->data + offset;
	}
#ifdef WITH_GRAD
	else {
		return p_dataType->grad + offset;
	}
#endif //WITH
	
	return nullptr;
}

template<typename scalar_t>
__device__ scalar_t getBlockDataNeighbor(const I4 &pos, const index_t dir, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const GridDataType type, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/
	
	const index_t dim = axisFromBound(dir);
	const bool isUpper = boundIsUpper(dir);
	const index_t faceSign = faceSignFromBound(dir);
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	
	if(isUpper ? pos.a[dim]==(block.size.a[dim]-1) : pos.a[dim]==0){ //check if there is a boundary in the direction we want to move
		
		switch(block.boundaries[dir].type){
			case BoundaryType::VALUE:
			case BoundaryType::DIRICHLET_VARYING:
			case BoundaryType::FIXED:
			case BoundaryType::GRADIENT:
				return zeroBound ? 0 : getFixedBoundaryData(tempPos, dir, p_block, domain, type);
			case BoundaryType::CONNECTED_GRID:
			{
				//handle multi-block grids, load from correct cell of the connected grid
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[dir].cb.connectedGridIndex;
				tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[dir].cb), domain, 1);
				p_block = p_connectedBlock;
				break;
			}
			case BoundaryType::PERIODIC:
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
				break;
			default:
				return 0;
		}
	} else {
		//same block, just update position
		tempPos.a[dim] += faceSign;
	}
	
	return getBlockData(tempPos, p_block, domain, type);
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__ void scatterBlockDataNeighbor(const scalar_t data, const I4 &pos, const index_t dir, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const GridDataType type, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/
	
	const index_t dim = axisFromBound(dir);
	const bool isUpper = boundIsUpper(dir);
	const index_t faceSign = faceSignFromBound(dir);
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	
	if(isUpper ? pos.a[dim]==(block.size.a[dim]-1) : pos.a[dim]==0){ //check if there is a boundary in the direction we want to move
		
		switch(block.boundaries[dir].type){
			case BoundaryType::VALUE:
			case BoundaryType::DIRICHLET_VARYING:
			case BoundaryType::FIXED:
			case BoundaryType::GRADIENT:
				//if(!zeroBound){ scatterFixedBoundaryData(data, tempPos, dir, p_block, domain, type); }
				return; // TODO: boundary not differentiable
			case BoundaryType::CONNECTED_GRID:
			{
				//handle multi-block grids, load from correct cell of the connected grid
				const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[dir].cb.connectedGridIndex;
				tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[dir].cb), domain, 1);
				p_block = p_connectedBlock;
				break;
			}
			case BoundaryType::PERIODIC:
				// compute flux to cell on other side
				// special case of connection to another block
				tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
				break;
			default:
				return;
		}
	} else {
		//same block, just update position
		tempPos.a[dim] += faceSign;
	}
	
	scatterBlockData(data, tempPos, p_block, domain, type);
}

#endif //WITH_GRAD

template<typename scalar_t>
__device__ scalar_t getBlockDataNeighborDiagonal(const I4 pos, const index_t dir1, const index_t dir2, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const GridDataType type, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/

	const bool dir1Empty = isEmptyBound(dir1, block.boundaries);
	// if dir1 leads to a prescibed boundary check dir2 first. if both are prescribed dir2 will be used
	const index_t dirs[2] = {dir1Empty ? dir2 : dir1, dir1Empty ? dir1 : dir2};
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	for(index_t i=0; i<2; ++i){
		const index_t bound = dirs[i];
		const index_t dim = axisFromBound(bound);
		const bool isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		if(isUpper ? tempPos.a[dim]==(p_block->size.a[dim]-1) : tempPos.a[dim]==0){ //check if there is a boundary in the direction we want to move
			switch(p_block->boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT:
					return zeroBound ? 0 : getFixedBoundaryData(tempPos, bound, p_block, domain, type);
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[bound].cb), domain, 1);
					p_block = p_connectedBlock;
					break;
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
					break;
				default:
					return 0;
			}
		} else {
			//same block, just update position
			tempPos.a[dim] += faceSign;
		}
	}
	
	return getBlockData(tempPos, p_block, domain, type);
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__ void scatterBlockDataNeighborDiagonal(const scalar_t data, const I4 pos, const index_t dir1, const index_t dir2,
		const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const GridDataType type, const bool zeroBound){
	/* pos: original position to start from
	dir: in [0,dim*2] as [-x,+x,-y,..,+z]
	*/

	const bool dir1Empty = isEmptyBound(dir1, block.boundaries);
	// if dir1 leads to a prescibed boundary check dir2 first. if both are prescribed dir2 will be used
	const index_t dirs[2] = {dir1Empty ? dir2 : dir1, dir1Empty ? dir1 : dir2};
	
	const BlockGPU<scalar_t> *p_block = &block;
	I4 tempPos = pos;
	for(index_t i=0; i<2; ++i){
		const index_t bound = dirs[i];
		const index_t dim = axisFromBound(bound);
		const bool isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		if(isUpper ? tempPos.a[dim]==(p_block->size.a[dim]-1) : tempPos.a[dim]==0){ //check if there is a boundary in the direction we want to move
			switch(p_block->boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT:
					//if(!zeroBound){ scatterFixedBoundaryData(data, tempPos, bound, p_block, domain, type);}
					return; // TODO: boundary not differentiable
				case BoundaryType::CONNECTED_GRID:
				{
					//handle multi-block grids, load from correct cell of the connected grid
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + p_block->boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPos<scalar_t>(tempPos, dim, &(p_block->boundaries[bound].cb), domain, 1);
					p_block = p_connectedBlock;
					break;
				}
				case BoundaryType::PERIODIC:
					// compute flux to cell on other side
					// special case of connection to another block
					tempPos.a[dim] = isUpper ? 0 : p_block->size.a[dim]-1;
					break;
				default:
					return;
			}
		} else {
			//same block, just update position
			tempPos.a[dim] += faceSign;
		}
	}
	
	scatterBlockData(data, tempPos, p_block, domain, type);
}

#endif //WITH_GRAD

/** Helper for getCornerValue(). */
template<typename scalar_t>
struct CycleDirection{
	index_t dir1;
	index_t dir2;
	const BlockGPU<scalar_t> *p_block;
	I4 pos;
};
/** Return type of getCornerValue(). */
template<typename scalar_t>
struct CornerValue{
	scalar_t data;
	index_t numCells; // 0 if data is from boundary.
	BoundaryConditionType boundType; // the type of the boundary if the data comes from a boundary
};

/**
 * compute the corner value of a cell by interpolating the adjacent cells
 * can exclude cells, depth 0 for center cell, depth 1 for direct neighbors
 * returns: CornerValue<scalar_t>
 *   .data: interpolated corner value
 *   .numCells: cells used to compute the value. set to 0 to indicate that the value was taken from a fixed boundary.
 */
template <typename scalar_t>
__device__
CornerValue<scalar_t> getCornerValue(const I4 &pos, const index_t dir1, const index_t dir2,
		const bool includeDepth0, const bool includeDepth1, const index_t maxDepth,
		const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const GridDataType type) {
	
	scalar_t data = 0; // accumulated data from included cells
	index_t numCells = 0; // traversed cells, used as divisor
	BoundaryConditionType boundType = BoundaryConditionType::DIRICHLET;
	
	// depth0, the center cell, assumed to be valid
	if(includeDepth0){
		data += getBlockData(pos, &block, domain, type);
	}
	++numCells;
	
	// setup for traversal. go in both directions around the corner. stop if same cell is found, fixed boundary is found, or maxDepth is reached.
	CycleDirection<scalar_t> cycleDir[2] = {
		{.dir1=dir1, .dir2=dir2, .p_block=&block, .pos=pos},
		{.dir1=dir2, .dir2=dir1, .p_block=&block, .pos=pos}
	};
	
	for(index_t depth=1; depth<=maxDepth; ++depth){
		
		for(index_t d=0; d<2; ++d){
			const index_t axis = axisFromBound(cycleDir[d].dir1);
			const index_t faceSign = faceSignFromBound(cycleDir[d].dir1);
			//const index_t isUpper = boundIsUpper(cycleDir[d].dir1);
			const bool atBound = isAtBound(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block);
			
			if(atBound){
				switch(cycleDir[d].p_block->boundaries[cycleDir[d].dir1].type){
					case BoundaryType::VALUE:
						return {.data=getFixedBoundaryData(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type),
								.numCells=0,
								.boundType=BoundaryConditionType::DIRICHLET};
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::FIXED:
					{
						// check if we can interpolate to the corner along the SAME boundary (no further checks for going to adjacent blocks, etc.)
						const bool atBound2 = isAtBound(cycleDir[d].pos, cycleDir[d].dir2, cycleDir[d].p_block);
						boundType = getFixedBoundaryType(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, type);
						if(atBound2){
							// TODO: extrapolate from other direction?
							return {.data=getFixedBoundaryData(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type),
									.numCells=0,
									.boundType=boundType};
						} else {
							I4 neighborPos = cycleDir[d].pos;
							neighborPos.a[axisFromBound(cycleDir[d].dir2)] += faceSignFromBound(cycleDir[d].dir2);
							return {.data=(getFixedBoundaryData(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type)
										+ getFixedBoundaryData(neighborPos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type)) * static_cast<scalar_t>(0.5),
									.numCells=0,
									.boundType=boundType};
						}
						break;
					}
					case BoundaryType::GRADIENT:
						// gradient is given, but this boundary is not (yet) supported.
						return {.data=0, .numCells=0, .boundType=BoundaryConditionType::NEUMANN};
					case BoundaryType::CONNECTED_GRID:
					{
						//handle multi-block grids, go correct cell of the connected grid
						const ConnectedBoundaryGPU<scalar_t> *p_cb = &(cycleDir[d].p_block->boundaries[cycleDir[d].dir1].cb);
						
						cycleDir[d].p_block = domain.blocks + p_cb->connectedGridIndex;
						
						cycleDir[d].pos.a[axis] += faceSign;
						cycleDir[d].pos = computeConnectedPos<scalar_t>(cycleDir[d].pos, axis, p_cb, domain, 1);
						
						// update directions, respecting any shuffling and inversion
						const index_t dir1 = cycleDir[d].dir1;
						cycleDir[d].dir1 = computeConnectedDir(cycleDir[d].dir2, axis, p_cb, domain);
						cycleDir[d].dir2 = invertBound(computeConnectedDir(dir1, axis, p_cb, domain));
						
						break;
					}
					case BoundaryType::PERIODIC:
					{
						const index_t isUpper = boundIsUpper(cycleDir[d].dir1);
						cycleDir[d].pos.a[axis] = isUpper ? 0 : cycleDir[d].p_block->size.a[axis]-1;
						
						const index_t dir1 = cycleDir[d].dir1;
						cycleDir[d].dir1 = cycleDir[d].dir2;
						cycleDir[d].dir2 = invertBound(dir1);
						break;
					}
					default:
						break;
				}
				
			} else {
				// just the neighbor cell in this block
				cycleDir[d].pos.a[axis] += faceSign;
				// update direction for next step to keep going around same corner
				const index_t dir1 = cycleDir[d].dir1;
				cycleDir[d].dir1 = cycleDir[d].dir2;
				cycleDir[d].dir2 = invertBound(dir1);
			}
			
			// check if same cell is found
			const index_t dOther = d^1; // (d+1)%2
			if(cycleDir[d].p_block==cycleDir[dOther].p_block && cycleDir[d].pos==cycleDir[dOther].pos){
				goto returnData; // break double loop
			}
			
			// add cell's data
			if(depth>1 || includeDepth1){
				data += getBlockData(cycleDir[d].pos, cycleDir[d].p_block, domain, type);
			}
			++numCells;
		}
		
	}
	
	returnData:
	return {.data=data / static_cast<scalar_t>(numCells), .numCells=numCells, .boundType=boundType} ;
}

#ifdef WITH_GRAD
/**
 * scatter the gradient of a corner value of a cell to the adjacent cells
 * can exclude cells, depth 0 for center cell, depth 1 for direct neighbors
 * input: CornerValue<scalar_t>
 *   .data: gradient value to scatter
 *   .numCells: cells used to compute the value, as returned by getCornerValue() with the same arguments.
 */
template <typename scalar_t>
__device__
void scatterCornerValue_GRAD(const CornerValue<scalar_t> cVal_grad, const I4 &pos, const index_t dir1, const index_t dir2,
		const bool includeDepth0, const bool includeDepth1, const index_t maxDepth,
		const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const GridDataType type) {
	
	// if cVal_grad.numCells==0 the forward value came from a boundary, so we must not scatter to normal cells here.
	// still have to go through the search to find that boundary.
	const bool isValueFromCells = cVal_grad.numCells>0;
	
	scalar_t data_grad = cVal_grad.data;
	if(isValueFromCells){ 
		data_grad /= static_cast<scalar_t>(cVal_grad.numCells);
	}
	
	// depth0, the center cell, assumed to be valid
	if(includeDepth0){
		//data += getBlockData(pos, &block, domain, type);
		scatterBlockData<scalar_t>(data_grad, pos, &block, domain, type);
	}
	
	// setup for traversal. go in both directions around the corner. stop if same cell is found, fixed boundary is found, or maxDepth is reached.
	CycleDirection<scalar_t> cycleDir[2] = {
		{.dir1=dir1, .dir2=dir2, .p_block=&block, .pos=pos},
		{.dir1=dir2, .dir2=dir1, .p_block=&block, .pos=pos}
	};
	
	for(index_t depth=1; depth<=maxDepth; ++depth){
		
		for(index_t d=0; d<2; ++d){
			const index_t axis = axisFromBound(cycleDir[d].dir1);
			const index_t faceSign = faceSignFromBound(cycleDir[d].dir1);
			//const index_t isUpper = boundIsUpper(cycleDir[d].dir1);
			const bool atBound = isAtBound(cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block);
			
			if(atBound){
				switch(cycleDir[d].p_block->boundaries[cycleDir[d].dir1].type){
					case BoundaryType::FIXED:
					{
						// check if we can interpolate to the corner along the SAME boundary (no further checks for going to adjacent blocks, etc.)
						const bool atBound2 = isAtBound(cycleDir[d].pos, cycleDir[d].dir2, cycleDir[d].p_block);
						if(atBound2){
							// TODO: extrapolate from other direction?
							scatterFixedBoundaryData(data_grad, cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type);
						} else {
							I4 neighborPos = cycleDir[d].pos;
							neighborPos.a[axisFromBound(cycleDir[d].dir2)] += faceSignFromBound(cycleDir[d].dir2);
							data_grad *= static_cast<scalar_t>(0.5);
							scatterFixedBoundaryData(data_grad, cycleDir[d].pos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type);
							scatterFixedBoundaryData(data_grad, neighborPos, cycleDir[d].dir1, cycleDir[d].p_block, domain, type);
						}
						return;
					}
					case BoundaryType::CONNECTED_GRID:
					{
						//handle multi-block grids, go correct cell of the connected grid
						const ConnectedBoundaryGPU<scalar_t> *p_cb = &(cycleDir[d].p_block->boundaries[cycleDir[d].dir1].cb);
						
						cycleDir[d].p_block = domain.blocks + p_cb->connectedGridIndex;
						
						cycleDir[d].pos.a[axis] += faceSign;
						cycleDir[d].pos = computeConnectedPos<scalar_t>(cycleDir[d].pos, axis, p_cb, domain, 1);
						
						// update directions, respecting any shuffling and inversion
						const index_t dir1 = cycleDir[d].dir1;
						cycleDir[d].dir1 = computeConnectedDir(cycleDir[d].dir2, axis, p_cb, domain);
						cycleDir[d].dir2 = invertBound(computeConnectedDir(dir1, axis, p_cb, domain));
						
						break;
					}
					case BoundaryType::PERIODIC:
					{
						const index_t isUpper = boundIsUpper(cycleDir[d].dir1);
						cycleDir[d].pos.a[axis] = isUpper ? 0 : cycleDir[d].p_block->size.a[axis]-1;
						
						const index_t dir1 = cycleDir[d].dir1;
						cycleDir[d].dir1 = cycleDir[d].dir2;
						cycleDir[d].dir2 = invertBound(dir1);
						break;
					}
					default:
						break;
				}
				
			} else {
				// just the neighbor cell in this block
				cycleDir[d].pos.a[axis] += faceSign;
				// update direction for next step to keep going around same corner
				const index_t dir1 = cycleDir[d].dir1;
				cycleDir[d].dir1 = cycleDir[d].dir2;
				cycleDir[d].dir2 = invertBound(dir1);
			}
			
			// check if same cell is found
			const index_t dOther = d^1; // (d+1)%2
			if(cycleDir[d].p_block==cycleDir[dOther].p_block && cycleDir[d].pos==cycleDir[dOther].pos){
				return; // break double loop
			}
			
			// scatter cell's data
			if(isValueFromCells && (depth>1 || includeDepth1)){
				//data += getBlockData(cycleDir[d].pos, cycleDir[d].p_block, domain, type);
				scatterBlockData(data_grad, cycleDir[d].pos, cycleDir[d].p_block, domain, type);
			}
		}
		
	}
}



#endif //WITH_GRAD

template <typename scalar_t, index_t DIMS>
__device__
Vector<scalar_t,DIMS> getBlockDataGradient(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain, const GridDataType type){
	Vector<scalar_t,DIMS> dataGrad = {.a={0}};
	for(index_t dim=0; dim<DIMS; ++dim){
		//getBlockDataNeighbor might return boundary values, which are defined on faces, not cell centers,
		// thus the finite difference using it directly would be wrong
		scalar_t distance = 2.0;
		for(index_t isUpper=0; isUpper<2; ++isUpper){
			index_t boundDir = (dim<<1) + isUpper;
			scalar_t value = 0;
			
			const CellInfo<scalar_t> cellInfo = resolveNeighborCell(pos, boundDir, &block, domain).cell; //don't need axis mapping, just the value and boundary information
			if(cellInfo.isBlock){
				value = getBlockData(cellInfo.pos, cellInfo.p_block, domain, type);
			} else {
				BoundaryConditionType boundType = 
					cellInfo.p_bound->data[gridDataTypeToIndex(type)].isStaticType ?
						cellInfo.p_bound->data[gridDataTypeToIndex(type)].boundaryType :
						cellInfo.p_bound->data[gridDataTypeToIndex(type)].p_boundaryTypes[pos.w];
						
				if(boundType==BoundaryConditionType::DIRICHLET){
					value = getFixedBoundaryData(pos, boundDir, &block, domain, type);
					distance -= 0.5;
				}else{ // NEUMANN, ignore boundary and use one-sided difference
					value = getBlockData(pos, &block, domain, type);
					distance -= 1.0;
				}
			}
			
			dataGrad.a[dim] += (isUpper*2 -1) * value;
		}
		dataGrad.a[dim] /= distance;
	}
	
	if(block.hasTransform){
		I4 tempPos = pos;
		tempPos.w = 0;
		const TransformGPU<scalar_t, DIMS> *T = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(block.transform) + flattenIndex(tempPos, block);
		//pressureGrad = matmul(T->Minv, pressureGrad);
		dataGrad = matmul(dataGrad, T->Minv);
	}
	
	return dataGrad;
}


/**
 * Face-based
 * assumes the Laplace term is added/positive on the LHS
 */
template <typename scalar_t, index_t DIMS>
__device__
scalar_t getNonOrthoLaplaceRHS_v2(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const int8_t nonOrthoFlags, const GridDataType type, const bool withViscosity, const bool withA, const bool useFaceTransform){
	
	if(!((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) || (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS))) { return 0;}
	
	scalar_t S = 0;
	
	const bool forPassiveScalar = gridDataTypeToBaseType(type)==GridDataType::PASSIVE_SCALAR;
	
	const Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> alphasP = getLaplaceCoefficientsFull<scalar_t, DIMS>(pos, &block);
	I4 posScalar = pos;
	posScalar.w = 0;
	const index_t flatPosScalarGlobal = flattenIndexGlobal(posScalar, block, domain); //block.globalOffset + flattenIndex(pos, block);
	const scalar_t one = static_cast<scalar_t>(1.0);
	const scalar_t raP = (withA ? one /domain.Adiag[flatPosScalarGlobal] : one) * (withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, forPassiveScalar, pos.w) : one);
	

	for(index_t face=0; face<(domain.numDims*2); ++face){
		const index_t axis = axisFromBound(face);
		const bool isUpper = boundIsUpper(face);
		const index_t faceSign = faceSignFromBound(face);

		const bool atBound = isAtBound(pos, face, &block); //isUpper ? pos.a[axis]==(block.size.a[axis]-1) : pos.a[axis]==0; // face is at a boundary
		const bool prescribedBound = atBound && isEmptyBound(face, block.boundaries);

		// face tangent direction(s), the non-orthogonal parts

		if(prescribedBound){
			// pressure boundaries are fixed to grad=0
			//if(gridDataTypeToBaseType(type)==GridDataType::PRESSURE){ continue; }
			if(getFixedBoundaryType(pos, face, &block, type)==BoundaryConditionType::NEUMANN || gridDataTypeToBaseType(type)==GridDataType::PRESSURE){ continue;}
			
			Vector<scalar_t, DIMS> boundAlpha = {.a={0}};
			if(useFaceTransform){
				boundAlpha = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block);
			}else if(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING){
				boundAlpha = getLaplaceCoefficientsNeighbourBoundary<scalar_t, DIMS, VaryingDirichletBoundaryGPU<scalar_t>>(pos, axis, &(block.boundaries[face].vdb));
			}else if(block.boundaries[face].type==BoundaryType::FIXED){
				boundAlpha = getLaplaceCoefficientsNeighbourBoundary<scalar_t, DIMS, FixedBoundaryGPU<scalar_t>>(pos, axis, &(block.boundaries[face].fb));
			}
			//Adiag does not exist at bounds, but is only needed for pressure which doesn't reach here.
			
			for(index_t dim=1; dim<DIMS; ++dim){
				const index_t tAxis = (axis+dim)%DIMS;
				// check both ends of the tangent axis
				//const index_t tFaceLower = tAxis<<1;
				//const index_t tFaceUpper = tFaceLower + 1;
				const bool tLowerAtBound = pos.a[tAxis]==0;
				const bool tUpperAtBound = pos.a[tAxis]==(block.size.a[tAxis]-1);
				// due to minimum resolution the cell can't be at 2 opposing boundaries.
				
				switch(block.boundaries[face].type){
					case BoundaryType::VALUE:
						// gradient along boundary is 0, unless this cell is also at a connected border in tangent direction
						if(tUpperAtBound || tLowerAtBound){
							// TODO: check tangential boundaries?
						} else {
							// gradient is 0, noting to add.
						}
						break;
					case BoundaryType::DIRICHLET_VARYING:
					case BoundaryType::FIXED:
						// gradient along boundary may be non-zero.
					{
						// TODO: check possible connected boundaries at tangential faces?
						I4 lowerPos = pos;
						I4 upperPos = pos;
						scalar_t distanceFactor = 0.5;
						if(!tLowerAtBound) {
							lowerPos.a[tAxis] -= 1;
						}
						if(!tUpperAtBound) {
							upperPos.a[tAxis] += 1;
						}
						if(tLowerAtBound || tUpperAtBound) {
							distanceFactor = 1.0; // one-sided difference if one side is not available for central difference.
						}
						const scalar_t tDataGrad = distanceFactor * (getFixedBoundaryData(upperPos, face, &block, domain, type) - getFixedBoundaryData(lowerPos, face, &block, domain, type));
						const scalar_t boundViscosity = withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, forPassiveScalar, pos.w) : one;
						S -= faceSign * boundAlpha.a[tAxis] * tDataGrad * boundViscosity *
							(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING ? (1 - block.boundaries[face].vdb.slip) : 1 ); // * domain.viscosity
						break;
					}
					case BoundaryType::GRADIENT:
						// gradient is given, but this boundary is not (yet) supported.
						break;
					default:
						break;
				}
			}
			continue; // next face
		} else {
			
			// face is not prescribed, neighbor cell exists (but may be over a block connection)
			
			const NeighborCellInfo<scalar_t> neighborInfo = resolveNeighborCell<scalar_t>(pos, face, &block, domain);
			Vector<scalar_t, DIMS> alphasN = {.a={0}};
			if(useFaceTransform){
				alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block);
			}else{
				alphasN = getLaplaceCoefficientsSingleAxisNeighbor<scalar_t, DIMS>(neighborInfo, face, domain);
			}
			
			const index_t flatPosGlobalN = flattenIndexGlobal(neighborInfo.cell, domain);
			const scalar_t raN = (withA ? one /domain.Adiag[flatPosGlobalN] : one)
				* (withViscosity ? getViscosityBlock<scalar_t>(neighborInfo.cell.pos, neighborInfo.cell.p_block, domain, forPassiveScalar, pos.w) : one);
			
			
			for(index_t dim=1; dim<DIMS; ++dim){
				const index_t tAxis = (axis+dim)%DIMS;
				scalar_t faceAlpha = 0;
				if(useFaceTransform){
					faceAlpha = alphasN.a[tAxis]*(raP + raN)*0.5;
				}else{
					faceAlpha = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, tAxis)]*raP + alphasN.a[tAxis]*raN )*0.5;
				}
				//const scalar_t faceAlpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, face, tAxis, DIMS);
				scalar_t tDataGrad = 0;
				for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){
					const index_t tFace = axisToBound(tAxis, tIsUpper);//(tAxis<<1) + tIsUpper;
					const index_t tFaceSign = faceSignFromBound(tFace);
					//const bool tAtBound = isAtBound(pos, tFace, &block);
					
					CornerValue<scalar_t> cVal = getCornerValue(pos, face, tFace,
						false, nonOrthoFlags & NON_ORTHO_DIRECT_RHS, 2,
						block, domain, type);
					const bool cornerAtBound = cVal.numCells==0;
					//const bool boundIsGradient = gridDataTypeToBaseType(type)==GridDataType::PRESSURE; // TODO: get from bound with FIXED boundary implementation
					const bool boundIsGradient = cVal.boundType==BoundaryConditionType::NEUMANN || gridDataTypeToBaseType(type)==GridDataType::PRESSURE;
					
					if(cornerAtBound && boundIsGradient){
						// the influence of this is ~0 if the grid is orthogonal at the boundary.
						// simple handling: ignore the gradient boundary condition and use one sided difference from other side (other side can't be at boundary).
						// other face adds corner interpolation by default, so this is calculated to turn that into the one-sided difference.
						//const scalar_t gradScale = 0.5;
						const index_t tOtherFace = invertBound(tFace);
						const index_t tOtherFaceSign = -tFaceSign;
						if(nonOrthoFlags & NON_ORTHO_DIRECT_RHS){
							tDataGrad += tFaceSign * getBlockDataNeighbor(pos, face, block, domain, type, false) * 0.75;
							tDataGrad += tOtherFaceSign * getBlockDataNeighbor(pos, tOtherFace, block, domain, type, false) * 0.25;
						}
						if(nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS){
							tDataGrad += tOtherFaceSign * getBlockDataNeighborDiagonal(pos, face, tOtherFace, block, domain, type, false) * 0.25;
						}
					} else {
						tDataGrad += tFaceSign * cVal.data; // TODO *0.5?
					}
				}
				
				S -= faceSign * faceAlpha * tDataGrad; // * domain.viscosity;
			}
			
		}
	}
	
	return S;
}
template <typename scalar_t>
__device__
scalar_t getNonOrthoLaplaceRHSDimSwitch_v2(const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const int8_t nonOrthoFlags, const GridDataType type, const bool withViscosity, const bool withA, const bool useFaceTransform){
	switch(domain.numDims){
	//case 1:
		//return 0; // 1D can't be non-orthogonal
	case 2:
		return getNonOrthoLaplaceRHS_v2<scalar_t, 2>(pos, block, domain, nonOrthoFlags, type, withViscosity, withA, useFaceTransform);
	case 3:
		return getNonOrthoLaplaceRHS_v2<scalar_t, 3>(pos, block, domain, nonOrthoFlags, type, withViscosity, withA, useFaceTransform);
	default:
		return 0;
	}
}

#ifdef WITH_GRAD

/**
 * Face-based
 * assumes the Laplace term is added/positive on the LHS
 */
template <typename scalar_t, index_t DIMS>
__device__
void scatterNonOrthoLaplaceRHS_v2_GRAD(const scalar_t S_grad, const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const int8_t nonOrthoFlags, const GridDataType type, const bool withViscosity, const bool withA, const bool useFaceTransform){
	
	if(!((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) || (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS))) { return;}
	
	const bool forPassiveScalar = gridDataTypeToBaseType(type)==GridDataType::PASSIVE_SCALAR;
	const GridDataType fwdType = gridDataTypeWithoutGrad(type);

	const Vector<scalar_t, getLaplaceCoefficientsFullLength(DIMS)> alphasP = getLaplaceCoefficientsFull<scalar_t, DIMS>(pos, &block);
	I4 posScalar = pos;
	posScalar.w = 0;
	const index_t flatPosScalarGlobal = flattenIndexGlobal(posScalar, block, domain); //block.globalOffset + flattenIndex(pos, block);
	const scalar_t one = static_cast<scalar_t>(1.0);
	const scalar_t raP = (withA ? one /domain.Adiag[flatPosScalarGlobal] : one);
	const scalar_t viscP = (withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, forPassiveScalar, pos.w) : one);
	const scalar_t raviscP = raP * viscP;
	const bool withDiffableDataFactor = withViscosity || withA;
	scalar_t raP_grad = 0; //raviscP_grad

	

	for(index_t face=0; face<(domain.numDims*2); ++face){
		const index_t axis = axisFromBound(face);
		const bool isUpper = boundIsUpper(face);
		const index_t faceSign = faceSignFromBound(face);

		const bool atBound = isAtBound(pos, face, &block); //isUpper ? pos.a[axis]==(block.size.a[axis]-1) : pos.a[axis]==0; // face is at a boundary
		const bool prescribedBound = atBound && isEmptyBound(face, block.boundaries);
		
		if(prescribedBound){
			if(block.boundaries[face].type!=BoundaryType::FIXED // only FixedBoundary boundaries are differentiable
					|| getFixedBoundaryType(pos, face, &block, type)==BoundaryConditionType::NEUMANN
					|| gridDataTypeToBaseType(type)==GridDataType::PRESSURE){ // pressure boundaries are fixed to grad=0
				continue;
			}
			
			Vector<scalar_t, DIMS> boundAlpha = {.a={0}};
			if(useFaceTransform){
				boundAlpha = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block);
			}else{
				boundAlpha = getLaplaceCoefficientsNeighbourBoundary<scalar_t, DIMS, FixedBoundaryGPU<scalar_t>>(pos, axis, &(block.boundaries[face].fb));
			}
			//Adiag does not exist at bounds, but is only needed for pressure which doesn't reach here.
			
			for(index_t dim=1; dim<DIMS; ++dim){
				const index_t tAxis = (axis+dim)%DIMS;
				// check both ends of the tangent axis
				const bool tLowerAtBound = pos.a[tAxis]==0;
				const bool tUpperAtBound = pos.a[tAxis]==(block.size.a[tAxis]-1);
				// due to minimum resolution the cell can't be at 2 opposing boundaries.
				
				I4 lowerPos = pos;
				I4 upperPos = pos;
				scalar_t distanceFactor = 0.5;
				if(!tLowerAtBound) {
					lowerPos.a[tAxis] -= 1;
				}
				if(!tUpperAtBound) {
					upperPos.a[tAxis] += 1;
				}
				if(tLowerAtBound || tUpperAtBound) {
					distanceFactor = 1.0; // one-sided difference if one side is not available for central difference.
				}
				
				const scalar_t tDataGrad_grad = - faceSign * boundAlpha.a[tAxis] * S_grad * distanceFactor *
					(block.boundaries[face].type==BoundaryType::DIRICHLET_VARYING ? (1 - block.boundaries[face].vdb.slip) : 1 );
				
				if(withViscosity){
					const scalar_t tDataGrad = (getFixedBoundaryData(upperPos, face, &block, domain, fwdType) - getFixedBoundaryData(lowerPos, face, &block, domain, fwdType));
					scatterViscosityBoundary_GRAD<scalar_t>(tDataGrad_grad * tDataGrad, pos, &(block.boundaries[face].fb), &block, domain, forPassiveScalar, pos.w);
				}
				
				const scalar_t boundViscosity = withViscosity ? getViscosityBlock<scalar_t>(pos, &block, domain, forPassiveScalar, pos.w) : one;
				scatterFixedBoundaryData(tDataGrad_grad * boundViscosity, upperPos, face, &block, domain, type);
				scatterFixedBoundaryData(-tDataGrad_grad * boundViscosity, lowerPos, face, &block, domain, type);
			}
			continue; // next face
		} else {
			// face is not prescribed, neighbor cell exists (but may be over a block connection)
			
			const NeighborCellInfo<scalar_t> neighborInfo = resolveNeighborCell<scalar_t>(pos, face, &block, domain);
			Vector<scalar_t, DIMS> alphasN = {.a={0}};
			if(useFaceTransform){
				alphasN = getLaplaceCoefficientsSingleFace<scalar_t, DIMS>(pos, face, &block);
			}else{
				alphasN = getLaplaceCoefficientsSingleAxisNeighbor<scalar_t, DIMS>(neighborInfo, face, domain);
			}
			
			const index_t flatPosGlobalN = flattenIndexGlobal(neighborInfo.cell, domain);
			const scalar_t raN = (withA ? one /domain.Adiag[flatPosGlobalN] : one);
			const scalar_t viscN = (withViscosity ? getViscosityBlock<scalar_t>(neighborInfo.cell.pos, neighborInfo.cell.p_block, domain, forPassiveScalar, pos.w) : one);
			const scalar_t raviscN = raN * viscN;
			scalar_t raN_grad = 0;
			
			for(index_t dim=1; dim<DIMS; ++dim){
				const index_t tAxis = (axis+dim)%DIMS;
				scalar_t faceAlpha = 0;
				if(useFaceTransform){
					faceAlpha = alphasN.a[tAxis]*(raviscP + raviscN)*0.5;
				}else{
					faceAlpha = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, tAxis)]*raviscP + alphasN.a[tAxis]*raviscN )*0.5;
				}
				
				//if(abs(faceAlpha) < static_cast<scalar_t>(1e-5)) { continue; }
				
				// dS / d tDataGrad: S -= faceSign * faceAlpha * tDataGrad
				const scalar_t tDataGrad_grad = -S_grad * faceSign * faceAlpha;
				
				scalar_t tDataGrad = 0;
				for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){
					const index_t tFace = axisToBound(tAxis, tIsUpper);//(tAxis<<1) + tIsUpper;
					const index_t tFaceSign = faceSignFromBound(tFace);
					//const bool tAtBound = isAtBound(pos, tFace, &block);
					
					CornerValue<scalar_t> cVal_grad = getCornerValue<scalar_t>(pos, face, tFace,
						false, nonOrthoFlags & NON_ORTHO_DIRECT_RHS, 2,
						block, domain, fwdType); //GridDataType::IS_FIXED_BOUNDARY);
					const bool cornerAtBound = cVal_grad.numCells==0;
					const bool boundIsGradient = cVal_grad.boundType==BoundaryConditionType::NEUMANN || gridDataTypeToBaseType(type)==GridDataType::PRESSURE; // TODO: get from bound with FIXED boundary implementation
					
					if(cornerAtBound && boundIsGradient){
						// the influence of this is ~0 if the grid is orthogonal at the boundary.
						// simple handling: ignore the gradient boundary condition and use one sided difference from other side (other side can't be at boundary).
						// other face adds corner interpolation by default, so this is calculated to turn that into the one-sided difference.
						//const scalar_t gradScale = 0.5;
						const index_t tOtherFace = invertBound(tFace);
						const index_t tOtherFaceSign = -tFaceSign;
						if(nonOrthoFlags & NON_ORTHO_DIRECT_RHS){
							if(withDiffableDataFactor) {tDataGrad += tFaceSign * getBlockDataNeighbor(pos, face, block, domain, fwdType, false) * 0.75;}
							scatterBlockDataNeighbor<scalar_t>(tDataGrad_grad * tFaceSign * static_cast<scalar_t>(0.75), pos, face, block, domain, type, false);
							if(withDiffableDataFactor) {tDataGrad += tOtherFaceSign * getBlockDataNeighbor(pos, tOtherFace, block, domain, fwdType, false) * 0.25;}
							scatterBlockDataNeighbor<scalar_t>(tDataGrad_grad * tOtherFaceSign * static_cast<scalar_t>(0.25), pos, tOtherFace, block, domain, type, false);
						}
						if(nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS){
							if(withDiffableDataFactor) {tDataGrad += tOtherFaceSign * getBlockDataNeighborDiagonal(pos, face, tOtherFace, block, domain, fwdType, false) * 0.25;}
							scatterBlockDataNeighborDiagonal<scalar_t>(tDataGrad_grad * tOtherFaceSign * static_cast<scalar_t>(0.25),
								pos, face, tOtherFace, block, domain, type, false);
						}
					} else {
						// d tDataGrad / d cVal.data: tDataGrad += tFaceSign * cVal.data;
						if(withDiffableDataFactor) {tDataGrad += tFaceSign * cVal_grad.data;}
						//cVal_grad.data = static_cast<scalar_t>(cVal_grad.numCells);
						cVal_grad.data = tDataGrad_grad * tFaceSign;
						scatterCornerValue_GRAD<scalar_t>(cVal_grad, pos, face, tFace,
							false, nonOrthoFlags & NON_ORTHO_DIRECT_RHS, 2,
							block, domain, type);
					}
				}
				
				//S -= faceSign * faceAlpha * tDataGrad;
				if(withDiffableDataFactor){
					const scalar_t faceAlpha_grad = -S_grad * faceSign * tDataGrad;
					if(useFaceTransform){
						//faceAlpha = alphasN.a[tAxis]*(raP + raN)*0.5;
						raP_grad += alphasN.a[tAxis]*0.5 * faceAlpha_grad;
						raN_grad += alphasN.a[tAxis]*0.5 * faceAlpha_grad;
					}else{
						//faceAlpha = (alphasP.a[getLaplaceCoefficientCenterIndex(axis, tAxis)]*raP + alphasN.a[tAxis]*raN )*0.5;
						raP_grad += alphasP.a[getLaplaceCoefficientCenterIndex(axis, tAxis)]*0.5 * faceAlpha_grad;
						raN_grad += alphasN.a[tAxis]*0.5 * faceAlpha_grad;
					}
				}
				
			}
			if(withViscosity){
				scatterViscosityBlock_GRAD<scalar_t>(raN_grad * raN, neighborInfo.cell.pos, neighborInfo.cell.p_block, domain, forPassiveScalar, pos.w);
				//scatterViscosityBlock_GRAD<scalar_t>(1, neighborInfo.cell.pos, neighborInfo.cell.p_block, domain, forPassiveScalar, pos.w);
			}
			if(withA){
				atomicAdd(domain.Adiag_grad + flatPosGlobalN, raN_grad * viscN * (-raN * raN));
			}
		}
	}
	if(withViscosity){
		scatterViscosityBlock_GRAD<scalar_t>(raP_grad * raP, pos, &block, domain, forPassiveScalar, pos.w);
	}
	if(withA){
		atomicAdd(domain.Adiag_grad + flatPosScalarGlobal, raP_grad * viscP * (-raP * raP));
	}
}
template <typename scalar_t>
__device__
void scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD(const scalar_t data, const I4 &pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain,
		const int8_t nonOrthoFlags, const GridDataType type, const bool withViscosity, const bool withA, const bool useFaceTransform){
	switch(domain.numDims){
	//case 1:
		//break; // 1D can't be non-orthogonal
	case 2:
		scatterNonOrthoLaplaceRHS_v2_GRAD<scalar_t, 2>(data, pos, block, domain, nonOrthoFlags, type, withViscosity, withA, useFaceTransform);
		break;
	case 3:
		scatterNonOrthoLaplaceRHS_v2_GRAD<scalar_t, 3>(data, pos, block, domain, nonOrthoFlags, type, withViscosity, withA, useFaceTransform);
		break;
	default:
		break;
	}
}

template<typename scalar_t>
__device__ void scatterPressureGradToWithBounds(const scalar_t pressureGrad, const I4 pos, const BlockGPU<scalar_t> &block, const DomainGPU<scalar_t> &domain){
	const int flatPos = flattenIndex(pos, block);
	I4 tempPos = pos;
	tempPos.w = 0; //pressure is scalar
	
	for(int dim=0; dim<domain.numDims; ++dim)
	{
		int bound = dim*2;
		
		if(pos.a[dim]<0){// lower boundary
			switch(block.boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					tempPos.a[dim] = 0;
					atomicAdd(block.pressure_grad + flattenIndex(tempPos, block), pressureGrad);
					return;
				case BoundaryType::CONNECTED_GRID:
				{
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					const I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain, -(pos.a[dim]+1));
					atomicAdd(p_connectedBlock->pressure_grad + flattenIndex(otherPos, p_connectedBlock), pressureGrad);
					return;
				}
				case BoundaryType::PERIODIC:
					tempPos.a[dim] += block.size.a[dim];
					atomicAdd(block.pressure_grad + flattenIndex(tempPos, block), pressureGrad);
					return;
				default:
					return;
			}
		}
		
		bound = dim*2 + 1;
		
		if(pos.a[dim]>=block.size.a[dim]){// upper boundary
			switch(block.boundaries[bound].type){
				case BoundaryType::VALUE:
				case BoundaryType::DIRICHLET_VARYING:
				case BoundaryType::FIXED:
				case BoundaryType::GRADIENT: //TODO: how to handle this case here?
					// enforce 0 pressure gradient to avoid changing the prescribed value
					tempPos.a[dim] = block.size.a[dim] - 1;
					atomicAdd(block.pressure_grad + flattenIndex(tempPos, block), pressureGrad);
					return;
				case BoundaryType::CONNECTED_GRID:
				{
					const BlockGPU<scalar_t> *p_connectedBlock = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
					const I4 otherPos = computeConnectedPos(tempPos, dim, &block.boundaries[bound].cb, domain, pos.a[dim]-block.size.a[dim]);
					atomicAdd(p_connectedBlock->pressure_grad + flattenIndex(otherPos, p_connectedBlock), pressureGrad);
					return;
				}
				case BoundaryType::PERIODIC:
					tempPos.a[dim] -= block.size.a[dim];
					atomicAdd(block.pressure_grad + flattenIndex(tempPos, block), pressureGrad);
					return;
				default:
					return;
			}
		}
	}
	atomicAdd(block.pressure_grad + flatPos, pressureGrad);
}

#endif //WITH_GRAD

__device__ int findLowestColumnIndex(int *indices, int size){
	int value = INT_MAX;
	int index = -1;
	for(int i=0;i<size;++i){
		if(indices[i]>=0 && indices[i]<value){
			value = indices[i];
			index = i;
		}
	}
	return index;
}

template<typename T>
__device__ index_t ArrayIndexOf(const T *array, const T value, const index_t size){
	for(int i=0;i<size;++i){
		if(array[i]==value){
			return i;
		}
	}
	return -1;
}

// load row from CSR matrix into csrValues (size must be at least spatialDims*2+1)
// loaded values will be sorted to be in [diag,-x,+x,-y,+y,-z,+z] order
template<typename scalar_t>
__device__
void LoadCSRrowNeighborSorted(const I4 &pos, const CSRmatrixGPU<scalar_t> &csr, const DomainGPU<scalar_t> &domain, const BlockGPU<scalar_t> &block, 
			scalar_t *csrValues){ // to be loaded like: diag,-x,+x,-y,+y,-z,+z
	
	const index_t flatPosGlobal = flattenIndexGlobal(pos, block, domain);
	
	const index_t csrStart = csr.row[flatPosGlobal];
	const index_t csrEnd = csr.row[flatPosGlobal+1];
	const index_t rowSize = csrEnd - csrStart;
	
	// row is sorted by global cell index, not neighbor direction
	index_t csrIndices[7] = {0};
	index_t i = 0;
	for(;i<rowSize && i<7;++i){
		csrIndices[i] = csr.index[csrStart+i];
	}
	for(; i<7;++i){
		csrIndices[i] = -1;
	}
	
	// undo global index sorting
	{ // diagonal entry
		const index_t aIndex = ArrayIndexOf(csrIndices, flatPosGlobal, 7);
		if(aIndex>=0){
			csrValues[0] = csr.value[csrStart+aIndex];
		}
	}
	// neighbor entries
	for(index_t bound=0; bound<(domain.numDims*2); ++bound){
		const index_t dim = axisFromBound(bound);
		const index_t isUpper = boundIsUpper(bound);
		const index_t faceSign = faceSignFromBound(bound);
		
		const bool atBound = isAtBound(pos, bound, block);
		if(!atBound || !isEmptyBound(bound, block.boundaries)){
			//calculate index of neighbour
			I4 tempPos = pos;
			tempPos.w = dim;
			const BlockGPU<scalar_t> *p_block = &block;
			// resolve neighbor cell
			if(atBound && block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
				p_block = domain.blocks + block.boundaries[bound].cb.connectedGridIndex;
				tempPos = computeConnectedPosWithChannel(tempPos, dim, &block.boundaries[bound].cb, domain);
			}else{
				if(atBound && block.boundaries[bound].type==BoundaryType::PERIODIC){
					tempPos.a[dim] = isUpper ? 0 : block.size.a[dim]-1;
				}else{
					tempPos.a[dim] = pos.a[dim] + faceSign;
				}
			}
			tempPos.w = 0; // indices are "scalar"
			const index_t nIndex = flattenIndexGlobal(tempPos, p_block, domain); //flattenIndex(tempPos, p_block) + p_block->globalOffset;
			const index_t aIndex = ArrayIndexOf(csrIndices, nIndex, 7);
			
			if(aIndex>=0){
				csrValues[bound + 1] = csr.value[csrStart+aIndex];
			}
		} // else invalid neighbor
	}
}

#define SETUP_KERNEL_PER_CELL(domain, idxName, offName) \
	int32_t threads; \
	dim3 blocks; \
	index_t *p_##idxName; \
	index_t *p_##offName; \
	std::vector<index_t> idxName; \
	std::vector<index_t> offName; \
	ComputeThreadBlocks(domain, threads, blocks, idxName, offName); \
	torch::Tensor t_blockIdxByThreadBlock = CopyBlockIndices(domain, idxName, offName, p_##idxName, p_##offName);

#define KERNEL_PER_CELL_LOOP(p_domain, p_idx, p_off, num, ...) \
	__shared__ DomainGPU<scalar_t> s_domain; \
	__shared__ BlockGPU<scalar_t> s_block; \
	if(threadIdx.x==0){ s_domain = *p_domain; } \
	__syncthreads(); \
	index_t repetitions = num/gridDim.x; \
	if(blockIdx.x<(num - gridDim.x*repetitions) ){ ++repetitions; } \
	index_t loadedBlockIdx = -1; \
	for(index_t r=0;r<repetitions;++r){ \
		const index_t currentThreadBlockIdx = gridDim.x*r + blockIdx.x; \
		const index_t targetBlockIdx = p_idx[currentThreadBlockIdx];  \
		const index_t threadBlockOffsetInBlock = p_off[currentThreadBlockIdx];  \
		if(threadIdx.x==0 && targetBlockIdx!=loadedBlockIdx){ \
			s_block = s_domain.blocks[targetBlockIdx]; \
			loadedBlockIdx = targetBlockIdx; \
		} \
		__syncthreads(); \
		const index_t flatPos = threadBlockOffsetInBlock*blockDim.x + threadIdx.x; \
		if(flatPos<s_block.stride.w){ \
			__VA_ARGS__ \
	}}


/* --- Advection/Diffusion --- */


template <typename scalar_t>
__global__ void PISO_build_matrix(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	__shared__ DomainGPU<scalar_t> s_domain;
	__shared__ BlockGPU<scalar_t> s_block; //info about current main block

	if(threadIdx.x==0){
		//TODO improve loading?
		s_domain = *p_domain;
	}
	__syncthreads();

	//naive implementation, 1 thread per cell, no sharing
	// each thread/cell builds its row in the matrix
	index_t repetitions = numThreadBlocks/gridDim.x; // ceilDiv(numThreadBlocks, gridDim.x);
	if(blockIdx.x<(numThreadBlocks - gridDim.x*repetitions)){
		++repetitions;
	}
	index_t loadedBlockIdx = -1;
	
	
	for(index_t r=0;r<repetitions;++r){
		const index_t currentThreadBlockIdx = gridDim.x*r + blockIdx.x;
		/*if(currentThreadBlockIdx>=numThreadBlocks){
			return;
		}*/

		const index_t targetBlockIdx = p_blockIdxByThreadBlock[currentThreadBlockIdx];
		const index_t threadBlockOffsetInBlock = p_threadBlockOffsetInBlock[currentThreadBlockIdx];
		if(threadIdx.x==0){
			if(targetBlockIdx!=loadedBlockIdx){
				s_block = s_domain.blocks[targetBlockIdx];
				loadedBlockIdx = targetBlockIdx;
			}
		}
		__syncthreads();

		const index_t flatPos = threadBlockOffsetInBlock*blockDim.x + threadIdx.x;
		if(flatPos<s_block.stride.w){
			const I4 pos = unflattenIndex(flatPos, s_block);
			
			//get fluxes
			scalar_t fluxes[6];
			computeFluxesNDLoop<scalar_t>(pos, fluxes, s_block, s_domain, nullptr);
			
			//compute matrix
			
			//if(flatPos==0){
			//	csrMatrixRow[0] = 0;
			//}
			
			const RowMeta row = getCSRMatrixRowEndOffsetFromBlockBoundaries3D(flatPos, s_block, s_domain);
			int rowStartOffset = row.endOffset - row.size + s_block.csrOffset;
			
			s_domain.C.row[s_block.globalOffset + flatPos+1] = row.endOffset + s_block.csrOffset;
			//csrMatrixIndex[flatPos+1] = row.size;
			// csrMatrixIndex[flatPos] = pos.x;
			// csrMatrixIndex[flatPos + c_domain.stride.w + 1] = pos.y;
			// csrMatrixIndex[flatPos + c_domain.stride.w*2 + 2] = pos.z;
			
			
			// row entries have to be in ascending column order (for cublas)
			// for a default inner cell: -z,-y,-x,diag,+x,+y,+z
			// for boundary cell
			// - if the boundary is open or closed (no connection), the entry is simply removed
			// - if the boundary is periodic, the entry moves to the other end
			//   - grid size 1: the cell connects to itself
			//   - grid size 2: connects to same cell in both directions
			//   - cell at lower z border, domain is z-periodic: -y,-x,diag,+x,+y,+z,-z
			
			// alternative: compute flat indices, sort
			int indices[7]; // diag,-x,+x,-y,+y,-z,+z
			scalar_t rowValues[7] = {0};
			
			const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
			scalar_t diag = det/timeStep; // + static_cast<scalar_t>(s_domain.numDims*2)*s_domain.viscosity; //1/dt;
			indices[0] = flatPos + s_block.globalOffset;
			
			// viscosity
			//const scalar_t viscosity = getViscosity(s_domain, forPassiveScalar, passiveScalarChannel);
			const scalar_t viscosity_p = getViscosityBlock(pos, &s_block, s_domain, forPassiveScalar, passiveScalarChannel);
			//const scalar_t viscosityHalf = static_cast<scalar_t>(0.5)*viscosity; // *0.5 comes from interpolating alpha later
			scalar_t alpha_p[3] = {1,1,1};
			for(index_t dim=0;dim<s_domain.numDims;++dim){
				I4 tempPos = pos;
				tempPos.w = dim;
				alpha_p[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
			}
			
			// for every face: add advective fluxes, compute and add viscosity/diffusion.
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
				const index_t dim = axisFromBound(bound);
				const index_t isUpper = boundIsUpper(bound);
				const index_t faceSign = faceSignFromBound(bound);
				
				const bool atBound = isAtBound(pos, bound, s_block);
				if(!atBound || !isEmptyBound(bound, s_block.boundaries)){
					
					{ // advective fluxes 
						const scalar_t faceFlux = faceSign * 0.5 * fluxes[bound];
						diag += faceFlux;
						rowValues[bound+1] += faceFlux;
					}
					
					{ // orthogonal viscosity
						//calculate index of neighbour
						I4 tempPos = pos;
						tempPos.w = dim;
						scalar_t alpha = 1;
						const BlockGPU<scalar_t> *p_block = &s_block;
						// resolve neighbor cell
						if(atBound && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
							p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
							tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
						}else{
							if(atBound && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
								tempPos.a[dim] = isUpper ? 0 : s_block.size.a[dim]-1;
							}else{
								tempPos.a[dim] = pos.a[dim] + faceSign;
							}
						}
						// need correct component/channel for laplace coefficient
						alpha = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
						tempPos.w = forPassiveScalar ?  passiveScalarChannel : 0;
						const scalar_t viscosity_n = getViscosityBlock(tempPos, p_block, s_domain, forPassiveScalar, passiveScalarChannel);
						tempPos.w = 0; // indices are "scalar"
						indices[bound+1] = flattenIndexGlobal(tempPos, p_block, s_domain); //flattenIndex(tempPos, p_block) + p_block->globalOffset;
						
						//const scalar_t viscosityCoeff = viscosityHalf*(alpha_p[dim] + alpha);
						const scalar_t viscosityCoeff = (alpha_p[dim]*viscosity_p + alpha*viscosity_n) * static_cast<scalar_t>(0.5);
						diag += viscosityCoeff;
						rowValues[bound+1] -= viscosityCoeff;
					}
					
					// laplace non-orthogonal coefficients from face tangential directions
					const bool includeNonOrthoNeighbors = nonOrthoFlags & NON_ORTHO_DIRECT_MATRIX;
					const bool includeNonOrthoDiag = nonOrthoFlags & NON_ORTHO_CENTER_MATRIX;
					if(s_domain.numDims>1 && (includeNonOrthoDiag || includeNonOrthoNeighbors)){
						scalar_t alphaInterp[12]; // size for 3D. contains viscosity
						if(s_domain.numDims==2) interpolateNonOrthoLaplaceComponents<scalar_t, 2>(pos, s_block, s_domain, alphaInterp, !forPassiveScalar, false, false); // currently always uses velocity viscosity
						else if(s_domain.numDims==3) interpolateNonOrthoLaplaceComponents<scalar_t, 3>(pos, s_block, s_domain, alphaInterp, !forPassiveScalar, false, false); // TODO: extend to use correct scalar viscosity
						
						I4 channelPos = pos;
						GridDataType dataType = GridDataType::VELOCITY;
						if(forPassiveScalar){
							channelPos.w = passiveScalarChannel;
							dataType = GridDataType::PASSIVE_SCALAR;
						}
					
						for(index_t i=1; i<s_domain.numDims; ++i){ // loop other axes
							const index_t tAxis = (dim + i)%s_domain.numDims;
							// *viscosity_p is a workaround for global scalar viscosity
							const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims) * (forPassiveScalar ? viscosity_p : static_cast<scalar_t>(1.0));
							
							if(alpha!=0){ // grid is non-orthogonal here
								for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){ // loop corners
									const index_t tFace = axisToBound(tAxis, tIsUpper); //(tAxis<<1) + tIsUpper;
									const index_t tFaceSign = faceSignFromBound(tFace);
									const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(channelPos, bound, tFace,
										false, false, 2, s_block, s_domain, dataType); // computes interpolation divisor and checks for boundaries.
									const bool cornerAtBound = cVal.numCells<1;
									const bool boundIsGradient = cVal.boundType==BoundaryConditionType::NEUMANN;
									
									if(cornerAtBound){
										if(boundIsGradient){
											// simplified treatment: ignore boundary gradient and use one-sided difference from other side
											const scalar_t interpolationNorm = 0.25; // from other side, can't be anything else but 1/4
											const index_t tFaceOther = invertBound(tFace);
											const scalar_t viscosityCoeff = faceSign * tFaceSign * alpha * interpolationNorm;
											if(includeNonOrthoDiag){
												diag -= 3 * viscosityCoeff;
											}
											if(includeNonOrthoNeighbors){
												rowValues[bound+1] -= 3 * viscosityCoeff;
												rowValues[tFaceOther+1] += viscosityCoeff;
											}
											// if diagonals where to be included in the matrix: 
											// rowValues[diagonalOther] += viscosityCoeff;
										}
										// else: prescribed value, added on RHS, nothing to do here.
									} else {
										// normal corner with interpolated value
										const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
										const scalar_t viscosityCoeff = faceSign * tFaceSign * alpha * interpolationNorm;
										if(includeNonOrthoDiag){
											diag -= viscosityCoeff; // TODO: is this correct?
										}
										if(includeNonOrthoNeighbors){
											rowValues[bound+1] -= viscosityCoeff; // TODO: is this correct?
											rowValues[tFace+1] -= viscosityCoeff;
										}
										// if diagonals were to be included in the matrix: 
										// rowValues[diagonal] -= viscosityCoeff;
									}
								}
							}
						}
					}
				}else{ // face is a prescribed boundary
					//  TODO: missing non-orthogonal coefficients?
					index_t slip = 0;
					switch(s_block.boundaries[bound].type){
						case BoundaryType::DIRICHLET:
						{
							slip = s_block.boundaries[bound].sdb.slip;
							break;
						}
						case BoundaryType::DIRICHLET_VARYING:
						{
							slip = s_block.boundaries[bound].vdb.slip;
							break;
						}
						case BoundaryType::FIXED:
						{
							// TODO: type based on what the matrix is for.
							BoundaryConditionType boundaryType; // = s_block.boundaries[bound].fb.velocity.boundaryType;
							if(forPassiveScalar){
								I4 tempPos = pos;
								tempPos.w = passiveScalarChannel;
								boundaryType = getFixedBoundaryType(tempPos, bound, &s_block, GridDataType::PASSIVE_SCALAR);
							}else{
								boundaryType = s_block.boundaries[bound].fb.velocity.boundaryType;
							}
							slip = boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
							break;
						}
					}
					diag += (1 - slip) * 2 * viscosity_p * alpha_p[dim];
					indices[bound+1] = -1; //invalid/unused
				}
			}
			
			
			for(int dim=s_domain.numDims;dim<3;++dim){
				indices[dim*2+1] = -1; //invalid/unused
				indices[dim*2+2] = -1; //invalid/unused
			}

			
			//TODO
			rowValues[0] = diag;
			
			// sort, naive for now
			for(int i=0;i<row.size;++i){
				int colIndex = findLowestColumnIndex(indices, 7);
				rowValues[colIndex] /= det;
				
				if(colIndex<0){
					s_domain.C.index[rowStartOffset + i] = -1;
					s_domain.C.value[rowStartOffset + i] = -1.0f;
				}else{
					s_domain.C.index[rowStartOffset + i] = indices[colIndex];
					s_domain.C.value[rowStartOffset + i] = rowValues[colIndex];
				}
				
				indices[colIndex] = -1;
			}
			
			s_domain.Adiag[flatPos + s_block.globalOffset] = rowValues[0];
		}
	}
}

#ifdef WITH_GRAD

template <typename scalar_t>
__global__ void PISO_build_matrix_GRAD(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	const scalar_t half = static_cast<scalar_t>(0.5);
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
		
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		const scalar_t rDet = 1/det;
		
		// load row from advection CSR matrix grad
		scalar_t csrValuesGrad[7] = {0}; // to be loaded like: diag,-x,+x,-y,+y,-z,+z
		LoadCSRrowNeighborSorted(pos, s_domain.C_grad, s_domain, s_block, csrValuesGrad);
		for(index_t i=0; i<(s_domain.numDims*2+1); ++i){
			csrValuesGrad[i] *= rDet;
		}
		
		// add gradient from separate diagonal
		//csrValuesGrad[0] += s_domain.Adiag_grad[flatPosGlobal];
		scalar_t diagGrad = s_domain.Adiag_grad[flatPosGlobal] * rDet + csrValuesGrad[0];
		
		
		scalar_t fluxesGrad[7] = {0};
		
		scalar_t viscosity_grad = 0;
		scalar_t alpha_p[3] = {1,1,1};
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			alpha_p[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
		}
		
		
		for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
			const index_t dim = axisFromBound(bound);
			const index_t isUpper = boundIsUpper(bound);
			const index_t faceSign = faceSignFromBound(bound);
			
			const bool atBound = isAtBound(pos, bound, s_block);
			if(!atBound || !isEmptyBound(bound, s_block.boundaries)){
				
				// flux grad and orthogonal diffusion grad
				{
					//calculate index of neighbour
					I4 tempPos = pos;
					tempPos.w = dim;
					scalar_t alpha = 1;
					const BlockGPU<scalar_t> *p_block = &s_block;
					// resolve neighbor cell
					if(atBound && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
						p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					}else{
						if(atBound && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
							tempPos.a[dim] = isUpper ? 0 : s_block.size.a[dim]-1;
						}else{
							tempPos.a[dim] = pos.a[dim] + faceSign;
						}
					}
					// need correct component/channel for laplace coefficient
					alpha = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
					
					fluxesGrad[bound] = faceSign * half * (diagGrad + csrValuesGrad[bound+1]);
					
					const scalar_t viscosityCoeff_grad = diagGrad - csrValuesGrad[bound+1];
					//viscosity_grad += half * (alpha_p[dim] + alpha) * viscosityCoeff_grad;
					viscosity_grad += half * alpha_p[dim] * viscosityCoeff_grad;
					const scalar_t viscosity_n_grad = half * alpha * viscosityCoeff_grad;
					tempPos.w = forPassiveScalar ? passiveScalarChannel : 0;
					scatterViscosityBlock_GRAD<scalar_t>(viscosity_n_grad, tempPos, p_block, s_domain, forPassiveScalar, passiveScalarChannel);
				
				}
				
				// non-ortho diffusion grad
				const bool includeNonOrthoNeighbors = nonOrthoFlags & NON_ORTHO_DIRECT_MATRIX;
				const bool includeNonOrthoDiag = nonOrthoFlags & NON_ORTHO_CENTER_MATRIX;
				if(s_domain.numDims>1 && (includeNonOrthoDiag || includeNonOrthoNeighbors)){
					scalar_t alphaInterp[12]; // size for 3D
					scalar_t alphaInterp_grad[12] = {0};
					if(s_domain.numDims==2) interpolateNonOrthoLaplaceComponents<scalar_t, 2>(pos, s_block, s_domain, alphaInterp, false, false, false);
					else if(s_domain.numDims==3) interpolateNonOrthoLaplaceComponents<scalar_t, 3>(pos, s_block, s_domain, alphaInterp, false, false, false);
					
					I4 channelPos = pos;
					GridDataType dataType = GridDataType::VELOCITY;
					if(forPassiveScalar){
						channelPos.w = passiveScalarChannel;
						dataType = GridDataType::PASSIVE_SCALAR;
					}
					
					for(index_t i=1; i<s_domain.numDims; ++i){ // loop other axes
						const index_t tAxis = (dim + i)%s_domain.numDims;
						const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims);
						
						if(alpha!=0){ // grid is non-orthogonal here
							scalar_t alphaVisc_grad = 0;
							for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){ // loop corners
								const index_t tFace = axisToBound(tAxis, tIsUpper); //(tAxis<<1) + tIsUpper;
								const index_t tFaceSign = faceSignFromBound(tFace);
								const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(channelPos, bound, tFace,
									false, false, 2, s_block, s_domain, dataType); // computes interpolation divisor and checks for boundaries.
								const bool cornerAtBound = cVal.numCells<1;
								const bool boundIsGradient = cVal.boundType==BoundaryConditionType::NEUMANN;
								
								if(cornerAtBound){
									if(boundIsGradient){
										// simplified treatment: ignore boudnary gradient and use one-sided difference from other side
										const scalar_t interpolationNorm = 0.25; // from other side, can't be anything else but 1/4
										const index_t tFaceOther = invertBound(tFace);
										//const scalar_t viscosityCoeff = faceSign * tFaceSign * viscosity * alpha * interpolationNorm;
										scalar_t viscosityCoeff_grad = 0;
										if(includeNonOrthoDiag){
											//diag -= 3 * viscosityCoeff;
											viscosityCoeff_grad -= 3 * diagGrad;
										}
										if(includeNonOrthoNeighbors){
											//rowValues[bound+1] -= 3 * viscosityCoeff;
											viscosityCoeff_grad -= 3 * csrValuesGrad[bound+1];
											//rowValues[tFaceOther+1] += viscosityCoeff;
											viscosityCoeff_grad += csrValuesGrad[tFaceOther+1];
										}
										// if diagonals where to be included in the matrix: 
										// rowValues[diagonalOther] += viscosityCoeff;
										alphaVisc_grad += faceSign * tFaceSign * interpolationNorm * viscosityCoeff_grad;
									}
									// else: prescribed value, added on RHS, nothing to do here.
								} else {
									// normal corner with interpolated value
									const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
									//const scalar_t viscosityCoeff = faceSign * tFaceSign * viscosity * alpha * interpolationNorm;
									scalar_t viscosityCoeff_grad = 0;
									if(includeNonOrthoDiag){
										//diag -= viscosityCoeff;
										viscosityCoeff_grad -= diagGrad;
									}
									if(includeNonOrthoNeighbors){
										//rowValues[bound+1] -= viscosityCoeff;
										viscosityCoeff_grad -= csrValuesGrad[bound+1];
										//rowValues[tFace+1] -= viscosityCoeff;
										viscosityCoeff_grad -= csrValuesGrad[tFace+1];
									}
									// if diagonals where to be included in the matrix: 
									// rowValues[diagonal] -= viscosityCoeff;
									alphaVisc_grad += faceSign * tFaceSign * interpolationNorm * viscosityCoeff_grad;
								}
							}
							if(forPassiveScalar){
								viscosity_grad += alphaVisc_grad * alpha;
							} else {
								//viscosity_grad += alphaVisc_grad * alpha;
								addInterpolatedNonOrthoLaplaceComponent_GRAD(alphaVisc_grad, alphaInterp_grad, bound, tAxis, s_domain.numDims);
							}
						}
					}
					if(!forPassiveScalar){
						if(s_domain.numDims==2) scatterNonOrthoLaplaceComponents_GRAD<scalar_t, 2>(alphaInterp_grad, pos, s_block, s_domain, true, false, false);
						else if(s_domain.numDims==3) scatterNonOrthoLaplaceComponents_GRAD<scalar_t, 3>(alphaInterp_grad, pos, s_block, s_domain, true, false, false);
					}
				}
				
			} else { // face is prescribed boundary
				index_t slip = 0;
				switch(s_block.boundaries[bound].type){
					case BoundaryType::DIRICHLET:
					{
						slip = s_block.boundaries[bound].sdb.slip;
						break;
					}
					case BoundaryType::DIRICHLET_VARYING:
					{
						slip = s_block.boundaries[bound].vdb.slip;
						break;
					}
					case BoundaryType::FIXED:
					{
						BoundaryConditionType boundaryType; // = s_block.boundaries[bound].fb.velocity.boundaryType;
						if(forPassiveScalar){
							I4 tempPos = pos;
							tempPos.w = passiveScalarChannel;
							boundaryType = getFixedBoundaryType(tempPos, bound, &s_block, GridDataType::PASSIVE_SCALAR);
						}else{
							boundaryType = s_block.boundaries[bound].fb.velocity.boundaryType;
						}
						//const BoundaryConditionType boundaryType = s_block.boundaries[bound].fb.velocity.boundaryType;
						slip = boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
						break;
					}
				}
				//diag += (1 + 1 - slip*2) * viscosity * alpha_p[dim];
				viscosity_grad += (1 - slip)*2  * alpha_p[dim] * diagGrad;
				
			}
			
		}
		
		ScatterFluxesGradNDLoop<scalar_t>(pos, fluxesGrad, s_block, s_domain, nullptr);
		{
			I4 tempPos = pos;
			tempPos.w = forPassiveScalar ? passiveScalarChannel : 0;
			scatterViscosityBlock_GRAD<scalar_t>(viscosity_grad, tempPos, &s_block, s_domain, forPassiveScalar, passiveScalarChannel);
		}
	)
}

#endif //WITH_GRAD

template <typename scalar_t>
__global__ void kPISO_build_scalar_advection_RHS(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		//weird issue: memory access violation if flatPos is used directly.
		
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		
		for(index_t channel=0; channel<s_domain.passiveScalarChannels; ++channel){
			I4 tempPos = pos;
			tempPos.w = channel;
			
			//const scalar_t viscosity = getViscosity(s_domain, s_domain.scalarViscosity!=nullptr, channel);
			
			const int tempFlatPos = flattenIndex(tempPos, s_block);
			scalar_t tempRHS = det * s_block.scalarData[tempFlatPos]/timeStep;
			//TODO: pressure, external forces?
			//if(flatPos<27 && s_block.globalOffset==0)
				
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					//const scalar_t isTangentialDir = static_cast<scalar_t>((bound>>1)!=dim); // * 2 -1;
					switch(s_block.boundaries[bound].type){
					case BoundaryType::FIXED:
					{
						I4 boundPos = pos;
						boundPos.w = channel;
						boundPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(boundPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t scalar = getFixedBoundaryData(boundPos, bound, &s_block, s_domain, GridDataType::PASSIVE_SCALAR); // may be value or gradient
						const BoundaryConditionType scalarBoundaryType = getFixedBoundaryType(boundPos, bound, &s_block, GridDataType::PASSIVE_SCALAR);
						
						const FixedBoundaryGPU<scalar_t> *p_fb = &(s_block.boundaries[bound].fb);
						
						boundPos.w = bound>>1;
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(boundPos, p_fb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(boundPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(boundPos, p_fb, s_domain) * faceNormal;
						
						// from advection
						tempRHS -= scalar * flux;
						// from viscosity
						const scalar_t viscosity = getViscosityFixedBoundary(tempPos, p_fb, &s_block, s_domain, true, channel);
						if(scalarBoundaryType==BoundaryConditionType::DIRICHLET){
							tempRHS += scalar * viscosity * 2 * alpha;
						}else{ // NEUMANN
							tempRHS += scalar * viscosity;
						}
						break;
					}
					case BoundaryType::VALUE:
					{
						// from advection
						tempRHS -= s_block.boundaries[bound].sdb.scalar * s_block.boundaries[bound].sdb.velocity.a[bound>>1] * faceNormal;
						// from viscosity
						const scalar_t viscosity = getViscosity(s_domain, s_domain.scalarViscosity!=nullptr, channel);
						tempRHS += (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * s_block.boundaries[bound].sdb.scalar; //* isTangentialDir
						break;
					}
					case BoundaryType::DIRICHLET_VARYING:
					{
						I4 tempPos = pos;
						tempPos.w = 0; // VaryingDirichletBoundary does not support channels
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t scalar = s_block.boundaries[bound].vdb.scalar[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)];
						tempPos.w = bound>>1;
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain) * faceNormal;
						
						// from advection
						tempRHS -= scalar * flux;
						// from viscosity
						const scalar_t viscosity = getViscosity(s_domain, s_domain.scalarViscosity!=nullptr, channel);
						tempRHS += scalar * (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					default:
						break;
					}
				}
			}
			
			
			// getNonOrthoLaplaceRHS assumes the laplace term to be positive/added on the LHS and returns negated coefficients for the RHS.
			// The diffusion term is negative on the LHS, so consequently the RHS has to be subtracted.
			//tempRHS -= getNonOrthoLaplaceRHSDimSwitch<scalar_t>(pos, s_block, s_domain, nonOrthoFlags, GridDataType::PASSIVE_SCALAR_RESULT, false) * s_domain.viscosity;
			const scalar_t viscosity = getViscosityBlock(tempPos, &s_block, s_domain, true, channel);
			tempRHS -= getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(tempPos, s_block, s_domain, nonOrthoFlags, GridDataType::PASSIVE_SCALAR_RESULT, false, false, false) * viscosity;
			
			tempRHS /= det; // in the used formulation, the advection and diffusion parts have to be divided by the determinant. Other source terms are unaffected, so add them below.
			
			const index_t flatPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			s_domain.scalarRHS[flatPosGlobal] = tempRHS;
		}
	)
}

#ifdef WITH_GRAD
template <typename scalar_t>
__global__ void kPISO_build_scalar_advection_RHS_GRAD(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		
		for(index_t channel=0; channel<s_domain.passiveScalarChannels; ++channel){
			I4 tempPos = pos;
			tempPos.w = channel;
			
			const scalar_t viscosity = getViscosity(s_domain, s_domain.scalarViscosity!=nullptr, channel);
			
			const index_t flatPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
		
			scalar_t RHSgrad = s_domain.scalarRHS_grad[flatPosGlobal];
			
			RHSgrad /= det;
			
			// non-ortho laplace gradients w.r.t. scalarResult
			scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD<scalar_t>(-RHSgrad * viscosity, tempPos, s_block, s_domain, nonOrthoFlags,
				GridDataType::PASSIVE_SCALAR_RESULT_GRAD, false, false, false);
			
			// non-ortho laplace gradients w.r.t. viscosity
			
			scalar_t viscosity_grad = -RHSgrad * getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(tempPos, s_block, s_domain,
				nonOrthoFlags, GridDataType::PASSIVE_SCALAR_RESULT, false, false, false);
			
			
			// gradients from/for boundaries
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					if(!(s_block.boundaries[bound].type==BoundaryType::FIXED)){ continue; }
					
					// forward values
					
					I4 tempPos = pos;
					tempPos.w = channel;
					tempPos.a[bound>>1] = 0;
					//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
					const scalar_t scalar = getFixedBoundaryData(tempPos, bound, &s_block, s_domain, GridDataType::PASSIVE_SCALAR);
					const BoundaryConditionType scalarBoundaryType = getFixedBoundaryType(tempPos, bound, &s_block, GridDataType::PASSIVE_SCALAR);
					
					tempPos.w = bound>>1;
					const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(tempPos, &s_block.boundaries[bound].fb, s_domain.numDims);
					//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
					const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(tempPos, &s_block.boundaries[bound].fb, s_domain) * faceNormal;
					
					// from advection
					//tempRHS -= scalar * flux;
					// from viscosity
					// TODO: simple slip from boundary type
					//const scalar_t slip = s_block.boundaries[bound].fb.passiveScalar.boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
					//tempRHS += scalar * (1-slip) * viscosity * 2 * alpha;
					
					// gradients
					
					scalar_t flux_grad = 0;
					// from advection
					flux_grad -= scalar * RHSgrad;
					
					scalar_t scalar_grad = 0;
					// from advection
					scalar_grad -= flux * RHSgrad;
					// from viscosity
					if(scalarBoundaryType==BoundaryConditionType::DIRICHLET){
						scalar_grad += viscosity * 2 * alpha * RHSgrad;
						viscosity_grad += scalar * 2 * alpha * RHSgrad;
					} else {
						scalar_grad += viscosity * RHSgrad;
						viscosity_grad += scalar * RHSgrad;
					}
					
					// scatter flux grad
					scatterContravariantComponentBoundaryFixedDimSwitch(flux_grad*faceNormal, tempPos, &s_block.boundaries[bound].fb, s_domain);
					//scatter scalar grad
					tempPos.w = channel;
					scatterFixedBoundaryData(scalar_grad, tempPos, bound, &s_block, s_domain, GridDataType::PASSIVE_SCALAR_GRAD);
					
				}
			}
			
			scatterViscosity_GRAD(viscosity_grad, s_domain, s_domain.scalarViscosity_grad!=nullptr, channel);
			
			
			RHSgrad *= det;
			
			const int tempFlatPos = flattenIndex(tempPos, s_block);
			s_block.scalarData_grad[tempFlatPos] = RHSgrad/timeStep;
			
		}
		
	)
}
#endif //WITH_GRAD

template <typename scalar_t, int DIMS>
__global__ void kPISO_build_advection_RHS(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
	{ //<- necessary for the macro to work with multiple-argument templates?
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const scalar_t det = s_block.hasTransform ? getDeterminant<scalar_t, DIMS>(s_block, pos) : 1;
		
		
		
		for(index_t dim=0;dim<DIMS;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const index_t tempFlatPos = flattenIndex(tempPos, s_block);
			scalar_t tempRHS = det * s_block.velocity[tempFlatPos]/timeStep; // /c_domain.timeStep
			
			
			//Source terms: external forces, boundary velocity

			//tempPos.w = dim;
			//tempPos.a[dim] = pos.a[dim];
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					//const scalar_t isTangentialDir = static_cast<scalar_t>((bound>>1)!=dim); // * 2 -1;
					switch(s_block.boundaries[bound].type){
					case BoundaryType::FIXED:
					{
						I4 tempPos = pos;
						tempPos.w = dim;
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t vel = getFixedBoundaryData(tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY);
						tempPos.w = bound>>1;
						const FixedBoundaryGPU<scalar_t> *p_fb = &(s_block.boundaries[bound].fb);
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(tempPos, p_fb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(tempPos, p_fb, s_domain) * faceNormal;
						
						// from advection
						tempRHS -= vel * flux;
						// from viscosity
						const scalar_t boundViscosity = getViscosityFixedBoundary(pos, p_fb, &s_block, s_domain, false, 0);
						// TODO: simple slip from boundary type
						const scalar_t slip = s_block.boundaries[bound].fb.velocity.boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
						tempRHS += vel * (1-slip) * boundViscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					case BoundaryType::VALUE:
					{
						// from advection
						tempRHS -= s_block.boundaries[bound].sdb.velocity.a[dim] * s_block.boundaries[bound].sdb.velocity.a[bound>>1] * faceNormal;
						// from viscosity
						const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
						tempRHS += (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * s_block.boundaries[bound].sdb.velocity.a[dim]; //* isTangentialDir
						break;
					}
					case BoundaryType::DIRICHLET_VARYING:
					{
						I4 tempPos = pos;
						tempPos.w = dim;
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t vel = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)];
						tempPos.w = bound>>1;
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain) * faceNormal;
						
						// from advection
						tempRHS -= vel * flux;
						// from viscosity
						const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
						tempRHS += vel * (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					default:
						break;
					}
				}
			}
			
			//tempRHS -= getNonOrthoLaplaceRHSDimSwitch<scalar_t>(pos, s_block, s_domain, nonOrthoFlags, GridDataType::VELOCITY_RESULT, false) * s_domain.viscosity;
			tempRHS -= getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(tempPos, s_block, s_domain, nonOrthoFlags, GridDataType::VELOCITY_RESULT, true, false, false); //* viscosity;

			tempRHS /= det; // in the used formulation, the advection and diffusion parts have to be divided by the determinant. Other source terms are unaffected, so add them below.

			tempRHS += getBlockVelocitySource(tempPos, &s_block);
			
			if(applyPressureGradient){
				const Vector<scalar_t, DIMS> pressureGrad = getPressureGradient<scalar_t, DIMS>(s_block, pos, s_domain);
				//const Vector<scalar_t, DIMS> pressureGrad = getPressureGradientFVM<scalar_t, DIMS>(s_block, pos, s_domain, 0);
				tempRHS -= pressureGrad.a[dim] * timeStep;
			}
			
			const index_t flatPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			s_domain.velocityRHS[flatPosGlobal] = tempRHS;
			//s_domain.velocityRHS[flatPos + s_block.globalOffset + s_domain.numCells*dim] = tempRHS;
		}
	})
}

#ifdef WITH_GRAD
template <typename scalar_t, int DIMS>
__global__ void kPISO_build_advection_RHS_GRAD(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
	 //<- necessary for the macro to work with multiple-argument templates...
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		const scalar_t det = s_block.hasTransform ? getDeterminant<scalar_t, DIMS>(s_block, pos) : 1;
		const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
		
		for(index_t dim=0;dim<DIMS;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const index_t tempFlatPos = flattenIndex(tempPos, s_block);
			
			scalar_t RHSgrad = s_domain.velocityRHS_grad[flatPos + s_block.globalOffset + s_domain.numCells*dim];
			

			scatterBlockVelocitySource_GRAD(RHSgrad, tempPos, &s_block);
			
			RHSgrad /= det;
			
			// non-ortho laplace gradients w.r.t. velocityResult
			scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD<scalar_t>(-RHSgrad, tempPos, s_block, s_domain, nonOrthoFlags,
				GridDataType::VELOCITY_RESULT_GRAD, true, false, false);
			
			// non-ortho laplace gradients w.r.t. viscosity, TODO: merge into scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD for per-cell treatment.
			//scalar_t viscosity_grad = -RHSgrad * getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(tempPos, s_block, s_domain, nonOrthoFlags, GridDataType::VELOCITY_RESULT, false, false, false);
			
			// TODO: grad from boundaries, w.r.t. viscosity and boundary velocity 
			
			// gradients from/for boundaries
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					if(!(s_block.boundaries[bound].type==BoundaryType::FIXED)){ continue; }
					
					// forward values
					
					I4 tempPos = pos;
					tempPos.w = dim;
					tempPos.a[bound>>1] = 0;
					//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
					const scalar_t vel = getFixedBoundaryData(tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY);
					tempPos.w = bound>>1;
					const FixedBoundaryGPU<scalar_t> *p_fb = &(s_block.boundaries[bound].fb);
					const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(tempPos, p_fb, s_domain.numDims);
					//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
					const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(tempPos, p_fb, s_domain) * faceNormal;
					
					// from advection
					//tempRHS -= vel * flux;
					// from viscosity
					// TODO: simple slip from boundary type
					const scalar_t slip = s_block.boundaries[bound].fb.velocity.boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
					//tempRHS += vel * (1-slip) * s_domain.viscosity * 2 * alpha; //* isTangentialDir
					
					// gradients
					
					scalar_t flux_grad = 0;
					// from advection
					flux_grad -= vel * RHSgrad;
					
					scalar_t vel_grad = 0;
					// from advection
					vel_grad -= flux * RHSgrad;
					// from viscosity
					const scalar_t boundViscosity = getViscosityFixedBoundary<scalar_t>(pos, p_fb, &s_block, s_domain, false, 0);
					vel_grad += (1-slip) * boundViscosity * 2 * alpha * RHSgrad;
					
					const scalar_t boundViscosity_grad = vel * (1-slip) * 2 * alpha * RHSgrad;
					
					// scatter flux grad
					scatterContravariantComponentBoundaryFixedDimSwitch<scalar_t>(flux_grad*faceNormal, tempPos, &s_block.boundaries[bound].fb, s_domain);
					//scatter vel grad
					tempPos.w = dim;
					scatterFixedBoundaryData<scalar_t>(vel_grad, tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY_GRAD);
					//scatter viscosity grad
					scatterViscosityBoundary_GRAD<scalar_t>(boundViscosity_grad, pos, p_fb, &s_block, s_domain, false, 0);
					
				}
			}
			
			//scatterViscosityBlock_GRAD<scalar_t>(viscosity_grad, pos, &s_block, s_domain, false, 0);
			
			
			RHSgrad *= det;
			
			s_block.velocity_grad[tempFlatPos] = RHSgrad / timeStep; // * det
		}
	)
}

#endif //WITH_GRAD



template <typename scalar_t>
void _SetupAdvectionMatrixEulerImplicit(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	// setup domain
	int32_t threads;
	dim3 blocks;
	std::vector<index_t> blockIdxByThreadBlock;
	std::vector<index_t> threadBlockOffsetInBlock;
	ComputeThreadBlocks(domain, threads, blocks, blockIdxByThreadBlock, threadBlockOffsetInBlock);
	index_t *p_blockIdxByThreadBlock;
	index_t *p_threadBlockOffsetInBlock;
	torch::Tensor t_blockIdxByThreadBlock = CopyBlockIndices(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock); //keep the torch::Tensor to deallocate at the end

	// make A matrix
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//LOG("Dispatch A matrix kernel");
	BEGIN_SAMPLE;
 	PISO_build_matrix<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, forPassiveScalar, passiveScalarChannel
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Build A-Matrix");
	
}

void SetupAdvectionMatrixEulerImplicit(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	if(forPassiveScalar){
		TORCH_CHECK(domain->hasPassiveScalar(), "domain does not have passive scalar.");
		TORCH_CHECK(0<=passiveScalarChannel && passiveScalarChannel<domain->getPassiveScalarChannels(), "Passive scalar channel index out of bounds.");
	}
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupAdvectionMatrixEulerImplicit", ([&] {
		_SetupAdvectionMatrixEulerImplicit<scalar_t>(
			domain, timeStep, nonOrthoFlags, forPassiveScalar, passiveScalarChannel
		);
	}));
}

#ifdef WITH_GRAD


template <typename scalar_t>
void _SetupAdvectionMatrixEulerImplicit_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	// setup domain
	int32_t threads;
	dim3 blocks;
	std::vector<index_t> blockIdxByThreadBlock;
	std::vector<index_t> threadBlockOffsetInBlock;
	ComputeThreadBlocks(domain, threads, blocks, blockIdxByThreadBlock, threadBlockOffsetInBlock);
	index_t *p_blockIdxByThreadBlock;
	index_t *p_threadBlockOffsetInBlock;
	torch::Tensor t_blockIdxByThreadBlock = CopyBlockIndices(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock); //keep the torch::Tensor to deallocate at the end

	// make A matrix
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//LOG("Dispatch A matrix kernel");
	BEGIN_SAMPLE;
 	PISO_build_matrix_GRAD<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, forPassiveScalar, passiveScalarChannel
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Build A-Matrix");
	
}

void SetupAdvectionMatrixEulerImplicit_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool forPassiveScalar, const index_t passiveScalarChannel){
	
	if(forPassiveScalar){
		TORCH_CHECK(domain->hasPassiveScalar(), "domain does not have passive scalar.");
		TORCH_CHECK(0<=passiveScalarChannel && passiveScalarChannel<domain->getPassiveScalarChannels(), "Passive scalar channel index out of bounds.");
	}
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupAdvectionMatrixEulerImplicit", ([&] {
		_SetupAdvectionMatrixEulerImplicit_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, forPassiveScalar, passiveScalarChannel
		);
	}));
}

#endif //WITH_GRAD

template <typename scalar_t>
void _SetupAdvectionScalarEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_scalar_advection_RHS<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("scalar Advection RHS");
	
}
void SetupAdvectionScalarEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags){
	
	TORCH_CHECK(domain->hasPassiveScalar(), "domain does not have passive scalar.");
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupAdvectionScalarEulerImplicitRHS", ([&] {
		_SetupAdvectionScalarEulerImplicitRHS<scalar_t>(
			domain, timeStep, nonOrthoFlags
		);
	}));
}

#ifdef WITH_GRAD
template <typename scalar_t>
void _SetupAdvectionScalarEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_scalar_advection_RHS_GRAD<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("scalar Advection RHS");
	
}
void SetupAdvectionScalarEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags){
	
	TORCH_CHECK(domain->hasPassiveScalar(), "domain does not have passive scalar.");
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupAdvectionScalarEulerImplicitRHS", ([&] {
		_SetupAdvectionScalarEulerImplicitRHS_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags
		);
	}));
}
#endif //WITH_GRAD

template <typename scalar_t, int DIMS>
void _SetupAdvectionVelocityEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_advection_RHS<scalar_t, DIMS><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, applyPressureGradient
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Velocity Advection RHS");
	
}
void SetupAdvectionVelocityEulerImplicitRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	DISPATCH_FTYPES_DIMS(domain, "SetupAdvectionVelocityEulerImplicitRHS",
		_SetupAdvectionVelocityEulerImplicitRHS<scalar_t, dim>(
			domain, timeStep, nonOrthoFlags, applyPressureGradient
		)
	)
}

#ifdef WITH_GRAD

template <typename scalar_t, int DIMS>
void _SetupAdvectionVelocityEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_advection_RHS_GRAD<scalar_t, DIMS><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, applyPressureGradient
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Velocity Advection RHS grad");
	
}
void SetupAdvectionVelocityEulerImplicitRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	DISPATCH_FTYPES_DIMS(domain, "SetupAdvectionVelocityEulerImplicitRHS_GRAD",
		_SetupAdvectionVelocityEulerImplicitRHS_GRAD<scalar_t, dim>(
			domain, timeStep, nonOrthoFlags, applyPressureGradient
		)
	)
}

#endif //WITH_GRAD

template <typename scalar_t, int DIMS>
void _SetupAdvectionEulerImplicitCombined(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// make A matrix
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	LOG("Dispatch A matrix kernel");
	BEGIN_SAMPLE;
 	PISO_build_matrix<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, false, 0
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Build A-Matrix");
	
	// make velocity RHS
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_advection_RHS<scalar_t, DIMS><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags, applyPressureGradient
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("Velocity Advection RHS");
	
	// make passive scalar RHS
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
 	kPISO_build_scalar_advection_RHS<scalar_t><<<blocks, threads>>>(
		reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
		p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
		nonOrthoFlags
	);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("scalar Advection RHS");
	
}
void SetupAdvectionEulerImplicitCombined(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool applyPressureGradient){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	DISPATCH_FTYPES_DIMS(domain, "SetupAdvectionEulerImplicitCombined",
		_SetupAdvectionEulerImplicitCombined<scalar_t, dim>(
			domain, timeStep, nonOrthoFlags, applyPressureGradient 
		)
	)
}



/* --- PRESSURE SOLVE --- */

template <typename scalar_t> //, index_t DIMS>
__global__ void PISO_build_pressure_matrix(DomainGPU<scalar_t> *p_domain, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flattenIndex(pos, s_block); //flatPos;
		
		const RowMeta row = getCSRMatrixRowEndOffsetFromBlockBoundaries3D(flatPos, s_block, s_domain);
		const index_t rowStartOffset = row.endOffset - row.size + s_block.csrOffset;
		
		s_domain.P.row[flatPosGlobal+1] = row.endOffset + s_block.csrOffset;
		//s_domain.P.row[s_block.globalOffset + threadIdx.x] = repetitions;
		
		// alternative: compute flat indices, sort
		index_t indices[7]; // diag,-x,+x,-y,+y,-z,+z
		scalar_t rowValues[7] = {0};
		
		// orthogonal transform coefficients
		scalar_t alphaP[3];
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			alphaP[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
		}
		const scalar_t raP = static_cast<scalar_t>(1.0) /s_domain.Adiag[flatPosGlobal];
		indices[0] = flatPosGlobal;
		
		for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
			const index_t dim = axisFromBound(bound);
			const index_t isUpper = boundIsUpper(bound);
			const index_t faceSign = faceSignFromBound(bound);
			const bool atBound = isAtBound(pos, bound, &s_block);
			const bool atPrescribedBound = atBound && isEmptyBound(bound, s_block.boundaries);
			
			if(!atPrescribedBound){
				// laplace difference to diffusion: sign, includes 1/A, does not include domain.viscosity
				{ // laplace orthogonal coefficients from face normal directions
					I4 tempPos = pos;
					tempPos.w = dim;
					const BlockGPU<scalar_t> *p_block = &s_block;
					
					if(atBound && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
						p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					}else {
						if(atBound && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
							tempPos.a[dim] = isUpper ? 0 : s_block.size.a[dim]-1;
						}else{
							tempPos.a[dim] = pos.a[dim] + faceSign;
						}
					}
					
					const scalar_t alphaN = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
					
					tempPos.w = 0;
					const index_t tempFlatPosGlobal = flattenIndex(tempPos, p_block) + p_block->globalOffset;
					const scalar_t raN = static_cast<scalar_t>(1.0) / s_domain.Adiag[tempFlatPosGlobal];
					
					scalar_t coefficient = 0;
					if(useFaceTransform){
						tempPos = pos;
						p_block = &s_block;
						//const scalar_t alphaFace = getLaplaceCoefficientOrthogonalFaceDimSwitch(pos, bound, &s_block, s_domain.numDims); // <- ISSUE here?
						const scalar_t alphaFace = getLaplaceCoefficientOrthogonalFaceDimSwitch(tempPos, bound, p_block, s_domain.numDims);
						// TODO: (1/aP + 1/aN)/2 vs. 1/((aP + aN)/2) ? 
						const scalar_t raFace = static_cast<scalar_t>(0.5) * (raP + raN);
						coefficient = alphaFace * raFace;
					}else{
						coefficient = static_cast<scalar_t>(0.5) * (alphaP[dim]*raP + alphaN*raN);
					}
					
					rowValues[0] -= coefficient;
					rowValues[bound+1] += coefficient;
					indices[bound+1] = tempFlatPosGlobal;
				}
				
				// laplace non-orthogonal coefficients from face tangential directions
				const bool includeNonOrthoNeighbors = nonOrthoFlags & NON_ORTHO_DIRECT_MATRIX;
				const bool includeNonOrthoDiag = nonOrthoFlags & NON_ORTHO_CENTER_MATRIX;
				if(s_domain.numDims>1 && (includeNonOrthoDiag || includeNonOrthoNeighbors)){
					// non-orthogonal transform coefficients
					scalar_t alphaInterp[12]; // size for 3D
					if(s_domain.numDims==2) interpolateNonOrthoLaplaceComponents<scalar_t, 2>(pos, s_block, s_domain, alphaInterp, false, true, useFaceTransform);
					else if(s_domain.numDims==3) interpolateNonOrthoLaplaceComponents<scalar_t, 3>(pos, s_block, s_domain, alphaInterp, false, true, useFaceTransform);
					
					for(index_t i=1; i<s_domain.numDims; ++i){ // loop other axes
						const index_t tAxis = (dim + i)%s_domain.numDims;
						const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims);
						
						if(alpha!=0){ // grid is non-orthogonal here
							for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){ // loop corners
								const index_t tFace = axisToBound(tAxis, tIsUpper); //(tAxis<<1) + tIsUpper;
								const index_t tFaceSign = faceSignFromBound(tFace);
								const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(pos, bound, tFace,
									false, false, 2, s_block, s_domain, GridDataType::IS_FIXED_BOUNDARY); // does not read data. computes interpolation divisor and checks for boundaries.
								const bool cornerAtBound = cVal.numCells<1;
								const bool boundIsGradient = true; //gridDataTypeToBaseType(type)==GridDataType::PRESSURE; // TODO: get from bound with FIXED boundary implementation
								
								if(cornerAtBound){
									if(boundIsGradient){
										// simplified treatment: ignore boudnary gradient and use one-sided difference from other side
										const scalar_t interpolationNorm = 0.25; // from other side, can't be anything else but 1/4
										const index_t tFaceOther = invertBound(tFace);
										const scalar_t coefficient = faceSign * tFaceSign * alpha * interpolationNorm;
										if(includeNonOrthoDiag){
											rowValues[0] += 3 * coefficient;
										}
										if(includeNonOrthoNeighbors){
											rowValues[bound+1] += 3 * coefficient;
											rowValues[tFaceOther+1] -= coefficient;
										}
										// if diagonals where to be included in the matrix: 
										// rowValues[diagonalOther] += coefficient;
									}
									// else: prescribed value, added on RHS, nothing to do here.
								} else {
									// normal corner with interpolated value
									const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
									const scalar_t coefficient = faceSign * tFaceSign * alpha * interpolationNorm;
									if(includeNonOrthoDiag){
										rowValues[0] += coefficient;
									}
									if(includeNonOrthoNeighbors){
										rowValues[bound+1] += coefficient;
										rowValues[tFace+1] += coefficient;
									}
									// if diagonals where to be included in the matrix: 
									// rowValues[diagonal] -= coefficient;
								}
							}
						}
					}
				}
			} else { // prescribedBound
				// TODO: some non-orthogonal handling?
				indices[bound+1] = -1; //invalid/unused
			}
		}
		
		
		for(int dim=s_domain.numDims;dim<3;++dim){
			indices[dim*2+1] = -1; //invalid/unused
			indices[dim*2+2] = -1; //invalid/unused
		}
		
		
		// sort, naive for now
		for(index_t i=0;i<row.size;++i){
			index_t colIndex = findLowestColumnIndex(indices, 7);
			
			/* if((rowStartOffset + i)>=127){
				
			}else  */if(colIndex<0){
				s_domain.P.index[rowStartOffset + i] = -1;
				s_domain.P.value[rowStartOffset + i] = -1.0f;
			}else{
				s_domain.P.index[rowStartOffset + i] = indices[colIndex];
				s_domain.P.value[rowStartOffset + i] = rowValues[colIndex];
			}
			
			indices[colIndex] = -1;
		}
	)
}

#ifdef WITH_GRAD

template <typename scalar_t> //, index_t DIMS>
__global__ void PISO_build_pressure_matrix_GRAD(DomainGPU<scalar_t> *p_domain, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	const scalar_t half = static_cast<scalar_t>(0.5);
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
		
		//const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		//const scalar_t rDet = 1/det;
		
		// load row from pressure CSR matrix grad
		scalar_t csrValuesGrad[7] = {0}; // to be loaded like: diag,-x,+x,-y,+y,-z,+z
		LoadCSRrowNeighborSorted(pos, s_domain.P_grad, s_domain, s_block, csrValuesGrad);
		
		// orthogonal transform coefficients
		scalar_t alphaP[3];
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			alphaP[dim] = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, &s_block, s_domain.numDims);
		}
		scalar_t raP_grad = 0;
		
		for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){
			const index_t dim = axisFromBound(bound);
			const index_t isUpper = boundIsUpper(bound);
			const index_t faceSign = faceSignFromBound(bound);
			
			const bool atBound = isAtBound(pos, bound, s_block);
			if(!atBound || !isEmptyBound(bound, s_block.boundaries)){
				
				// flux grad and orthogonal diffusion grad
				{
					//calculate index of neighbour
					I4 tempPos = pos;
					tempPos.w = dim;
					scalar_t alphaN = 1;
					const BlockGPU<scalar_t> *p_block = &s_block;
					// resolve neighbor cell
					if(atBound && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
						p_block = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
						tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					}else{
						if(atBound && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
							tempPos.a[dim] = isUpper ? 0 : s_block.size.a[dim]-1;
						}else{
							tempPos.a[dim] = pos.a[dim] + faceSign;
						}
					}
					// need correct component/channel for laplace coefficient
					alphaN = getLaplaceCoefficientOrthogonalDimSwitch(tempPos, p_block, s_domain.numDims);
					
					tempPos.w = 0;
					const index_t tempFlatPosGlobal = flattenIndex(tempPos, p_block) + p_block->globalOffset;
					const scalar_t raN = static_cast<scalar_t>(1.0) / s_domain.Adiag[tempFlatPosGlobal];
					
					const scalar_t coefficient_grad = csrValuesGrad[bound+1] - csrValuesGrad[0];
					raP_grad += half * alphaP[dim] * coefficient_grad;
					const scalar_t raN_grad = half * alphaN * coefficient_grad;
					
					tempPos.w = 0;
					atomicAdd(s_domain.Adiag_grad + tempFlatPosGlobal, -raN_grad*raN*raN);
				
				}
				
				// non-ortho diffusion grad
				const bool includeNonOrthoNeighbors = nonOrthoFlags & NON_ORTHO_DIRECT_MATRIX;
				const bool includeNonOrthoDiag = nonOrthoFlags & NON_ORTHO_CENTER_MATRIX;
				if(s_domain.numDims>1 && (includeNonOrthoDiag || includeNonOrthoNeighbors)){
					scalar_t alphaInterp[12]; // size for 3D
					scalar_t alphaInterp_grad[12] = {0};
					if(s_domain.numDims==2) interpolateNonOrthoLaplaceComponents<scalar_t, 2>(pos, s_block, s_domain, alphaInterp, false, false, false);
					else if(s_domain.numDims==3) interpolateNonOrthoLaplaceComponents<scalar_t, 3>(pos, s_block, s_domain, alphaInterp, false, false, false);
					
					I4 channelPos = pos;
					
					for(index_t i=1; i<s_domain.numDims; ++i){ // loop other axes
						const index_t tAxis = (dim + i)%s_domain.numDims;
						const scalar_t alpha = getInterpolatedNonOrthoLaplaceComponent(alphaInterp, bound, tAxis, s_domain.numDims);
						
						if(alpha!=0){ // grid is non-orthogonal here
							scalar_t alphaRa_grad = 0;
							for(index_t tIsUpper=0; tIsUpper<2; ++tIsUpper){ // loop corners
								const index_t tFace = axisToBound(tAxis, tIsUpper); //(tAxis<<1) + tIsUpper;
								const index_t tFaceSign = faceSignFromBound(tFace);
								const CornerValue<scalar_t> cVal = getCornerValue<scalar_t>(channelPos, bound, tFace,
									false, false, 2, s_block, s_domain, GridDataType::IS_FIXED_BOUNDARY); // computes interpolation divisor and checks for boundaries.
								const bool cornerAtBound = cVal.numCells<1;
								const bool boundIsGradient = true; //cVal.boundType==BoundaryConditionType::NEUMANN;
								
								if(cornerAtBound){
									if(boundIsGradient){
										// simplified treatment: ignore boudnary gradient and use one-sided difference from other side
										const scalar_t interpolationNorm = 0.25; // from other side, can't be anything else but 1/4
										const index_t tFaceOther = invertBound(tFace);
										
										scalar_t coefficient_grad = 0;
										if(includeNonOrthoDiag){
											coefficient_grad += 3 * csrValuesGrad[0];
										}
										if(includeNonOrthoNeighbors){
											coefficient_grad += 3 * csrValuesGrad[bound+1];
											coefficient_grad -= csrValuesGrad[tFaceOther+1];
										}
										
										alphaRa_grad += faceSign * tFaceSign * interpolationNorm * coefficient_grad;
									}
									// else: prescribed value, added on RHS, nothing to do here.
								} else {
									// normal corner with interpolated value
									const scalar_t interpolationNorm = 1.0 / static_cast<scalar_t>(cVal.numCells);
									//const scalar_t viscosityCoeff = faceSign * tFaceSign * viscosity * alpha * interpolationNorm;
									scalar_t coefficient_grad = 0;
									if(includeNonOrthoDiag){
										coefficient_grad += csrValuesGrad[0];
									}
									if(includeNonOrthoNeighbors){
										coefficient_grad += csrValuesGrad[bound+1];
										coefficient_grad += csrValuesGrad[tFace+1];
									}
									// if diagonals where to be included in the matrix: 
									// rowValues[diagonal] -= viscosityCoeff;
									alphaRa_grad += faceSign * tFaceSign * interpolationNorm * coefficient_grad;
								}
							}
							{
								//ra_grad += alphaRa_grad * alpha;
								addInterpolatedNonOrthoLaplaceComponent_GRAD(alphaRa_grad, alphaInterp_grad, bound, tAxis, s_domain.numDims);
							}
						}
					}
					
					if(s_domain.numDims==2) scatterNonOrthoLaplaceComponents_GRAD<scalar_t, 2>(alphaInterp_grad, pos, s_block, s_domain, false, true, false);
					else if(s_domain.numDims==3) scatterNonOrthoLaplaceComponents_GRAD<scalar_t, 3>(alphaInterp_grad, pos, s_block, s_domain, false, true, false);
					
				}
				
			}
			
		}
		const scalar_t raP = static_cast<scalar_t>(1.0) /s_domain.Adiag[flatPosGlobal];
		atomicAdd(s_domain.Adiag_grad + flatPosGlobal, -raP_grad*raP*raP);
	); //KERNEL_PER_CELL_LOOP
}

#endif //WITH_GRAD

#define PRESSURE_RHS_WITH_BOUNDARY_SOURCES

template <typename scalar_t>
__global__ void PISO_build_pressure_rhs(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		const scalar_t rDiag = 1 / s_domain.Adiag[flatPosGlobal];
		
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
		
		// load row from advection CSR matrix
		index_t csrIndices[7];
		scalar_t csrValues[7];
		const index_t csrStart = s_domain.C.row[flatPosGlobal];
		const index_t csrEnd = s_domain.C.row[flatPosGlobal+1];
		const index_t rowSize = csrEnd - csrStart;
		for(index_t i=0;i<rowSize && i<7;++i){
			csrIndices[i] = s_domain.C.index[csrStart+i];
			csrValues[i] = s_domain.C.value[csrStart+i];
		}
		
		for(index_t dim=0;dim<s_domain.numDims;++dim){ // loop velocity components
			I4 compPos = pos;
			compPos.w = dim;
			const index_t flatCompPos = flattenIndex(compPos, s_block);
			
			const scalar_t velOld = s_block.velocity[flatCompPos]/timeStep; //det * 
			
			scalar_t H = 0;
			for(index_t i=0;i<rowSize;++i){
				index_t idx = csrIndices[i]; // these are global, but for scalars
				if(idx!=flatPosGlobal){ //omit diagonal entries
					const scalar_t vel = s_domain.velocityResult[idx + s_domain.numCells*dim];
					H += csrValues[i] * vel; //H'u*
				}
			}
			
			// source terms
			scalar_t S = 0;
			
			// - non-orthogonal transformation - AFTER divergence!
			
#ifdef PRESSURE_RHS_WITH_BOUNDARY_SOURCES
			// - boundary source terms. TODO: is this correct here? yes
			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1); // [0,1] -> [-1,1]
					//const scalar_t isTangentialDir = static_cast<scalar_t>((bound>>1)!=dim); // * 2 -1;
					switch(s_block.boundaries[bound].type){
					case BoundaryType::FIXED:
					{
						I4 tempPos = pos;
						tempPos.w = dim;
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t vel = getFixedBoundaryData(tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY);
						tempPos.w = bound>>1;
						const FixedBoundaryGPU<scalar_t> *p_fb = &(s_block.boundaries[bound].fb);
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(tempPos, p_fb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(tempPos, p_fb, s_domain) * faceNormal;
						
						// from advection
						S -= vel * flux;
						// from viscosity
						const scalar_t boundViscosity = getViscosityFixedBoundary(pos, p_fb, &s_block, s_domain, false, 0);
						// TODO: simple slip from boundary type
						const scalar_t slip = s_block.boundaries[bound].fb.velocity.boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
						S += vel * (1-slip) * boundViscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					case BoundaryType::VALUE:
					{
						// from advection
						S -= s_block.boundaries[bound].sdb.velocity.a[dim] * s_block.boundaries[bound].sdb.velocity.a[bound>>1] * faceNormal;
						// from viscosity
						const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
						S += viscosity * 2 * s_block.boundaries[bound].sdb.velocity.a[dim]; //* isTangentialDir
						break;
					}
					case BoundaryType::DIRICHLET_VARYING:
					{
						I4 tempPos = pos;
						tempPos.w = dim;
						tempPos.a[bound>>1] = 0;
						//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
						const scalar_t vel = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)];
						//tempPos.w = 0;
						tempPos.w = bound>>1;
						const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain.numDims);
						//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)]*faceNormal;
						const scalar_t flux = getContravariantComponentBoundaryVaryingDimSwitch(tempPos, &s_block.boundaries[bound].vdb, s_domain) * faceNormal;
						
						// from advection
						S -= vel * flux;
						// from viscosity
						const scalar_t viscosity = getViscosityBlock(pos, &s_block, s_domain, false, 0);
						S += vel * (1-s_block.boundaries[bound].vdb.slip) * viscosity * 2 * alpha; //* isTangentialDir
						break;
					}
					default:
						break;
					}
				}
			}
			
			// TODO: add non-ortho boundary values from advection?

			S /= det;
#endif
			S += getBlockVelocitySource(compPos, &s_block);
			
			s_domain.pressureRHS[flatPosGlobal + s_domain.numCells*dim] = rDiag *(velOld-H + S);
			
		}
	)
}

#ifdef WITH_GRAD
template <typename scalar_t>
__global__ void PISO_build_pressure_rhs_GRAD(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		const scalar_t rDiag = 1 / s_domain.Adiag[flatPosGlobal];
		scalar_t rDiag_grad = 0;
		
#ifdef PRESSURE_RHS_WITH_BOUNDARY_SOURCES
		const scalar_t det = s_block.hasTransform ? getDeterminantDimSwitch(s_block, pos, s_domain.numDims) : 1;
#endif
		
		// load row from advection CSR matrix
		index_t csrIndices[7];
		scalar_t csrValues[7];
		const index_t csrStart = s_domain.C.row[flatPosGlobal];
		const index_t csrEnd = s_domain.C.row[flatPosGlobal+1];
		const index_t rowSize = csrEnd - csrStart;
		for(index_t i=0;i<rowSize && i<7;++i){
			csrIndices[i] = s_domain.C.index[csrStart+i];
			csrValues[i] = s_domain.C.value[csrStart+i];
		}
		scalar_t csrValues_grad[7] = {0};
		
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			I4 compPos = pos;
			compPos.w = dim;
			const index_t flatCompPos = flattenIndex(compPos, s_block);
			
			const scalar_t pGrad = s_domain.pressureRHS_grad[flatPosGlobal + s_domain.numCells*dim];
			
			rDiag_grad += s_domain.pressureRHS[flatPosGlobal + s_domain.numCells*dim] / rDiag * pGrad;
			
			s_block.velocity_grad[flatCompPos] = rDiag/timeStep * pGrad;
			
			//scalar_t H = 0;
			for(index_t i=0;i<rowSize;++i){
				index_t idx = csrIndices[i]; // these are global, but for scalars
				if(idx!=flatPosGlobal){ //omit diagonal entries
					// w.r.t. velocityResult
					atomicAdd(s_domain.velocityResult_grad + (idx + s_domain.numCells*dim), -1*csrValues[i]*rDiag*pGrad);
					// w.r.t. C.value
					const scalar_t vel = s_domain.velocityResult[idx + s_domain.numCells*dim];
					csrValues_grad[i] -= 1*vel*rDiag*pGrad;
				}
			}
			
			// source terms
			scalar_t S_grad = rDiag * pGrad;

			scatterBlockVelocitySource_GRAD(S_grad, compPos, &s_block);

#ifdef PRESSURE_RHS_WITH_BOUNDARY_SOURCES

			S_grad /= det;
			
			//scalar_t viscosity_grad = 0;

			for(index_t bound=0; bound<(s_domain.numDims*2); ++bound){ 
				if(((bound&1)==0 && pos.a[bound>>1]==0) || ((bound&1)==1 && pos.a[bound>>1]==s_block.size.a[bound>>1]-1)){
					const scalar_t faceNormal = static_cast<scalar_t>((bound&1)*2 -1);
					if(!(s_block.boundaries[bound].type==BoundaryType::FIXED)){ continue; }
					
					// forward values
					
					I4 tempPos = pos;
					tempPos.w = dim;
					tempPos.a[bound>>1] = 0;
					//const index_t flatTempPos = flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride);
					const scalar_t vel = getFixedBoundaryData(tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY);
					tempPos.w = bound>>1;
					const FixedBoundaryGPU<scalar_t> *p_fb = &(s_block.boundaries[bound].fb);
					const scalar_t alpha = getLaplaceCoefficientOrthogonalBoundaryFixedDimSwitch(tempPos, p_fb, s_domain.numDims);
					//const scalar_t flux = s_block.boundaries[bound].vdb.velocity[flattenIndex(tempPos, s_block.boundaries[bound].vdb.stride)] * faceNormal;
					const scalar_t flux = getContravariantComponentBoundaryFixedDimSwitch(tempPos, p_fb, s_domain) * faceNormal;
					
					// from advection
					//tempRHS -= vel * flux;
					// from viscosity
					// TODO: simple slip from boundary type
					const scalar_t slip = s_block.boundaries[bound].fb.velocity.boundaryType==BoundaryConditionType::DIRICHLET ? 0 : 1;
					//tempRHS += vel * (1-slip) * s_domain.viscosity * 2 * alpha; //* isTangentialDir
					
					// gradients
					
					scalar_t flux_grad = 0;
					// from advection
					flux_grad -= vel * S_grad;
					
					scalar_t vel_grad = 0;
					// from advection
					vel_grad -= flux * S_grad;
					// from viscosity
					const scalar_t boundViscosity = getViscosityFixedBoundary(pos, p_fb, &s_block, s_domain, false, 0);
					vel_grad += (1-slip) * boundViscosity * 2 * alpha * S_grad;
					
					const scalar_t boundViscosity_grad = vel * (1-slip) * 2 * alpha * S_grad;
					
					// scatter flux grad
					scatterContravariantComponentBoundaryFixedDimSwitch<scalar_t>(flux_grad*faceNormal, tempPos, &s_block.boundaries[bound].fb, s_domain);
					//scatter vel grad
					tempPos.w = dim;
					scatterFixedBoundaryData<scalar_t>(vel_grad, tempPos, bound, &s_block, s_domain, GridDataType::VELOCITY_GRAD);
					//scatter viscosity grad
					scatterViscosityBoundary_GRAD<scalar_t>(boundViscosity_grad, pos, p_fb, &s_block, s_domain, false, 0);
					
				}
			}
			//scatterViscosity_GRAD(viscosity_grad, s_domain, false, 0);
#endif //PRESSURE_RHS_WITH_BOUNDARY_SOURCES
		}
		
		//TODO: torch autograd logistics
		for(index_t i=0;i<rowSize && i<7;++i){
			s_domain.C_grad.value[csrStart+i] = csrValues_grad[i];
		}
		
		//s_domain.Adiag_grad[flatPosGlobal] = rDiag_grad * (- rDiag * rDiag);
		// needs to be added to not overwrite Adiag_grad from SetupPressureRHSdiv_GRAD. Does not need atomics here.
		s_domain.Adiag_grad[flatPosGlobal] += rDiag_grad * (- rDiag * rDiag);
		
	)
}

#endif //WITH_GRAD


template <typename scalar_t>
__global__ void k_computePressureRHSdivergenceFromFlux(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const bool useFaceTransform, const bool timeStepNorm){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t fluxes[6]; //DIMS*2
		if(useFaceTransform){
			switch(s_domain.numDims){
				case 1:
					computeFluxesWithFaceTransforms<scalar_t, 1>(pos, fluxes, s_block, s_domain, s_domain.pressureRHS);
					break;
				case 2:
					computeFluxesWithFaceTransforms<scalar_t, 2>(pos, fluxes, s_block, s_domain, s_domain.pressureRHS);
					break;
				case 3:
					computeFluxesWithFaceTransforms<scalar_t, 3>(pos, fluxes, s_block, s_domain, s_domain.pressureRHS);
					break;
				default:
					break;
			}
		}else{
			computeFluxesNDLoop(pos, fluxes, s_block, s_domain, s_domain.pressureRHS);
		}
			
		scalar_t div = 0;
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			div += fluxes[dim*2+1] - fluxes[dim*2];
		}
		
		
		//const scalar_t det = getDeterminantDimSwitch(s_block, pos, s_domain.numDims);
		
		if(timeStepNorm){
			div /= timeStep;
		}
		
		s_domain.pressureRHSdiv[flatPos + s_block.globalOffset] = div;
	)
}

#ifdef WITH_GRAD

template <typename scalar_t>
__global__ void k_computePressureRHSdivergenceFromFlux_GRAD(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const bool timeStepNorm){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		index_t flatPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
		
		scalar_t divGrad = s_domain.pressureRHSdiv_grad[flatPosGlobal];
		
		if(timeStepNorm){
			divGrad /= timeStep;
		}
		
		scalar_t fluxesGrad[6]; //DIMS*2
			
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			fluxesGrad[dim*2+1] = divGrad;
			fluxesGrad[dim*2] = -divGrad;
		}
		
		ScatterFluxesGradNDLoop(pos, fluxesGrad, s_block, s_domain, s_domain.pressureRHS_grad);
		
	)
}
#endif //WITH_GRAD

template <typename scalar_t>
__global__ void k_pressureRHSaddNonOrthoComponents(DomainGPU<scalar_t> *p_domain, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t S = s_domain.pressureRHSdiv[flatPos + s_block.globalOffset];
		
		// - non-orthogonal transformation
		if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
			//S = getNonOrthoLaplacePressureRHSDimSwitch<scalar_t>(pos, s_block, s_domain, nonOrthoFlags);
			S += getNonOrthoLaplaceRHSDimSwitch_v2<scalar_t>(pos, s_block, s_domain, nonOrthoFlags, GridDataType::PRESSURE_RESULT, false, true, useFaceTransform);
		}
		
		
		s_domain.pressureRHSdiv[flatPos + s_block.globalOffset] = S;
	)
}
#ifdef WITH_GRAD
template <typename scalar_t>
__global__ void k_pressureRHSaddNonOrthoComponents_GRAD(DomainGPU<scalar_t> *p_domain, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks,
		const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		const scalar_t RHSdiv_grad = s_domain.pressureRHSdiv_grad[flatPos + s_block.globalOffset];
		
		if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
			scatterNonOrthoLaplaceRHSDimSwitch_v2_GRAD<scalar_t>(RHSdiv_grad, pos, s_block, s_domain, nonOrthoFlags,
				GridDataType::PRESSURE_RESULT_GRAD, false, true, useFaceTransform);
		}
	)
}

#endif //WITH_GRAD

template <typename scalar_t>
void _SetupPressureCorrection(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// make P matrix
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	PISO_build_pressure_matrix<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix");
	
	// make P rhs
	BEGIN_SAMPLE;
	PISO_build_pressure_rhs<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS");
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			useFaceTransform, timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS");
	
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
}
void SetupPressureCorrection(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureCorrection", ([&] {
		_SetupPressureCorrection<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}



template <typename scalar_t>
void _SetupPressureMatrix(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// make P matrix
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	PISO_build_pressure_matrix<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix");
	
}
void SetupPressureMatrix(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureMatrix", ([&] {
		_SetupPressureMatrix<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform
		);
	}));
}

template <typename scalar_t>
void _SetupPressureRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	
	// make P rhs
	BEGIN_SAMPLE;
	PISO_build_pressure_rhs<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS");
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			useFaceTransform, timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS");
	
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
}
void SetupPressureRHS(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureRHS", ([&] {
		_SetupPressureRHS<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}

template <typename scalar_t>
void _SetupPressureRHSdiv(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			useFaceTransform, timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS");
	
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
}
void SetupPressureRHSdiv(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureRHSdiv", ([&] {
		_SetupPressureRHSdiv<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}

#ifdef WITH_GRAD
template <typename scalar_t>
void _SetupPressureCorrection_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents_GRAD<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS grad");
	
	// make P rhs
	BEGIN_SAMPLE;
	PISO_build_pressure_rhs_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS grad");
	
	BEGIN_SAMPLE;
	PISO_build_pressure_matrix_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix grad");
	
}
void SetupPressureCorrection_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureCorrection_GRAD", ([&] {
		_SetupPressureCorrection_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}

template <typename scalar_t>
void _SetupPressureMatrix_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	PISO_build_pressure_matrix_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			nonOrthoFlags, useFaceTransform
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-Matrix grad");
	
}
void SetupPressureMatrix_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const int8_t nonOrthoFlags, const bool useFaceTransform){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureMatrix_GRAD", ([&] {
		_SetupPressureMatrix_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform
		);
	}));
}


template <typename scalar_t>
void _SetupPressureRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// Non-ortho components
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents_GRAD<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS grad");
	
	// make P rhs
	BEGIN_SAMPLE;
	PISO_build_pressure_rhs_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Build p-RHS grad");
	
}
void SetupPressureRHS_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureRHS_GRAD", ([&] {
		_SetupPressureRHS_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}


template <typename scalar_t>
void _SetupPressureRHSdiv_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// Non-ortho components
	if((nonOrthoFlags & NON_ORTHO_DIRECT_RHS) | (nonOrthoFlags & NON_ORTHO_DIAGONAL_RHS)){
		BEGIN_SAMPLE;
		k_pressureRHSaddNonOrthoComponents_GRAD<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), //timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
				nonOrthoFlags, useFaceTransform
			);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		END_SAMPLE("p-RHS add non ortho");
	}
	
	// divergence P rhs
	BEGIN_SAMPLE;
	k_computePressureRHSdivergenceFromFlux_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size(),
			timeStepNorm
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence p-RHS grad");
	
}
void SetupPressureRHSdiv_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep,
		const int8_t nonOrthoFlags, const bool useFaceTransform, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "SetupPressureRHSdiv_GRAD", ([&] {
		_SetupPressureRHSdiv_GRAD<scalar_t>(
			domain, timeStep, nonOrthoFlags, useFaceTransform, timeStepNorm
		);
	}));
}

#endif //WITH_GRAD

/* --- Velocity correction --- */

template<typename scalar_t>
__device__ void tempGetPressureGradientDimSwitch(const BlockGPU<scalar_t> &block, const I4 &pos, const DomainGPU<scalar_t> &domain, scalar_t *pressureGradOut){
	switch(domain.numDims){
	case 1:
	{
		const Vector<scalar_t, 1> pressureGrad = getPressureGradient<scalar_t, 1>(block, pos, domain);
		pressureGradOut[0] = pressureGrad.a[0];
		return;
	}
	case 2:
	{
		const Vector<scalar_t, 2> pressureGrad = getPressureGradient<scalar_t, 2>(block, pos, domain);
		pressureGradOut[0] = pressureGrad.a[0];
		pressureGradOut[1] = pressureGrad.a[1];
		return;
	}
	case 3:
	{
		const Vector<scalar_t, 3> pressureGrad = getPressureGradient<scalar_t, 3>(block, pos, domain);
		pressureGradOut[0] = pressureGrad.a[0];
		pressureGradOut[1] = pressureGrad.a[1];
		pressureGradOut[2] = pressureGrad.a[2];
		return;
	}
	default:
		return;
	}
}

// pressure gradient differenced over +-1
template <typename scalar_t>
__global__ void PISO_update_velocity(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep, const bool timeStepNorm,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t rDiag = 1/s_domain.Adiag[flatPosGlobal];
		
		scalar_t pressureGrad[3];
		tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = 0;
			
			scalar_t velUpdate = - rDiag * pressureGrad[dim]; // * timeStep;
			
			if(timeStepNorm){
				velUpdate *= timeStep;
			}
			
			
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim];
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			velUpdate += s_domain.pressureRHS[flatCompPosGlobal];
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}

template<typename scalar_t>
__device__ void tempGetPressureGradientFVMDimSwitch(const BlockGPU<scalar_t> &block, const I4 &pos, const DomainGPU<scalar_t> &domain,
		const index_t gradientInterpolation, scalar_t *pressureGradOut){
	switch(domain.numDims){
	case 1:
	{
		const Vector<scalar_t, 1> pressureGrad = getPressureGradientFVM<scalar_t, 1>(block, pos, domain, gradientInterpolation);
		pressureGradOut[0] = pressureGrad.a[0];
		return;
	}
	case 2:
	{
		const Vector<scalar_t, 2> pressureGrad = getPressureGradientFVM<scalar_t, 2>(block, pos, domain, gradientInterpolation);
		pressureGradOut[0] = pressureGrad.a[0];
		pressureGradOut[1] = pressureGrad.a[1];
		return;
	}
	case 3:
	{
		const Vector<scalar_t, 3> pressureGrad = getPressureGradientFVM<scalar_t, 3>(block, pos, domain, gradientInterpolation);
		pressureGradOut[0] = pressureGrad.a[0];
		pressureGradOut[1] = pressureGrad.a[1];
		pressureGradOut[2] = pressureGrad.a[2];
		return;
	}
	default:
		return;
	}
}

// finite volume pressure gradient
template <typename scalar_t>
__global__ void PISO_update_velocity_PressureFVM(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep, const index_t gradientInterpolation, const bool timeStepNorm,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		//const scalar_t det = getDeterminantDimSwitch(s_block, pos, s_domain.numDims); // cell volume
		const scalar_t rDiag = 1/s_domain.Adiag[flatPosGlobal];
		
		scalar_t pressureGrad[3];
		tempGetPressureGradientFVMDimSwitch<scalar_t>(s_block, pos, s_domain, gradientInterpolation, pressureGrad);
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			I4 tempPos = pos;
			tempPos.w = 0;
			scalar_t velUpdate = - rDiag * pressureGrad[dim]; // * timeStep;
			
			if(timeStepNorm){
				velUpdate *= timeStep;
			}
			
			tempPos.w = dim;
			tempPos.a[dim] = pos.a[dim];
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			velUpdate += s_domain.pressureRHS[flatCompPosGlobal];
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}


template <typename scalar_t>
__global__ void PISO_update_velocity_v4_orthogonal(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//TODO: non-orthogonal transformations
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t ra = 1/s_domain.Adiag[flatPosGlobal];
		
		//scalar_t pressureGrad[3];
		//tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		const scalar_t p = s_block.pressure[flattenIndex(pos, s_block)];
		const scalar_t det = getDeterminantDimSwitch(s_block, pos, s_domain.numDims);
		
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			scalar_t velUpdate = 0;
			
			I4 tempPos = pos;
			tempPos.w = dim;
			const int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			const scalar_t hbyA = s_domain.pressureRHS[flatCompPosGlobal];
			const scalar_t t = getTransformMetricOrthogonalDimSwitch<scalar_t>(pos, &s_block, s_domain.numDims);
			const scalar_t cc = det*t; //covariant transform coefficient
			
			//lower boundary
			index_t bound = dim*2;
			if(pos.a[dim]!=0 || !isEmptyBound(bound, s_block.boundaries)){
				tempPos = pos;
				tempPos.w = dim;
				index_t tempFlatPosGlobal = 0;
				scalar_t pL = 0;
				scalar_t hbyAL = 0;
				scalar_t tL = 1; //transform metric
				scalar_t detL = 1;
				if(pos.a[dim]==0 && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
					const BlockGPU<scalar_t> *p_connectedBlock = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					hbyAL = s_domain.pressureRHS[flattenIndexGlobal(tempPos, p_connectedBlock, s_domain)];
					tL = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, p_connectedBlock, s_domain.numDims);
					tempPos.w = 0;
					detL = getDeterminantDimSwitch(p_connectedBlock, tempPos, s_domain.numDims);
					const index_t blockFlatIdx = flattenIndex(tempPos, p_connectedBlock);
					pL = p_connectedBlock->pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + p_connectedBlock->globalOffset;
				}else {
					if(pos.a[dim]==0 && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = s_block.size.a[dim]-1;
					}else{
						tempPos.a[dim] = pos.a[dim]-1;
					}
					hbyAL = s_domain.pressureRHS[flattenIndexGlobal(tempPos, s_block, s_domain)];
					tL = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
					tempPos.w = 0;
					detL = getDeterminantDimSwitch(s_block, tempPos, s_domain.numDims);
					const index_t blockFlatIdx = flattenIndex(tempPos, s_block);
					pL = s_block.pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + s_block.globalOffset;
				}
				const scalar_t raL = 1 /s_domain.Adiag[tempFlatPosGlobal];
				const scalar_t ccL = detL*tL;
				velUpdate += (cc*hbyA + ccL*hbyAL - (cc*t*ra + ccL*tL*raL)*(p - pL) * timeStep) * static_cast<scalar_t>(0.25);
			}
			//upper boundary
			++bound;
			if(pos.a[dim]!=s_block.size.a[dim]-1 || !isEmptyBound(bound, s_block.boundaries)){
				tempPos = pos;
				tempPos.w = dim;
				index_t tempFlatPosGlobal = 0;
				scalar_t pU = 0;
				scalar_t hbyAU = 0;
				scalar_t tU = 1; //transform metric
				scalar_t detU = 1;
				if(pos.a[dim]==s_block.size.a[dim]-1 && s_block.boundaries[bound].type==BoundaryType::CONNECTED_GRID){
					const BlockGPU<scalar_t> *p_connectedBlock = s_domain.blocks + s_block.boundaries[bound].cb.connectedGridIndex;
					tempPos = computeConnectedPosWithChannel(tempPos, dim, &s_block.boundaries[bound].cb, s_domain);
					hbyAU = s_domain.pressureRHS[flattenIndexGlobal(tempPos, p_connectedBlock, s_domain)];
					tU = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, p_connectedBlock, s_domain.numDims);
					tempPos.w = 0;
					detU = getDeterminantDimSwitch(p_connectedBlock, tempPos, s_domain.numDims);
					const index_t blockFlatIdx = flattenIndex(tempPos, p_connectedBlock);
					pU = p_connectedBlock->pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + p_connectedBlock->globalOffset;
				}else {
					if(pos.a[dim]==s_block.size.a[dim]-1 && s_block.boundaries[bound].type==BoundaryType::PERIODIC){
						tempPos.a[dim] = 0;
					}else{
						tempPos.a[dim] = pos.a[dim]+1;
					}
					hbyAU = s_domain.pressureRHS[flattenIndexGlobal(tempPos, s_block, s_domain)];
					tU = getTransformMetricOrthogonalDimSwitch<scalar_t>(tempPos, &s_block, s_domain.numDims);
					tempPos.w = 0;
					detU = getDeterminantDimSwitch(s_block, tempPos, s_domain.numDims);
					const index_t blockFlatIdx = flattenIndex(tempPos, s_block);
					pU = s_block.pressure[blockFlatIdx];
					tempFlatPosGlobal = blockFlatIdx + s_block.globalOffset;
				}
				const scalar_t raU = 1 /s_domain.Adiag[tempFlatPosGlobal];
				const scalar_t ccU = detU*tU;
				velUpdate += (cc*hbyA + ccU*hbyAU - (cc*t*ra + ccU*tU*raU)*(pU - p) * timeStep) * static_cast<scalar_t>(0.25);
			}
			
			//velUpdate is a flux here, transform back with inverse transform
			//orthogonal, so can use inverse of t
			velUpdate /= det*t;
			s_domain.velocityResult[flatCompPosGlobal] = velUpdate;
		}
	)
}

template <typename scalar_t>
void _CorrectVelocity(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const index_t version, const bool timeStepNorm){ //, const torch::Tensor &timeStepCPU){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	BEGIN_SAMPLE;
	switch(version){
	case 1:
		PISO_update_velocity<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0], timeStepNorm,
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	// 2, 3: DO NOT USE, this is a smoothing kernel
	case 4:
		PISO_update_velocity_v4_orthogonal<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0],
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	case 5:
		PISO_update_velocity_PressureFVM<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0], 0, timeStepNorm, // original linear interpolation of metrics/fluxes
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	case 6:
		PISO_update_velocity_PressureFVM<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0], 3, timeStepNorm, // use face metrics instead of interpolation
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	default:
		PISO_update_velocity<scalar_t><<<blocks, threads>>>(
				reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0], timeStepNorm,
				p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
			);
		break;
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("correct u");
	
}
void CorrectVelocity(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const index_t version, const bool timeStepNorm){ //, const torch::Tensor &timeStep){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CorrectVelocity", ([&] {
		_CorrectVelocity<scalar_t>(
			domain, timeStep, version, timeStepNorm
		);
	}));
}

#ifdef WITH_GRAD
template<typename scalar_t>
__device__ void tempScatterPressureGradientGradDimSwitch(const scalar_t *pressureGradGradIn, const BlockGPU<scalar_t> &block, const I4 &pos, const DomainGPU<scalar_t> &domain){
	switch(domain.numDims){
	case 1:
	{
		const Vector<scalar_t, 1> pressureGradGrad = {.a={pressureGradGradIn[0]}};
		scatterPressureGradientGrad<scalar_t, 1>(pressureGradGrad, block, pos, domain);
		return;
	}
	case 2:
	{
		const Vector<scalar_t, 2> pressureGradGrad = {.a={pressureGradGradIn[0], pressureGradGradIn[1]}};
		scatterPressureGradientGrad<scalar_t, 2>(pressureGradGrad, block, pos, domain);
		return;
	}
	case 3:
	{
		const Vector<scalar_t, 3> pressureGradGrad = {.a={pressureGradGradIn[0], pressureGradGradIn[1], pressureGradGradIn[2]}};
		scatterPressureGradientGrad<scalar_t, 3>(pressureGradGrad, block, pos, domain);
		return;
	}
	default:
		return;
	}
}

template <typename scalar_t>
__global__ void PISO_update_velocity_GRAD(DomainGPU<scalar_t> *p_domain, const scalar_t timeStep, const bool timeStepNorm,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		const index_t flatPosGlobal = s_block.globalOffset + flatPos;
		
		const scalar_t rDiag = 1/s_domain.Adiag[flatPosGlobal];
		scalar_t rDiag_grad = 0;
		
		scalar_t pressureGrad[3];
		tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		scalar_t pressureGradGrad[3] = {0};
		
		for(int dim=0;dim<s_domain.numDims;++dim){
			
			I4 tempPos = pos;
			tempPos.w = dim;
			int flatCompPosGlobal = flattenIndexGlobal(tempPos, s_block, s_domain);
			
			scalar_t velUpdateGrad = s_domain.velocityResult_grad[flatCompPosGlobal];
			
			// w.r.t. pressure rhs
			s_domain.pressureRHS_grad[flatCompPosGlobal] = velUpdateGrad;
			
			if(timeStepNorm){
				velUpdateGrad *= timeStep;
			}
			
			//FWD: scalar_t velUpdate = - rDiag * pressureGrad[dim];
			
			pressureGradGrad[dim] = - rDiag * velUpdateGrad;
			rDiag_grad -= pressureGrad[dim] * velUpdateGrad;
			
		}
		
		// scatter pressure grad grad:
		tempScatterPressureGradientGradDimSwitch(pressureGradGrad, s_block, pos, s_domain);
		s_domain.Adiag_grad[flatPosGlobal] = - rDiag_grad * rDiag * rDiag;
	)
}

template <typename scalar_t>
void _CorrectVelocity_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStepCPU, const bool timeStepNorm){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	BEGIN_SAMPLE;
	PISO_update_velocity_GRAD<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), timeStepCPU.data_ptr<scalar_t>()[0], timeStepNorm,
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("correct u grad");
	
}
void CorrectVelocity_GRAD(std::shared_ptr<Domain> domain, const torch::Tensor &timeStep, const bool timeStepNorm){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Gradient for Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	CHECK_INPUT_HOST(timeStep);
	TORCH_CHECK(timeStep.dim()==1, "timeStep must be 1D.");
	TORCH_CHECK(timeStep.size(0)==1, "timeStep must be a scalar.");
	TORCH_CHECK(timeStep.scalar_type()==domain->getDtype(), "Data type of timeStep does not match.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CorrectVelocity_GRAD", ([&] {
		_CorrectVelocity_GRAD<scalar_t>(
			domain, timeStep, timeStepNorm
		);
	}));
}

#endif //WITH_GRAD

/* --- Utility --- */



template <typename scalar_t>
__global__ void k_computeVelocityDivergenceFromFlux(DomainGPU<scalar_t> *p_domain, scalar_t *divergence, //const scalar_t timeStep,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	// divergence of colocated vector field
	// using central differences
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t fluxes[6]; //DIMS*2
		computeFluxesNDLoop<scalar_t>(pos, fluxes, s_block, s_domain, nullptr);
			
		scalar_t div = 0;
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			div += fluxes[dim*2+1] - fluxes[dim*2];
		}
		
		divergence[flatPos + s_block.globalOffset] = div;
	)
}

template <typename scalar_t>
void _ComputeVelocityDivergence(std::shared_ptr<Domain> domain, torch::Tensor &divergence){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// divergence
	BEGIN_SAMPLE;
	k_computeVelocityDivergenceFromFlux<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), divergence.data_ptr<scalar_t>(), p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Divergence Velocity");
	
}
torch::Tensor ComputeVelocityDivergence(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	torch::Tensor divergence = torch::zeros_like(domain->pressureResult);
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputeVelocityDivergence", ([&] {
		_ComputeVelocityDivergence<scalar_t>(
			domain, divergence
		);
	}));
	
	return divergence;
}

// finite volume pressure gradient
template <typename scalar_t>
__global__ void k_computePressureGradient(DomainGPU<scalar_t> *p_domain, const bool useFVM, const index_t gradientInterpolation, scalar_t *gradient,
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	//vel.x = pressureRHS.x - inv(A)*gradX(pressure)
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t pressureGrad[3];
		if(useFVM){
			tempGetPressureGradientFVMDimSwitch<scalar_t>(s_block, pos, s_domain, gradientInterpolation, pressureGrad);
		}else{
			tempGetPressureGradientDimSwitch<scalar_t>(s_block, pos, s_domain, pressureGrad);
		}
		
		for(index_t dim=0;dim<s_domain.numDims;++dim){
			pos.w = dim;
			const index_t flatCompPosGlobal = flattenIndexGlobal(pos, s_block, s_domain);
			gradient[flatCompPosGlobal] = pressureGrad[dim];
		}
	)
}
template <typename scalar_t>
void _ComputePressureGradient(std::shared_ptr<Domain> domain, const bool useFVM, const index_t gradientInterpolation, torch::Tensor &gradient){
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// gradient
	BEGIN_SAMPLE;
	k_computePressureGradient<scalar_t><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), useFVM, gradientInterpolation, gradient.data_ptr<scalar_t>(),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("Pressure Gradient");
	
}
torch::Tensor ComputePressureGradient(std::shared_ptr<Domain> domain, const bool useFVM, const index_t gradientInterpolation){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	//TORCH_CHECK(domain->getNumBlocks()<2, "Multi-block is not yet implemented.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	torch::Tensor gradient = torch::zeros_like(domain->velocityResult);
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "ComputePressureGradient", ([&] {
		_ComputePressureGradient<scalar_t>(
			domain, useFVM, gradientInterpolation, gradient
		);
	}));
	
	return gradient;
}


/* --- Copy functions between the per-block grids and the corresponding global domain.*Result grids --- */

template <typename scalar_t>
void _CopyScalarResultToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t channel=0; channel<domain->getPassiveScalarChannels(); ++channel){
			CUDA_CHECK_RETURN(cudaMemcpy(
				block->getPassiveScalarDataPtr<scalar_t>() + blockSizeFlat*channel, //passiveScalar.data_ptr<scalar_t>(),
				domain->scalarResult.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*channel,
				sizeof(scalar_t)*blockSizeFlat,
				cudaMemcpyDeviceToDevice
			)); //dst, src, bytes, kind
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyScalarResultToBlocks");
	
}
void CopyScalarResultToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(domain->hasPassiveScalar(), "Domain has no passive scalar set.")
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyScalarResultToBlocks", ([&] {
		_CopyScalarResultToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyScalarResultFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t channel=0; channel<domain->getPassiveScalarChannels(); ++channel){
			CUDA_CHECK_RETURN(cudaMemcpy(
				domain->scalarResult.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*channel,
				block->getPassiveScalarDataPtr<scalar_t>() + blockSizeFlat*channel, //passiveScalar.data_ptr<scalar_t>(),
				sizeof(scalar_t)*blockSizeFlat,
				cudaMemcpyDeviceToDevice
			)); //dst, src, bytes, kind
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyScalarResultFromBlocks");
	
}
void CopyScalarResultFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	TORCH_CHECK(domain->hasPassiveScalar(), "Domain has no passive scalar set.")
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyScalarResultFromBlocks", ([&] {
		_CopyScalarResultFromBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyPressureResultToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		CUDA_CHECK_RETURN(cudaMemcpy(
			block->pressure.data_ptr<scalar_t>(),
			domain->pressureResult.data_ptr<scalar_t>() + block->globalOffset,
			sizeof(scalar_t)*block->getStrides().w,
			cudaMemcpyDeviceToDevice
		)); //dst, src, bytes, kind
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyPressureResultToBlocks");
	
}
void CopyPressureResultToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyPressureResultToBlocks", ([&] {
		_CopyPressureResultToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyPressureResultFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		CUDA_CHECK_RETURN(cudaMemcpy(
			domain->pressureResult.data_ptr<scalar_t>() + block->globalOffset,
			block->pressure.data_ptr<scalar_t>(),
			sizeof(scalar_t)*block->getStrides().w,
			cudaMemcpyDeviceToDevice
		)); //dst, src, bytes, kind
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyPressureResultFromBlocks");
	
}
void CopyPressureResultFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyPressureResultFromBlocks", ([&] {
		_CopyPressureResultFromBlocks<scalar_t>(domain);
	}));
}


template <typename scalar_t>
void _CopyVelocityResultToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t dim=0; dim<domain->getSpatialDims(); ++dim){
			CUDA_CHECK_RETURN(cudaMemcpy(
				block->velocity.data_ptr<scalar_t>() + blockSizeFlat*dim,
				domain->velocityResult.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*dim,
				sizeof(scalar_t)*blockSizeFlat,
				cudaMemcpyDeviceToDevice
			)); //dst, src, bytes, kind
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyVelocityResultToBlocks");
	
}
void CopyVelocityResultToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyVelocityResultToBlocks", ([&] {
		_CopyVelocityResultToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyVelocityResultFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t dim=0; dim<domain->getSpatialDims(); ++dim){
			CUDA_CHECK_RETURN(cudaMemcpy(
				domain->velocityResult.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*dim,
				block->velocity.data_ptr<scalar_t>() + blockSizeFlat*dim,
				sizeof(scalar_t)*blockSizeFlat,
				cudaMemcpyDeviceToDevice
			)); //dst, src, bytes, kind
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyVelocityResultFromBlocks");
	
}
void CopyVelocityResultFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyVelocityResultFromBlocks", ([&] {
		_CopyVelocityResultFromBlocks<scalar_t>(domain);
	}));
}

#ifdef WITH_GRAD

template <typename scalar_t>
void _CopyScalarResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t channel=0; channel<domain->getPassiveScalarChannels(); ++channel){
			CUDA_CHECK_RETURN(cudaMemcpy(
				domain->scalarResult_grad.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*channel,
				block->passiveScalar_grad.data_ptr<scalar_t>() + blockSizeFlat*channel, //passiveScalar.data_ptr<scalar_t>(),
				sizeof(scalar_t)*blockSizeFlat,
				cudaMemcpyDeviceToDevice
			)); //dst, src, bytes, kind
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyScalarResultGradFromBlocks");
	
}
void CopyScalarResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyScalarResultGradFromBlocks", ([&] {
		_CopyScalarResultGradFromBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyScalarResultGradToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t channel=0; channel<domain->getPassiveScalarChannels(); ++channel){
			CUDA_CHECK_RETURN(cudaMemcpy(
				block->passiveScalar_grad.data_ptr<scalar_t>() + blockSizeFlat*channel, //passiveScalar.data_ptr<scalar_t>(),
				domain->scalarResult_grad.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*channel,
				sizeof(scalar_t)*blockSizeFlat,
				cudaMemcpyDeviceToDevice
			)); //dst, src, bytes, kind
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyScalarResultGradToBlocks");
	
}
void CopyScalarResultGradToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyScalarResultGradToBlocks", ([&] {
		_CopyScalarResultGradToBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyPressureResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		CUDA_CHECK_RETURN(cudaMemcpy(
			domain->pressureResult_grad.data_ptr<scalar_t>() + block->globalOffset,
			block->pressure_grad.data_ptr<scalar_t>(),
			sizeof(scalar_t)*block->getStrides().w,
			cudaMemcpyDeviceToDevice
		)); //dst, src, bytes, kind
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyPressureResultGradFromBlocks");
	
}
void CopyPressureResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyPressureResultGradFromBlocks", ([&] {
		_CopyPressureResultGradFromBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyVelocityResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t dim=0; dim<domain->getSpatialDims(); ++dim){
			CUDA_CHECK_RETURN(cudaMemcpy(
				domain->velocityResult_grad.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*dim,
				block->velocity_grad.data_ptr<scalar_t>() + blockSizeFlat*dim,
				sizeof(scalar_t)*blockSizeFlat,
				cudaMemcpyDeviceToDevice
			)); //dst, src, bytes, kind
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyVelocityResultGradFromBlocks");
	
}
void CopyVelocityResultGradFromBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyVelocityResultGradFromBlocks", ([&] {
		_CopyVelocityResultGradFromBlocks<scalar_t>(domain);
	}));
}
template <typename scalar_t>
void _CopyVelocityResultGradToBlocks(std::shared_ptr<Domain> domain){
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	BEGIN_SAMPLE;
	
	for(auto block : domain->blocks){
		const index_t blockSizeFlat = block->getStrides().w;
		for(index_t dim=0; dim<domain->getSpatialDims(); ++dim){
			CUDA_CHECK_RETURN(cudaMemcpy(
				block->velocity_grad.data_ptr<scalar_t>() + blockSizeFlat*dim,
				domain->velocityResult_grad.data_ptr<scalar_t>() + block->globalOffset + domain->getTotalSize()*dim,
				sizeof(scalar_t)*blockSizeFlat,
				cudaMemcpyDeviceToDevice
			)); //dst, src, bytes, kind
		}
	}
	
	CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	END_SAMPLE("CopyVelocityResultGradToBlocks");
	
}
void CopyVelocityResultGradToBlocks(std::shared_ptr<Domain> domain){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	AT_DISPATCH_FLOATING_TYPES(domain->getDtype(), "CopyVelocityResultGradToBlocks", ([&] {
		_CopyVelocityResultGradToBlocks<scalar_t>(domain);
	}));
}
#endif //WITH_GRAD


/* --- Sub-Grid Scale models --- */

template<typename scalar_t, int DIMS>
__global__
void k_SGSviscosityIncompressibleSmagorinsky(DomainGPU<scalar_t> *p_domain, const scalar_t coefficient, scalar_t **pp_blockViscosity_out, 
		const index_t *p_blockIdxByThreadBlock, const index_t *p_threadBlockOffsetInBlock, const index_t numThreadBlocks){
	
	
	KERNEL_PER_CELL_LOOP(p_domain, p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, numThreadBlocks,
		
		const I4 pos = unflattenIndex(flatPos, s_block);
		
		scalar_t d = 0; // norm of strain-rate tensor
		{
			MatrixSquare<scalar_t, DIMS> velocityGrads = {0};
			I4 tempPos = pos;
			for(index_t i=0; i<DIMS; ++i){
				tempPos.w = i;
				velocityGrads.v[i] = getBlockDataGradient<scalar_t, DIMS>(tempPos, s_block, s_domain, GridDataType::VELOCITY);
			}

			for(index_t i=0; i<DIMS ; ++i){
				for(index_t j=i; j<DIMS; ++j){
					scalar_t s = static_cast<scalar_t>(0.5) * (velocityGrads.a[i][j] + velocityGrads.a[j][i]);
					s *= s;
					if(i!=j){
						s *= 2; // abusing symmetry with j=i start for inner loop
					}
					d += s;
				}
			}
			d = sqrt(2*d);
		}

		scalar_t delta = 0; // cell size, here max length along the grid axes
		if(s_block.hasTransform){
			// first transform matrix contains the cell size as column vectors
			MatrixSquare<scalar_t, DIMS> transformMetrics = reinterpret_cast<TransformGPU<scalar_t, DIMS>*>(s_block.transform)[flattenIndex(pos, s_block)].M;
			for(index_t i=0; i<DIMS ; ++i){
				// length/magnitude of a column vector
				scalar_t colMag = 0;
				for(index_t rowIdx=0; rowIdx<DIMS; ++rowIdx){
					colMag += transformMetrics.a[rowIdx][i] * transformMetrics.a[rowIdx][i];
				}
				//colMag = _sqrtT<scalar_t>(colMag) no need to compute sqrt for every dimension
				delta = max(delta, colMag); // TODO: rows or columns? columns
			}
			//delta = _sqrtT<scalar_t>(delta); is squared later anyway
		} else {
			delta = 1;
		}

		pp_blockViscosity_out[targetBlockIdx][flatPos] = coefficient * delta * d; // * delta
	)

}

template<typename scalar_t, int DIMS>
__host__
void _SGSviscosityIncompressibleSmagorinsky(std::shared_ptr<Domain> domain, const torch::Tensor coefficient, std::vector<torch::Tensor> SGSviscosities){
	
	const size_t alignmentBytes = alignof(scalar_t*);
	const size_t atlasSizeBytes = sizeof(scalar_t*) * SGSviscosities.size();
	size_t allocSizeBytes = atlasSizeBytes + alignmentBytes;
	
	auto byteOptions = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided).device(domain->getDevice().type(), domain->getDevice().index());
	auto byteOptionsCPU = torch::TensorOptions().dtype(torch::kUInt8).layout(torch::kStrided);
	
	torch::Tensor t_pointers_SGSviscosities_CPU = torch::zeros(allocSizeBytes, byteOptionsCPU);
	torch::Tensor t_pointers_SGSviscosities_GPU = torch::zeros(allocSizeBytes, byteOptions);
	
	void *p_host = reinterpret_cast<void*>(t_pointers_SGSviscosities_CPU.data_ptr<uint8_t>());
	void *p_device = reinterpret_cast<void*>(t_pointers_SGSviscosities_GPU.data_ptr<uint8_t>());
	
	TORCH_CHECK(std::align(alignmentBytes, atlasSizeBytes, p_host, allocSizeBytes), "Failed to align CPU block viscosity atlas.")
	TORCH_CHECK(std::align(alignmentBytes, atlasSizeBytes, p_device, allocSizeBytes), "Failed to align GPU block viscosity atlas.")
	
	// pointer to host memory containing device pointers
	scalar_t **pp_SGSviscosities_host = reinterpret_cast<scalar_t**>(p_host);
	for(size_t i=0; i<SGSviscosities.size(); ++i){
		pp_SGSviscosities_host[i] = SGSviscosities[i].data_ptr<scalar_t>();
	}
	
	CopyToGPU(p_device, p_host, atlasSizeBytes);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	SETUP_KERNEL_PER_CELL(domain, blockIdxByThreadBlock, threadBlockOffsetInBlock)
	
	// gradient
	BEGIN_SAMPLE;
	k_SGSviscosityIncompressibleSmagorinsky<scalar_t, DIMS><<<blocks, threads>>>(
			reinterpret_cast<DomainGPU<scalar_t>*>(domain->atlas.p_device), coefficient.data_ptr<scalar_t>()[0],
			reinterpret_cast<scalar_t**>(p_device),
			p_blockIdxByThreadBlock, p_threadBlockOffsetInBlock, blockIdxByThreadBlock.size()
		);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	END_SAMPLE("k_SGSviscosityIncompressibleSmagorinsky");
}

std::vector<torch::Tensor> SGSviscosityIncompressibleSmagorinsky(std::shared_ptr<Domain> domain, const torch::Tensor coefficient){
	
	TORCH_CHECK(domain->getNumBlocks()>0, "Domain does not contain blocks.")
	TORCH_CHECK(domain->IsInitialized(), "Domain is not initialized.")
	TORCH_CHECK(!domain->IsTensorChanged(), "Domain's tensors have been changed, use UpdateDomain() to set the new pointers.");
	
	CHECK_INPUT_HOST(coefficient);
	TORCH_CHECK(coefficient.dim()==1 && coefficient.size(0)==1, "coefficient tensor must have shape [1]");
	
	std::vector<torch::Tensor> SGSviscosities;
	for(const auto &block : domain->getBlocks()){
		SGSviscosities.push_back(torch::zeros_like(block->pressure)); //pressure has correct shape with a single channel and must always exist.
	}
	
	DISPATCH_FTYPES_DIMS(domain, "SGSviscosityIncompressibleSmagorinsky", 
		// TODO: allocate tensor for pointers (**scalar_t)
		// gather pointers from SGSviscosities and copy to GPU tensor
		// dispatch k_SGSviscosityIncompressibleSmagorinsky
		_SGSviscosityIncompressibleSmagorinsky<scalar_t, dim>(domain, coefficient, SGSviscosities);
	);
	
	return SGSviscosities;
}

/* --- linear solve --- */

template <typename scalar_t>
solverReturn_t _SolveLinear(std::shared_ptr<CSRmatrix> A, torch::Tensor RHS, torch::Tensor x, torch::Tensor maxit, torch::Tensor tol, const ConvergenceCriterion conv,
		const bool useBiCG, const bool matrixRankDeficient, const index_t residualResetSteps, const bool transposeA, const bool printResidual, const bool returnBestResult,
		const bool withPreconditioner){
	
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	//BEGIN_SAMPLE;
	
	const index_t n = A->getRows();
	const index_t nnz = A->getSize();
	const index_t nBatches = RHS.size(0)/n;
	
	//TORCH_CHECK((nBatches*n)==RHS.size(0), "Linear Solver: A ("+std::to_string(n)+") does not match RHS ("+std::to_string(RHS.size(0))+") and can't be broadcasted.");
	
	//void bicgstabSolveGPU(const scalar_t *aVal, const index_t *aIndex, const index_t aRow, const index_t n, const index_t nnz, const scalar_t *_f, scalar_t *_x, const index_t nBatches=1);
	solverReturn_t ret;
	if(useBiCG){
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		BEGIN_SAMPLE;
		ret = bicgstabSolveGPU<scalar_t>(A->value.data_ptr<scalar_t>(), A->index.data_ptr<index_t>(), A->row.data_ptr<index_t>(), n, nnz,
		RHS.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), nBatches,
		withPreconditioner,
		maxit.data_ptr<index_t>()[0], tol.data_ptr<scalar_t>()[0], conv, transposeA);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
		END_SAMPLE("SolveLinear_BiCGstab");
	}/*
	else if(matrixRankDeficient){
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		BEGIN_SAMPLE;
		ret = cgSolvePreconGPU<scalar_t>(A->value.data_ptr<scalar_t>(), A->index.data_ptr<index_t>(), A->row.data_ptr<index_t>(),
			n, nnz, RHS.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), nBatches, maxit.data_ptr<index_t>()[0], tol.data_ptr<scalar_t>()[0], conv,
			residualResetSteps, nullptr, transposeA, printResidual, returnBestResult);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
		END_SAMPLE("SolveLinear_CG");
	}*/
	else{
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
		BEGIN_SAMPLE;
		ret = cgSolveGPU<scalar_t>(A->value.data_ptr<scalar_t>(), A->index.data_ptr<index_t>(), A->row.data_ptr<index_t>(),
			n, nnz, RHS.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), nBatches, maxit.data_ptr<index_t>()[0], tol.data_ptr<scalar_t>()[0], conv,
			matrixRankDeficient, residualResetSteps, nullptr, transposeA, printResidual, returnBestResult);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
		END_SAMPLE("SolveLinear_CG");
	}
	
	//CUDA_CHECK_RETURN(cudaDeviceSynchronize()); 
	//END_SAMPLE("SolveLinear");
	return ret;
}
solverReturn_t SolveLinear(std::shared_ptr<CSRmatrix> A, torch::Tensor RHS, torch::Tensor x, torch::Tensor maxit, torch::Tensor tol, const ConvergenceCriterion conv,
		const bool useBiCG, const bool matrixRankDeficient, const index_t residualResetSteps, const bool transposeA, const bool printResidual, const bool returnBestResult,
		const bool withPreconditioner){
	
	const index_t n = A->getRows();
	
	CHECK_INPUT_CUDA(RHS);
	TORCH_CHECK(RHS.dim()==1, "RHS must be a vector.");
	TORCH_CHECK((RHS.size(0)%n)==0, "Size of RHS must be a multiple of the matrix size.");
	TORCH_CHECK(RHS.scalar_type()==A->getDtype(), "Data type of RHS must match A.");
	
	CHECK_INPUT_CUDA(x);
	TORCH_CHECK(x.dim()==1, "x must be a vector.");
	TORCH_CHECK(x.size(0)==RHS.size(0), "Size of x must match RHS.");
	TORCH_CHECK(x.scalar_type()==A->getDtype(), "Data type of x must match A.");
	
	CHECK_INPUT_HOST(maxit);
	TORCH_CHECK(maxit.dim()==1, "maxit must be 1D.");
	TORCH_CHECK(maxit.size(0)==1, "maxit must be a scalar.");
	TORCH_CHECK(maxit.dtype()==torch_kIndex, "Data type of maxit must be int32.");
	
	CHECK_INPUT_HOST(tol);
	TORCH_CHECK(tol.dim()==1, "tol must be 1D.");
	TORCH_CHECK(tol.size(0)==1, "tol must be a scalar.");
	TORCH_CHECK(tol.scalar_type()==A->getDtype(), "Data type of tol must match A.");
	
	solverReturn_t ret;

	AT_DISPATCH_FLOATING_TYPES(A->getDtype(), "SolveLinear", ([&] {
		ret = _SolveLinear<scalar_t>(A, RHS, x, maxit, tol, conv, useBiCG, matrixRankDeficient, residualResetSteps, transposeA, printResidual, returnBestResult, withPreconditioner);
	}));

	return ret;
}

template <typename scalar_t>
void _SparseOuterProduct(torch::Tensor &a, torch::Tensor &b, std::shared_ptr<CSRmatrix> out_pattern){
	
	const index_t n = out_pattern->getRows();
	const index_t nnz = out_pattern->getSize();
	
	OuterProductToSparseMatrix(a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out_pattern->value.data_ptr<scalar_t>(), out_pattern->index.data_ptr<index_t>(), out_pattern->row.data_ptr<index_t>(), n, nnz);
}
void SparseOuterProduct(torch::Tensor &a, torch::Tensor &b, std::shared_ptr<CSRmatrix> out_pattern){
	const index_t n = out_pattern->getRows();
	
	CHECK_INPUT_CUDA(a);
	TORCH_CHECK(a.dim()==1, "a must be a vector.");
	TORCH_CHECK(a.size(0)==n, "Size of a must match matrix size.");
	TORCH_CHECK(a.scalar_type()==out_pattern->getDtype(), "Data type of a must match C.");
	
	CHECK_INPUT_CUDA(b);
	TORCH_CHECK(b.dim()==1, "b must be a vector.");
	TORCH_CHECK(b.size(0)==n, "Size of b must match matrix size.");
	TORCH_CHECK(b.scalar_type()==out_pattern->getDtype(), "Data type of b must match C.");

	AT_DISPATCH_FLOATING_TYPES(out_pattern->getDtype(), "SparseOuterProduct", ([&] {
		_SparseOuterProduct<scalar_t>(a, b, out_pattern);
	}));
}

/* move to grid_gen.cu?*/

struct GridInfo{
	I4 size;
	I4 stride;
};

__host__ inline GridInfo MakeGridInfo(const index_t sizeX=1, const index_t sizeY=1, const index_t sizeZ=1, const index_t channels=1){
	GridInfo grid;
	memset(&grid, 0, sizeof(GridInfo));
	grid.size.x = sizeX;
	grid.size.y = sizeY;
	grid.size.z = sizeZ;
	grid.size.w = channels;
	grid.stride.x = 1;
	grid.stride.y = sizeX;
	grid.stride.z = sizeX*sizeY;
	grid.stride.w = sizeX*sizeY*sizeZ;
	return grid;
}

template<typename scalar_t, int DIMS>
__global__ void k_TransformVectors(const scalar_t *p_vectors, const GridInfo gridInfo, TransformGPU<scalar_t,DIMS> *p_transforms, scalar_t *p_vectorsTransformed, const bool inverse){
    for(index_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
        const I4 pos = unflattenIndex(flatIdx, gridInfo.size, gridInfo.stride);
        TransformGPU<scalar_t,DIMS> *p_T = p_transforms + flatIdx;

		MatrixSquare<scalar_t,DIMS> t = inverse ? p_T->Minv : p_T->M;

		Vector<scalar_t,DIMS> v;
		for(index_t dim=0; dim<DIMS; ++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			v.a[dim] = p_vectors[flattenIndex(tempPos, gridInfo.stride)];
		}

		v = matmul(t, v);

		for(index_t dim=0; dim<DIMS; ++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			p_vectorsTransformed[flattenIndex(tempPos, gridInfo.stride)] = v.a[dim];
		}
		
	}
}

#ifdef DISPATCH_FTYPES_DIMS
#undef DISPATCH_FTYPES_DIMS
#endif
#define DISPATCH_FTYPES_DIMS(FTYPE, DIMS, NAME, ...) \
	AT_DISPATCH_FLOATING_TYPES(FTYPE, NAME, ([&] { \
		SWITCH_DIMS(DIMS, \
			__VA_ARGS__; \
		); \
	}));

torch::Tensor TransformVectors(const torch::Tensor &vectors, const torch::Tensor &transforms, const bool inverse){
	// vectors: NCDHW
	// transforms: NDHWT
    CHECK_INPUT_CUDA(vectors);
	TORCH_CHECK(2<vectors.dim() && vectors.dim()<6, "vectors must have batch and channel dimension and be 1-3D.");
	TORCH_CHECK(vectors.size(0)==1, "vectors batch dimension must be 1.");
	index_t dims = vectors.dim()-2;
	TORCH_CHECK(vectors.size(1)==dims, "vectors channel dimension must match spatial dimensionality.");
	
    CHECK_INPUT_CUDA(transforms);
	TORCH_CHECK(transforms.dim() == vectors.dim(), "dimensionality of transforms must match vectors.");
	TORCH_CHECK(transforms.size(0)==1, "transforms batch dimension must be 1.");
	TORCH_CHECK(transforms.size(-1)==TransformNumValues(dims), "transforms channels must match spatial dimensions.");
	for(index_t dim=1; dim<dims+1; ++dim){
		TORCH_CHECK(transforms.size(dim) == vectors.size(dim+1), "spatial dimensions of transforms must match vectors.");
	}
	
	
	const GridInfo gridInfo = MakeGridInfo(vectors.size(-1), dims>1?vectors.size(-2):1, dims>2?vectors.size(-3):1, vectors.size(1));
	
    auto valueOptions = torch::TensorOptions().dtype(vectors.scalar_type()).layout(torch::kStrided).device(vectors.device().type(), vectors.device().index());
	
    torch::Tensor transformedVectors = torch::zeros_like(vectors);
    //const index_t totalSize = sizeX*sizeY; ==grid.strides.w
    
	DISPATCH_FTYPES_DIMS(vectors.scalar_type(), dims, "TransformVectors",
		int minGridSize = 0, blockSize = 0, gridSize = 0;
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_TransformVectors<scalar_t, dim>, 0, 0);
	    gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
		k_TransformVectors<scalar_t, dim><<<gridSize, blockSize>>>(
			vectors.data_ptr<scalar_t>(), gridInfo, reinterpret_cast<TransformGPU<scalar_t,dim>*>(transforms.data_ptr<scalar_t>()), transformedVectors.data_ptr<scalar_t>(), inverse
		);
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	);
	
	return transformedVectors;
}
