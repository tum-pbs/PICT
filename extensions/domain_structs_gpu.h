#pragma once

#ifndef _INCLUDE_DOMAIN_STRUCTS_GPU
#define _INCLUDE_DOMAIN_STRUCTS_GPU

#include "custom_types.h"

#define ALIGN_UP(addr, align) (((addr) + (align-1) ) & ~(align-1))
#define ALIGN_ADDR_UP(addr, align) ((reinterpret_cast<uintptr_t>(addr) + static_cast<uintptr_t>(align-1) ) & ~static_cast<uintptr_t>(align-1))

#define WITH_GRAD

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

/** 
 * The different data types (velocity, pressure, passive scalar) and their various fields (per-block data, RHS, global result).
 * Organized to allow to use this for indexing (in order passive scalar, pressure [div], velocity, [pressure RHS]).
 * Allows flag checking for:
 * - scalar(pressure , passive scalar)/vector(velocity, pressureRHS)
 * - block data/rhs/result
 * - local(block data)/global(rhs, result) for position indexing
 * - fwd/gradient
 */
enum class GridDataType : int8_t{
	// ordered to use for indexing
	_INDEX_MASK    = 3, // first 2 bits are used for indexing
	_IS_VECTOR     = 2, // 2nd bit indicates that this is a vector quantitiy
	PASSIVE_SCALAR = 0, // 0b00000
	PRESSURE       = 1, // 0b00001
	VELOCITY       = 2, // 0b00010
	
	_IS_RESULT     = 4, //3rd bit indicates using the global result field
	PASSIVE_SCALAR_RESULT = PASSIVE_SCALAR | _IS_RESULT, // 0b00100
	PRESSURE_RESULT       = PRESSURE       | _IS_RESULT, // 0b00101
	VELOCITY_RESULT       = VELOCITY       | _IS_RESULT, // 0b00110
	
	_IS_RHS        = 8, //4th bit indicates using the global RHS field
	PASSIVE_SCALAR_RHS = PASSIVE_SCALAR | _IS_RHS, // 0b01000
	PRESSURE_RHS_DIV   = PRESSURE       | _IS_RHS, // 0b01001
	VELOCITY_RHS       = VELOCITY       | _IS_RHS, // 0b01010
	PRESSURE_RHS       = PRESSURE | VELOCITY | _IS_RHS, // 0b01011
	
	_IS_GLOBAL     = _IS_RESULT | _IS_RHS, //3rd and 4th bit indicate this being a global field (result or RHS)
	
	_IS_GRAD       = 16, //5th bit indicates the gradient fields
	PASSIVE_SCALAR_GRAD = PASSIVE_SCALAR | _IS_GRAD, // 0b10000
	PRESSURE_GRAD       = PRESSURE       | _IS_GRAD, // 0b10001
	VELOCITY_GRAD       = VELOCITY       | _IS_GRAD, // 0b10010
	
	PASSIVE_SCALAR_RESULT_GRAD = PASSIVE_SCALAR | _IS_RESULT | _IS_GRAD, // 0b10100
	PRESSURE_RESULT_GRAD       = PRESSURE       | _IS_RESULT | _IS_GRAD, // 0b10101
	VELOCITY_RESULT_GRAD       = VELOCITY       | _IS_RESULT | _IS_GRAD, // 0b10110
	
	PASSIVE_SCALAR_RHS_GRAD = PASSIVE_SCALAR | _IS_RHS | _IS_GRAD, // 0b11000
	PRESSURE_RHS_DIV_GRAD   = PRESSURE       | _IS_RHS | _IS_GRAD, // 0b11001
	VELOCITY_RHS_GRAD       = VELOCITY       | _IS_RHS | _IS_GRAD, // 0b11010
	PRESSURE_RHS_GRAD       = PRESSURE | VELOCITY | _IS_RHS | _IS_GRAD, // 0b11011
	
	IS_FIXED_BOUNDARY = 32 //6th bit auxiliary for boundary detection, should return 1 if the value would be read from a fixed boundary, 0 otherwise
};
HOST_DEVICE constexpr
bool isVectorDataType(const GridDataType type){
	return static_cast<int8_t>(type) & static_cast<int8_t>(GridDataType::_IS_VECTOR);
	//return !isScalarDataType(type);
}
HOST_DEVICE constexpr
bool isScalarDataType(const GridDataType type){
	return !isVectorDataType(type);
	//return gridDataTypeToBaseType(type)!=GridDataType::PRESSURE;
}
HOST_DEVICE constexpr
bool isResultDataType(const GridDataType type){
	return static_cast<int8_t>(type) & static_cast<int8_t>(GridDataType::_IS_RESULT);
}
HOST_DEVICE constexpr
bool isRHSDataType(const GridDataType type){
	return static_cast<int8_t>(type) & static_cast<int8_t>(GridDataType::_IS_RHS);
}
HOST_DEVICE constexpr
bool isGlobalDataType(const GridDataType type){
	return static_cast<int8_t>(type) & static_cast<int8_t>(GridDataType::_IS_GLOBAL);
}
HOST_DEVICE constexpr
bool isGradDataType(const GridDataType type){
	return static_cast<int8_t>(type) & static_cast<int8_t>(GridDataType::_IS_GRAD);
}
HOST_DEVICE constexpr
index_t gridDataTypeToIndex(const GridDataType type){
	return static_cast<int8_t>(type) & static_cast<int8_t>(GridDataType::_INDEX_MASK);
}
HOST_DEVICE constexpr
GridDataType gridDataTypeToBaseType(const GridDataType type){
	return static_cast<GridDataType>(static_cast<int8_t>(type) & static_cast<int8_t>(GridDataType::_INDEX_MASK));
}
HOST_DEVICE constexpr
GridDataType gridDataTypeWithoutGrad(const GridDataType type){
	return static_cast<GridDataType>(static_cast<int8_t>(type) & (~static_cast<int8_t>(GridDataType::_IS_GRAD)));
}

enum class BoundaryType : int8_t{
	DIRICHLET=0,
	FIRST_TYPE=0,
	VALUE=0,
	DIRICHLET_VARYING=1,
	
	NEUMANN=10,
	SECOND_TYPE=10,
	GRADIENT=10,
	
	FIXED=30, // TODO: replace DIRICHLET and NEUMANN
	
	CONNECTED_GRID=20,
	PERIODIC=21
	
};

#include "transformations.h"

template <typename scalar_t>
using S4 = Vector<scalar_t, 4>;

using I4 = S4<int32_t>;
using U4 = S4<dim_t>;
using F4 = S4<float>;
using D4 = S4<double>;

template <typename scalar_t>
inline S4<scalar_t> makeS4(const scalar_t x = 0, const scalar_t y = 0, const scalar_t z = 0, const scalar_t w = 0){
	//return {{.x=x, .y=y, .z=z, .w=w}};
	return {{x, y, z, w}};
}
const auto makeI4 = makeS4<int32_t>;
const auto makeU4 = makeS4<dim_t>;
const auto makeF4 = makeS4<float>;
const auto makeD4 = makeS4<double>;


// structs

template <typename scalar_t, int DIM>
struct TransformGPU{
	MatrixSquare<scalar_t, DIM> M;
	MatrixSquare<scalar_t, DIM> Minv;
	scalar_t det;
};
inline index_t TransformNumValues(const index_t nDims){
	return nDims*nDims*2+1;
}

template <typename scalar_t>
struct CSRmatrixGPU{
	scalar_t *value;
	index_t *index;
	index_t *row;
};

/** Unified version of all prescibed boundaries.*/
using BoundaryConditionType_base_type = int8_t; // linker/torch does not like std::underlying_type<BoundaryConditionType>
enum class BoundaryConditionType : BoundaryConditionType_base_type{
	DIRICHLET=0,
	FIRST_TYPE=0,
	VALUE=0,
	
	NEUMANN=1,
	SECOND_TYPE=1,
	GRADIENT=1
};


template <typename scalar_t> //, intdex_t DIMS>
struct FixedBoundaryDataGPU {
	union{
		BoundaryConditionType boundaryType; // for velocity
		BoundaryConditionType *p_boundaryTypes; // for multi-channel passive scalar
	};
	
	bool isStaticType;
	bool isStatic;
	// uint32_t flags; // isStatic for data&grad and boundaryType
	// if static, index p_data with only pos.w, otherwise with flattenIndex(pos, stride)
	scalar_t *data;
#ifdef WITH_GRAD
	scalar_t *grad;
#endif //WITH_GRAD
	
};

template <typename scalar_t>
struct FixedBoundaryGPU{
	I4 size;
	I4 stride;
	union{
		struct {
			FixedBoundaryDataGPU<scalar_t> passiveScalar;
			FixedBoundaryDataGPU<scalar_t> pressure; // unused, but needed for indexing
			FixedBoundaryDataGPU<scalar_t> velocity;
		};
		FixedBoundaryDataGPU<scalar_t> data[3]; // to index with GridDataType
	};
	bool hasTransform;
	scalar_t *transform;
};

template <typename scalar_t>
struct StaticDirichletBoundaryGPU{
	scalar_t slip;
	S4<scalar_t> velocity;
	scalar_t scalar;
#ifdef WITH_GRAD
	//scalar_t *slip_grad;
	scalar_t *velocity_grad;
	scalar_t *scalar_grad;
#endif //WITH_GRAD
	//bool hasTransform;
	//scalar_t *transform;
};

template <typename scalar_t>
struct StaticNeumannBoundaryGPU{
	scalar_t slip;
	S4<scalar_t> boundaryGradient;
	scalar_t scalarGradient;
};

template <typename scalar_t>
struct VaryingDirichletBoundaryGPU{
	scalar_t slip;
	I4 size;
	I4 stride;
	scalar_t *velocity;
	scalar_t *scalar;
#ifdef WITH_GRAD
	//scalar_t *slip_grad;
	scalar_t *velocity_grad;
	scalar_t *scalar_grad;
#endif //WITH_GRAD
	bool hasTransform;
	scalar_t *transform;
};

template <typename scalar_t>
struct ConnectedBoundaryGPU{
	index_t connectedGridIndex;
	// dim_t connectedFace;
	// dim_t connectedAxis1;
	// dim_t connectedAxis2;
	U4 axes;
};

template <typename scalar_t>
struct PeriodicBoundaryGPU{
};

template <typename scalar_t>
struct BoundaryGPU{
	BoundaryType type;
	union{
		FixedBoundaryGPU<scalar_t> fb;
		StaticDirichletBoundaryGPU<scalar_t> sdb;
		VaryingDirichletBoundaryGPU<scalar_t> vdb;
		StaticNeumannBoundaryGPU<scalar_t> snb;
		ConnectedBoundaryGPU<scalar_t> cb;
		PeriodicBoundaryGPU<scalar_t> pb;
	};
};

template <typename scalar_t>
struct BlockGPU{
	index_t globalOffset; //offset in cells from first block start. used e.g. for CSR indices
	index_t csrOffset;
	I4 size;
	I4 stride;
	scalar_t *viscosity;
	bool isViscosityStatic;
	union{
		struct {
			scalar_t *scalarData;
			scalar_t *pressure;
			scalar_t *velocity;
		};
		scalar_t *data[3]; // to index with GridDataType
	};
	scalar_t *velocitySource;
	bool isVelocitySourceStatic;
	//scalar_t *velocityUpdate;
#ifdef WITH_GRAD
	scalar_t *viscosity_grad;
	union{
		struct {
			scalar_t *scalarData_grad;
			scalar_t *pressure_grad;
			scalar_t *velocity_grad;
		};
		scalar_t *grad[3]; // to index with GridDataType
	};
	scalar_t *velocitySource_grad;
	//scalar_t *velocityUpdate_grad;
#endif
	BoundaryGPU<scalar_t> boundaries[6];
	bool hasTransform;
	scalar_t *transform;
	bool hasFaceTransform;
	scalar_t *faceTransform;
};

template <typename scalar_t>
struct DomainGPU{
	index_t numBlocks;
	index_t numCells; //totalSize, globalSize
	BlockGPU<scalar_t> *blocks;
	//scalar_t timeStep;
	dim_t numDims;
	index_t passiveScalarChannels;
	
	scalar_t viscosity;
	scalar_t *scalarViscosity;
	bool scalarViscosityStatic;
	
	CSRmatrixGPU<scalar_t> C;
	scalar_t *Adiag;
	CSRmatrixGPU<scalar_t> P;

	union{
		struct{
			scalar_t *scalarRHS;
			scalar_t *pressureRHSdiv;
			scalar_t *velocityRHS;
			scalar_t *pressureRHS;
		};
		scalar_t *RHS[4];
	};
	
	union{
		struct{
			scalar_t *scalarResult;
			scalar_t *pressureResult;
			scalar_t *velocityResult;
		};
		scalar_t *results[3];
	};

#ifdef WITH_GRAD
	scalar_t *viscosity_grad;
	scalar_t *scalarViscosity_grad;
	
	CSRmatrixGPU<scalar_t> C_grad;
	scalar_t *Adiag_grad;
	CSRmatrixGPU<scalar_t> P_grad;
	
	union{
		struct{
			scalar_t *scalarRHS_grad;
			scalar_t *pressureRHSdiv_grad;
			scalar_t *velocityRHS_grad;
			scalar_t *pressureRHS_grad;
		};
		scalar_t *RHS_grad[4];
	};
	
	union{
		struct{
			scalar_t *scalarResult_grad;
			scalar_t *pressureResult_grad;
			scalar_t *velocityResult_grad;
		};
		scalar_t *results_grad[3];
	};

#endif
};

struct DomainAtlasSet{
	size_t sizeBytes;
	size_t blocksOffsetBytes;
	void* p_host;
	void* p_device;
};

void CopyToGPU(void *p_dst, const void *p_src, const size_t bytes);
void CopyDomainToGPU(const DomainAtlasSet &domainAtlas);

#ifdef HOST_DEVICE
#undef HOST_DEVICE
#endif

#endif //_INCLUDE_DOMAIN_STRUCTS_GPU