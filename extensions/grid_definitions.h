#pragma once
#ifndef GRID_DEFINITIONS
#define GRID_DEFINITIONS

#include "custom_types.h"
#include "transformations.h"
#include <iostream>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define HOST __host__
#else
#define HOST
#define HOST_DEVICE
#endif

using I4 = Vector<int32_t, 4>;
using F4 = Vector<float, 4>;

struct GridInfo{
	I4 size;
	I4 stride;
};

HOST
inline GridInfo MakeGridInfo(const index_t sizeX=1, const index_t sizeY=1, const index_t sizeZ=1, const index_t channels=1){
	GridInfo grid;
	grid.size = {.a={0}};
	grid.stride= {.a={0}};
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



HOST_DEVICE 
inline I4 unflattenIndex(const index_t idx, const I4 &size, const I4 &stride){
	return {.a={idx%size.x, (idx/stride.y)%size.y, (idx/stride.z)%size.z, (idx/stride.w)%size.w}};
}
HOST_DEVICE 
inline I4 unflattenIndex(const index_t idx, const GridInfo &grid){
	return unflattenIndex(idx, grid.size, grid.stride);
}

HOST_DEVICE 
inline index_t flattenIndex(const I4 &pos, const I4 &stride){
	return pos.x + stride.y*pos.y + stride.z*pos.z + stride.w*pos.w;
}

HOST_DEVICE 
inline index_t flattenIndex(const I4 &pos, const GridInfo &grid){
	//return pos.x + block.stride.y*pos.y + block.stride.z*pos.z + block.stride.w*pos.w;
	return flattenIndex(pos, grid.stride);
}

template<typename scalar_t>
inline HOST_DEVICE scalar_t frac(const scalar_t v){
	return v - floor(v);
}



#undef HOST_DEVICE
#undef HOST

#endif // GRID_DEFINITIONS