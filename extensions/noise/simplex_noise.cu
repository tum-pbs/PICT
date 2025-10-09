
#include "simplex_noise.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include "simplex_noise.h"
#include "../dispatch.h"

template<typename scalar_t, int32_t DIMS>
__global__
void k_GenerateSimplexNoiseVariation_Simplex(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset){
	
	for(int32_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const I4 pos = unflattenIndex(flatIdx, gridInfo);
		
		Vector<scalar_t, DIMS> noisePos{.a={0}};
		for(int32_t dim=0;dim<DIMS;++dim){
			noisePos.a[dim] = pos.a[dim];
		}
		noisePos = noisePos * scale + offset;
		
		const scalar_t value = snoise(noisePos);
		
		grid[flatIdx] = value;
	}
	
}

template<typename scalar_t, int32_t DIMS>
inline __device__
scalar_t fBm(const Vector<scalar_t, DIMS> pos, const int32_t octaves, const scalar_t lacunarity, const scalar_t gain){
	
	scalar_t sum = 0;
	scalar_t freq = 1;
	scalar_t amp = 0.5;
	for (uint32_t i = 0; i<octaves; i++){
		scalar_t n = snoise(pos * freq);
		sum += n*amp;
		freq *= lacunarity;
		amp *= gain;
	}
	
	return sum;
}

template<typename scalar_t, int32_t DIMS>
__global__
void k_GenerateSimplexNoiseVariation_FBM(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset,
		const int32_t octaves, const scalar_t lacunarity, const scalar_t gain){
	
	for(int32_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const I4 pos = unflattenIndex(flatIdx, gridInfo);
		
		Vector<scalar_t, DIMS> noisePos{.a={0}};
		for(int32_t dim=0;dim<DIMS;++dim){
			noisePos.a[dim] = pos.a[dim];
		}
		noisePos = noisePos * scale + offset;
		
		const scalar_t noise = fBm(noisePos, octaves, lacunarity, gain);
		
		grid[flatIdx] = noise;
	}
	
}

template<typename scalar_t, int32_t DIMS>
__global__
void k_GenerateSimplexNoiseVariation_Ridged(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset){
	
	for(int32_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const I4 pos = unflattenIndex(flatIdx, gridInfo);
		
		Vector<scalar_t, DIMS> noisePos{.a={0}};
		for(int32_t dim=0;dim<DIMS;++dim){
			noisePos.a[dim] = pos.a[dim];
		}
		noisePos = noisePos * scale + offset;
		
		const scalar_t value = 1.0 - fabs(snoise(noisePos));
		
		grid[flatIdx] = value;
	}
	
}

template<typename scalar_t>
inline __device__
scalar_t ridge(scalar_t h, const scalar_t offset){
	h = offset - fabs(h);
	return h*h;
}

template<typename scalar_t, int32_t DIMS>
__global__
void k_GenerateSimplexNoiseVariation_RidgedMF(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset,
		const scalar_t ridgeOffset, const int32_t octaves, const scalar_t lacunarity, const scalar_t gain){
	
	for(int32_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const I4 pos = unflattenIndex(flatIdx, gridInfo);
		
		Vector<scalar_t, DIMS> noisePos{.a={0}};
		for(int32_t dim=0;dim<DIMS;++dim){
			noisePos.a[dim] = pos.a[dim];
		}
		noisePos = noisePos * scale + offset;
		
		
		scalar_t sum = 0;
		scalar_t freq = 1;
		scalar_t amp = 0.5;
		scalar_t prev = 1;
		for (uint32_t i = 0; i<octaves; i++){
			scalar_t n = ridge(snoise(noisePos * freq), ridgeOffset);
			sum += n*amp*prev;
			prev = n;
			freq *= lacunarity;
			amp *= gain;
		}
		
		
		grid[flatIdx] = sum;
	}
	
}

const double numEps = 1e-4;

template<typename scalar_t, int32_t DIMS>
inline __device__
Vector<scalar_t, DIMS> snoiseGradientNumerical(const Vector<scalar_t, DIMS> pos, const scalar_t eps){
	
	const scalar_t epsNorm = 1.0/(2.0*eps);
	Vector<scalar_t, DIMS> noiseGrad{.a={0}};
	for(int32_t i=0; i<DIMS; ++i){
		Vector<scalar_t, DIMS> tempPos = pos;
		tempPos.a[i] += eps;
		noiseGrad.a[i] = snoise(tempPos);
		tempPos.a[i] = pos.a[i] - eps;
		noiseGrad.a[i] -= snoise(tempPos);
		noiseGrad.a[i] *= epsNorm;
	}
	
	return noiseGrad;
}

template<typename scalar_t, int32_t DIMS>
__global__
void k_GenerateSimplexNoiseVariation_GradientNumerical(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset){
	
	for(int32_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const I4 pos = unflattenIndex(flatIdx, gridInfo);
		
		Vector<scalar_t, DIMS> noisePos{.a={0}};
		for(int32_t dim=0;dim<DIMS;++dim){
			noisePos.a[dim] = pos.a[dim];
		}
		noisePos = noisePos * scale + offset;
		
		const Vector<scalar_t, DIMS> noiseGrad = snoiseGradientNumerical<scalar_t, DIMS>(noisePos, numEps);
		
		for(int32_t dim=0;dim<DIMS;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const int32_t tempFlatPos = flattenIndex(tempPos, gridInfo);
			grid[tempFlatPos] = noiseGrad.a[dim];
		}
	}
	
}

template<typename scalar_t, int32_t DIMS>
inline __device__
Vector<scalar_t, DIMS> snoiseCurlNumerical(const Vector<scalar_t, DIMS> pos, const scalar_t eps){
	return {0};
}
template<>
inline __device__
Vector<float, 2> snoiseCurlNumerical<float, 2>(const Vector<float, 2> pos, const float eps){
	const Vector<float, 2> noiseGrad = snoiseGradientNumerical<float, 2>(pos, eps);
	return Vector<float, 2>{noiseGrad.y, -noiseGrad.x};
}
template<>
inline __device__
Vector<float, 3> snoiseCurlNumerical<float, 3>(const Vector<float, 3> pos, const float eps){
	// offsets from https://github.com/simongeilfus/SimplexNoise
	// I think the offset is necessary as otherwise curl(grad(snoise))=0
	Vector<float, 3> dx = snoiseGradientNumerical<float, 3>(pos, eps);
	Vector<float, 3> dy = snoiseGradientNumerical<float, 3>(pos + Vector<float, 3>{123.456f, 789.012f, 345.678f}, eps);
	Vector<float, 3> dz = snoiseGradientNumerical<float, 3>(pos + Vector<float, 3>{901.234f, 567.891f, 234.567f}, eps);
	return Vector<float, 3>{dz.y - dy.z, dx.z - dz.x, dy.x - dx.y};
}

template<typename scalar_t, int32_t DIMS>
__global__
void k_GenerateSimplexNoiseVariation_CurlNumerical(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset){
	
	for(int32_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const I4 pos = unflattenIndex(flatIdx, gridInfo);
		
		Vector<scalar_t, DIMS> noisePos{.a={0}};
		for(int32_t dim=0;dim<DIMS;++dim){
			noisePos.a[dim] = pos.a[dim];
		}
		noisePos = noisePos * scale + offset;
		
		const Vector<scalar_t, DIMS> curl = snoiseCurlNumerical<scalar_t, DIMS>(noisePos, numEps);
		
		for(int32_t dim=0;dim<DIMS;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const int32_t tempFlatPos = flattenIndex(tempPos, gridInfo);
			grid[tempFlatPos] = curl.a[dim];
		}
	}
	
}

template<typename scalar_t, int32_t DIMS>
inline __device__
Vector<scalar_t, DIMS> FBMnoiseGradientNumerical(const Vector<scalar_t, DIMS> pos, const scalar_t eps,
		const int32_t octaves, const scalar_t lacunarity, const scalar_t gain){
	
	const scalar_t epsNorm = 1.0/(2.0*eps);
	Vector<scalar_t, DIMS> noiseGrad{.a={0}};
	for(int32_t i=0; i<DIMS; ++i){
		Vector<scalar_t, DIMS> tempPos = pos;
		tempPos.a[i] += eps;
		noiseGrad.a[i] = fBm(tempPos, octaves, lacunarity, gain);
		tempPos.a[i] = pos.a[i] - eps;
		noiseGrad.a[i] -= fBm(tempPos, octaves, lacunarity, gain);
		noiseGrad.a[i] *= epsNorm;
	}
	
	return noiseGrad;
}

template<typename scalar_t, int32_t DIMS>
__global__
void k_GenerateSimplexNoiseVariation_GradientFBMnumerical(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset,
		const int32_t octaves, const scalar_t lacunarity, const scalar_t gain){
	
	for(int32_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const I4 pos = unflattenIndex(flatIdx, gridInfo);
		
		Vector<scalar_t, DIMS> noisePos{.a={0}};
		for(int32_t dim=0;dim<DIMS;++dim){
			noisePos.a[dim] = pos.a[dim];
		}
		noisePos = noisePos * scale + offset;
		
		const Vector<scalar_t, DIMS> noiseGrad = FBMnoiseGradientNumerical<scalar_t, DIMS>(noisePos, numEps, octaves, lacunarity, gain);
		
		for(int32_t dim=0;dim<DIMS;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const int32_t tempFlatPos = flattenIndex(tempPos, gridInfo);
			grid[tempFlatPos] = noiseGrad.a[dim];
		}
	}
	
}

template<typename scalar_t, int32_t DIMS>
inline __device__
Vector<scalar_t, DIMS> FBMnoiseCurlNumerical(const Vector<scalar_t, DIMS> pos, const scalar_t eps,
		const int32_t octaves, const scalar_t lacunarity, const scalar_t gain){
	return {0};
}
template<>
inline __device__
Vector<float, 2> FBMnoiseCurlNumerical<float, 2>(const Vector<float, 2> pos, const float eps,
		const int32_t octaves, const float lacunarity, const float gain){
	const Vector<float, 2> noiseGrad = FBMnoiseGradientNumerical<float, 2>(pos, eps, octaves, lacunarity, gain);
	return Vector<float, 2>{noiseGrad.y, -noiseGrad.x};
}
template<>
inline __device__
Vector<float, 3> FBMnoiseCurlNumerical<float, 3>(const Vector<float, 3> pos, const float eps,
		const int32_t octaves, const float lacunarity, const float gain){
	// offsets from https://github.com/simongeilfus/SimplexNoise
	// I think the offset is necessary as otherwise curl(grad(snoise))=0
	Vector<float, 3> dx = FBMnoiseGradientNumerical<float, 3>(pos, eps, octaves, lacunarity, gain);
	Vector<float, 3> dy = FBMnoiseGradientNumerical<float, 3>(pos + Vector<float, 3>{123.456f, 789.012f, 345.678f}, eps, octaves, lacunarity, gain);
	Vector<float, 3> dz = FBMnoiseGradientNumerical<float, 3>(pos + Vector<float, 3>{901.234f, 567.891f, 234.567f}, eps, octaves, lacunarity, gain);
	return Vector<float, 3>{dz.y - dy.z, dx.z - dz.x, dy.x - dx.y};
}

template<typename scalar_t, int32_t DIMS>
__global__
void k_GenerateSimplexNoiseVariation_CurlFBMnumerical(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset,
		const int32_t octaves, const scalar_t lacunarity, const scalar_t gain){
	
	for(int32_t flatIdx = blockIdx.x * blockDim.x + threadIdx.x; flatIdx < gridInfo.stride.w; flatIdx += blockDim.x * gridDim.x){
		
		const I4 pos = unflattenIndex(flatIdx, gridInfo);
		
		Vector<scalar_t, DIMS> noisePos{.a={0}};
		for(int32_t dim=0;dim<DIMS;++dim){
			noisePos.a[dim] = pos.a[dim];
		}
		noisePos = noisePos * scale + offset;
		
		const Vector<scalar_t, DIMS> curl = FBMnoiseCurlNumerical<scalar_t, DIMS>(noisePos, numEps, octaves, lacunarity, gain);
		
		for(int32_t dim=0;dim<DIMS;++dim){
			I4 tempPos = pos;
			tempPos.w = dim;
			const int32_t tempFlatPos = flattenIndex(tempPos, gridInfo);
			grid[tempFlatPos] = curl.a[dim];
		}
	}
	
}

using namespace SimplexNoise;

template<typename scalar_t, int32_t DIMS>
__host__
void GenerateSimplexNoiseVariation_Switch(scalar_t *grid, const GridInfo gridInfo, const Vector<scalar_t, DIMS> scale, const Vector<scalar_t, DIMS> offset,
		const NoiseVariation variation, const float ridgeOffset, const int32_t octaves, const float lacunarity, const float gain){
	
	switch(variation){
		case NoiseVariation::SIMPLEX:
		{
			int minGridSize = 0, blockSize = 0, gridSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_GenerateSimplexNoiseVariation_Simplex<scalar_t, DIMS>, 0, 0);
			gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
			k_GenerateSimplexNoiseVariation_Simplex<scalar_t, DIMS><<<gridSize, blockSize>>>(
				grid, gridInfo, scale, offset
			);
			break;
		}
		case NoiseVariation::FRACTAL_BROWNIAN_MOTION:
		{
			int minGridSize = 0, blockSize = 0, gridSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_GenerateSimplexNoiseVariation_FBM<scalar_t, DIMS>, 0, 0);
			gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
			k_GenerateSimplexNoiseVariation_FBM<scalar_t, DIMS><<<gridSize, blockSize>>>(
				grid, gridInfo, scale, offset, octaves, lacunarity, gain
			);
			break;
		}
		case NoiseVariation::RIDGED:
		{
			int minGridSize = 0, blockSize = 0, gridSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_GenerateSimplexNoiseVariation_Ridged<scalar_t, DIMS>, 0, 0);
			gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
			k_GenerateSimplexNoiseVariation_Ridged<scalar_t, DIMS><<<gridSize, blockSize>>>(
				grid, gridInfo, scale, offset
			);
			break;
		}
		case NoiseVariation::RIDGED_MULTI_FRACTAL:
		{
			int minGridSize = 0, blockSize = 0, gridSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_GenerateSimplexNoiseVariation_RidgedMF<scalar_t, DIMS>, 0, 0);
			gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
			k_GenerateSimplexNoiseVariation_RidgedMF<scalar_t, DIMS><<<gridSize, blockSize>>>(
				grid, gridInfo, scale, offset, ridgeOffset, octaves, lacunarity, gain
			);
			break;
		}
		case NoiseVariation::GRADIENT:
		{
			int minGridSize = 0, blockSize = 0, gridSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_GenerateSimplexNoiseVariation_GradientNumerical<scalar_t, DIMS>, 0, 0);
			gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
			k_GenerateSimplexNoiseVariation_GradientNumerical<scalar_t, DIMS><<<gridSize, blockSize>>>(
				grid, gridInfo, scale, offset
			);
			break;
		}
		case NoiseVariation::GRADIENT_FBM:
		{
			int minGridSize = 0, blockSize = 0, gridSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_GenerateSimplexNoiseVariation_GradientFBMnumerical<scalar_t, DIMS>, 0, 0);
			gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
			k_GenerateSimplexNoiseVariation_GradientFBMnumerical<scalar_t, DIMS><<<gridSize, blockSize>>>(
				grid, gridInfo, scale, offset, octaves, lacunarity, gain
			);
			break;
		}
		case NoiseVariation::CURL:
		{
			int minGridSize = 0, blockSize = 0, gridSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_GenerateSimplexNoiseVariation_CurlNumerical<scalar_t, DIMS>, 0, 0);
			gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
			k_GenerateSimplexNoiseVariation_CurlNumerical<scalar_t, DIMS><<<gridSize, blockSize>>>(
				grid, gridInfo, scale, offset
			);
			break;
		}
		case NoiseVariation::CURL_FBM:
		{
			int minGridSize = 0, blockSize = 0, gridSize = 0;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, k_GenerateSimplexNoiseVariation_CurlFBMnumerical<scalar_t, DIMS>, 0, 0);
			gridSize = (gridInfo.stride.w + blockSize - 1) / blockSize;
			k_GenerateSimplexNoiseVariation_CurlFBMnumerical<scalar_t, DIMS><<<gridSize, blockSize>>>(
				grid, gridInfo, scale, offset, octaves, lacunarity, gain
			);
			break;
		}
		default:
			break;
	}
}

torch::Tensor SimplexNoise::GenerateSimplexNoiseVariation(const std::vector<int32_t> outputShape, const torch::Device GPUdevice,
		const std::vector<float> scale, const std::vector<float> offset,
		const NoiseVariation variation, const float ridgeOffset, const int32_t octaves, const float lacunarity, const float gain){
	
	const size_t dims = outputShape.size();
	//TORCH_CHECK(dims==2, "Only 2D is supported.");
	TORCH_CHECK(dims==scale.size(), "size of scale must match output shape.");
	TORCH_CHECK(dims==offset.size(), "size of offset must match output shape.");
	for(size_t dim=0;dim<dims;++dim){
		TORCH_CHECK(outputShape[dim]>0, "output shape dimensions must be positive.");
	}
	
	int32_t channels = 1;
	if(variation==NoiseVariation::GRADIENT || variation==NoiseVariation::GRADIENT_FBM || variation==NoiseVariation::CURL || variation==NoiseVariation::CURL_FBM){
		channels = dims;
	}
	
	const GridInfo grid = MakeGridInfo(outputShape[0], dims>1?outputShape[1]:1, dims>2?outputShape[2]:1, channels);
	
	const torch::Dtype torch_kScalar = torch::kFloat32;
    auto valueOptions = torch::TensorOptions().dtype(torch_kScalar).layout(torch::kStrided).device(GPUdevice.type(), GPUdevice.index());
	
	std::vector<int64_t> tensorSize;
	tensorSize.push_back(1); // batch size
	tensorSize.push_back(channels); // channels
	for(int32_t dim=dims-1;dim>=0;--dim){
		tensorSize.push_back(grid.size.a[dim]); //spatial dimension, zyx
	}
	
    torch::Tensor tensor = torch::zeros(tensorSize, valueOptions);
	
	float *p_tensor_data = tensor.data_ptr<float>();
	
	switch(dims){
		case 2:
		{
			Vector<float, 2> s{.a={scale[0], scale[1]}};
			Vector<float, 2> o{.a={offset[0], offset[1]}};
			GenerateSimplexNoiseVariation_Switch<float, 2>(p_tensor_data, grid, s, o,
				variation, ridgeOffset, octaves, lacunarity, gain);
			break;
		}
		case 3:
		{
			Vector<float, 3> s{.a={scale[0], scale[1], scale[2]}};
			Vector<float, 3> o{.a={offset[0], offset[1], offset[2]}};
			GenerateSimplexNoiseVariation_Switch<float, 3>(p_tensor_data, grid, s, o,
				variation, ridgeOffset, octaves, lacunarity, gain);
			break;
		}
		default:
			break;
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	
	return tensor;
}