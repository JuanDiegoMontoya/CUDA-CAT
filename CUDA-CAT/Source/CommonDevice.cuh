#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<int X, int Y>
__device__ glm::ivec3 expand(unsigned index)
{
	int k = index / (X * Y);
	int j = (index % (X * Y)) / X;
	int i = index - j * X - k * X * Y;
	return { i, j, k };
}

template<int X, int Y>
__device__ int flatten(glm::ivec3 coord)
{
	return coord.x + coord.y * X + coord.z * X * Y;
}

template<int X, int Y, int Z>
__device__ bool inBoundary(glm::ivec3 p)
{
	return
		p.x >= 0 && p.x < X &&
		p.y >= 0 && p.y < Y &&
		p.z >= 0 && p.z < Z;
}

__device__
bool inBound(int a, int b)
{
	return a >= 0 && a < b;
}