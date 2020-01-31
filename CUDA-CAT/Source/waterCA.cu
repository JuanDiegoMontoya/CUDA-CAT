#include "stdafx.h"
#include "waterCA.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

WCA::Cell* WCA::Grid;
WCA::Cell* WCA::TGrid;

namespace WCA
{
	__device__
	glm::ivec3 expand(unsigned index)
	{
		int k = index / (GRID_SIZE_X * GRID_SIZE_Y);
		int j = (index % (GRID_SIZE_X * GRID_SIZE_Y)) / GRID_SIZE_X;
		int i = index - j * GRID_SIZE_X - k * GRID_SIZE_X * GRID_SIZE_Y;
		return { i, j, k };
	}


	__device__
	int flatten(glm::ivec3 coord)
	{
		return coord.x + coord.y * GRID_SIZE_X + coord.z * GRID_SIZE_X * GRID_SIZE_Y;
	}


	// updates grid using water rules
	// writes results to tempGrid to avoid race condition
	__global__
	void updateGrid(Cell* grid, Cell* tempGrid, int n)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n; i += stride)
		{
			// get coord from index
			glm::ivec3 pos = expand(i);

		}

	}


	void InitWCA()
	{
		cudaMallocManaged(&Grid, CELL_COUNT * sizeof(Cell));
		cudaMallocManaged(&TGrid, CELL_COUNT * sizeof(Cell));

		// populate the grid with water and walls
		for (int z = 0; z < GRID_SIZE_Z; z++)
		{
			int zPart = z * GRID_SIZE_Y * GRID_SIZE_Z;
			for (int y = 0; y < GRID_SIZE_Y; y++)
			{
				int yzPart = y * GRID_SIZE_Y + zPart;
				for (int x = 0; x < GRID_SIZE_X; x++)
				{
					// compute final part of flattened index
					int index = x + yzPart;

					// place walls at edges
					if (
						x == GRID_SIZE_X - 1 || x == 0 ||
						y == GRID_SIZE_Y - 1 || y == 0 ||
						z == GRID_SIZE_Z - 1 || z == 0)
						Grid[index].isWall_ = true;
				}
			}
		}
	}


	void UpdateWCA()
	{
		updateGrid<<<numBlocks, blockSize>>>(Grid, TGrid, CELL_COUNT);
		cudaDeviceSynchronize();

		// TGrid contains updated grid values after update
		// we no longer care about values in Grid anymore
		// a simple swap of pointers will suffice
		std::swap(Grid, TGrid);
	}


	void ShutdownWCA()
	{
		cudaFree(Grid);
		cudaFree(TGrid);
	}
}