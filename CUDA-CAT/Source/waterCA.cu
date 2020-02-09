#include "stdafx.h"
#include "waterCA.h"
#include "CAMesh.h"

//render
#include "shader.h"
#include "camera.h"
#include "pipeline.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

WCA::Cell* WCA::Grid;
WCA::Cell* WCA::TGrid;
bool WCA::UpdateWallMesh = false;
bool WCA::UpdateWaterMesh = false;
static Mesh* wallMesh = nullptr;
static Mesh* waterMesh = nullptr;

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
	void updateGridWater(Cell* grid, Cell* tempGrid, int n)
	{
		const glm::ivec3 side[] =
		{
			glm::ivec3(-1, 0, 0),
			glm::ivec3(1, 0, 0),
			glm::ivec3(0, 0,-1),
			glm::ivec3(0, 0, 1)
		};
		const glm::ivec3 down = { 0, -1, 0 };
		const glm::ivec3 up = { 0, 1, 0 };

		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n; i += stride)
		{
			// get coord from index
			glm::ivec3 pos = expand(i);
			Cell cell = grid[i];
			
			// move some water down if there is space
			glm::ivec3 downPos = pos + down;
			glm::ivec3 upPos = pos + up;
			if (downPos.y > 0)
			{
				int downIdx = flatten(downPos);
				Cell downCell = grid[downIdx];
				if (cell.fill_ > 0 && downCell.fill_ < MAX_WATER && !downCell.isWall_)
				{
					cell.fill_ -= WMOV_INC;
					cell.velocity_ = down;
					//atomicAdd(&tempGrid[downIdx].fill_, WMOV_INC);
					//downCell.fill_ += WMOV_INC;
					//tempGrid[downIdx] = downCell;
				}
			}
			// add some water to this if above has more
			if (upPos.y < GRID_SIZE_Y)
			{
				int upIdx = flatten(upPos);
				Cell upCell = grid[upIdx];
				if (upCell.fill_ > 0 && cell.fill_ < MAX_WATER && !upCell.isWall_)
				{
					cell.fill_ += WMOV_INC;
					cell.velocity_ = down;
				}
			}
			
			//if (nearCellIdx > -1)
			//	tempGrid[nearCellIdx] = nearCell;
			tempGrid[i] = cell;
		}

	}


	__device__
	bool inBoundary(glm::ivec3 p)
	{
		return
			p.x >= 0 && p.x < GRID_SIZE_X &&
			p.y >= 0 && p.y < GRID_SIZE_Y &&
			p.z >= 0 && p.z < GRID_SIZE_Z;
	}

	__device__
	bool inBound(int a, int b)
	{
		return a >= 0 && a < b;
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
						/*y == GRID_SIZE_Y - 1 ||*/ y == 0 || // leave top open
						z == GRID_SIZE_Z - 1 || z == 0)
						Grid[index].isWall_ = true;

					if (x == 4 && y == 4 && z == 4) Grid[index].fill_ = 1.0f;
					if (x == 6 && y == 4 && z == 4) Grid[index].fill_ = .2f;
				}
			}
		}

		// init grids to be same
		for (int i = 0; i < CELL_COUNT; i++)
			TGrid[i] = Grid[i];

		wallMesh = new Mesh(GenWallMesh(Grid));
		waterMesh = new Mesh(GenWaterMesh(Grid));
	}


	void UpdateWCA()
	{
		updateGridWater<<<numBlocks, blockSize>>>(Grid, TGrid, CELL_COUNT);
		cudaDeviceSynchronize();

		// TGrid contains updated grid values after update
		// we no longer care about values in Grid anymore
		// a simple swap of pointers will suffice
		std::swap(Grid, TGrid);

		UpdateWaterMesh = true;
		UpdateWallMesh = true;
	}


	void RenderWCA()
	{
		if (UpdateWaterMesh)
		{
			delete waterMesh;
			//waterMesh = new Mesh(GenWaterMesh(Grid));
			waterMesh = new Mesh(GenGoLMesh(Grid));
			UpdateWaterMesh = false;
		}

		if (UpdateWallMesh)
		{
			delete wallMesh;
			wallMesh = new Mesh(GenWallMesh(Grid));
			UpdateWallMesh = false;
		}

		ShaderPtr sr = Shader::shaders["flatPhong"];
		sr->Use();
		sr->setMat4("u_proj", Render::GetCamera()->GetProj());
		sr->setMat4("u_view", Render::GetCamera()->GetView());
		sr->setMat4("u_model", glm::mat4(1));
		sr->setVec3("u_color", glm::vec3(.1));
		sr->setVec3("u_viewpos", Render::GetCamera()->GetPos());

		wallMesh->Draw();

		sr->setVec3("u_color", { 0, .4, .9 });
		waterMesh->Draw();
	}


	void ShutdownWCA()
	{
		cudaFree(Grid);
		cudaFree(TGrid);
	}
}