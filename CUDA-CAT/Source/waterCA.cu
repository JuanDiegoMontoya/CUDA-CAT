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
	__global__
		void updateGridGOL(Cell* grid, Cell* tempGrid, int n, GOLRule r)
	{
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (int i = index; i < n; i += stride)
		{
			Cell nextState = grid[i];
			glm::ivec3 thisPos = expand(i);
			bool thisAlive = grid[i].fill_ != 0;
			int env = 0;

			// iterate each of 26 neighbors, checking their living status if they exist
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int z = -1; z <= 1; z++)
					{
						glm::ivec3 nPos = thisPos + glm::ivec3(x, y, z);
						if (inBoundary(nPos) && grid[flatten(nPos)].fill_ != 0)
							env++;
					}
				}
			}

			// A living cell remains living if it has between 2 and 3 living neighbors,
			// A dead cell will become alive if it has between 3 and 3 living neighbors.
			if (thisAlive)
			{
				if (env >= r.eL && env <= r.eH)
					nextState.fill_ = 1;
				else
					nextState.fill_ = 0;
			}
			else // this cell dead
			{
				if (env >= r.fL && env <= r.fH)
					nextState.fill_ = 1;
				else
					nextState.fill_ = 0;
			}

			tempGrid[i] = nextState;
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

	void InitGOLCA()
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

					if (
						x >= 3 && x < 6 &&
						y >= 3 && y < 6 &&
						z >= 3 && z < 6)
						Grid[index].fill_ = 1;
					else
						Grid[index].fill_ = 0;
					Grid[index].fill_ = rand() % 3 == 0;
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
		//updateGridWater<<<numBlocks, blockSize>>>(Grid, TGrid, CELL_COUNT);
		GOLRule r1 = { 10, 21, 10, 21 };
		GOLRule r2 = { 4, 5, 2, 6 };
		GOLRule r3 = { 5, 7, 6, 6 };
		GOLRule r4 = { 4, 5, 5, 5 };
		GOLRule r5 = { 3, 3, 3, 3 };
		updateGridGOL<<<numBlocks, blockSize>>>(Grid, TGrid, CELL_COUNT, r5);
		cudaDeviceSynchronize();

		// TGrid contains updated grid values after update
		// we no longer care about values in Grid anymore
		// a simple swap of pointers will suffice
		std::swap(Grid, TGrid);

		if (waterMesh)
		{
			delete waterMesh;
			waterMesh = new Mesh(GenWaterMesh(Grid));
		}

		if (UpdateWallMesh)
		{
			delete wallMesh;
			wallMesh = new Mesh(GenWallMesh(Grid));
		}
	}


	void RenderWCA()
	{
		ShaderPtr sr = Shader::shaders["flat"];
		sr->Use();
		sr->setMat4("u_proj", Render::GetCamera()->GetProj());
		sr->setMat4("u_view", Render::GetCamera()->GetView());
		sr->setMat4("u_model", glm::mat4(1));
		sr->setVec3("u_color", glm::vec3(.1));

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