#include "stdafx.h"
#include "PipeWater.h"

#include "shader.h"
#include "camera.h"
#include "pipeline.h"
#include "mesh.h"
#include "utilities.h"

#include "CAMesh.h"
#include "CommonDevice.cuh"

template class PipeWater<200, 1, 200>;

// ######################################################
// ######################################################
// ######################################################
//               TODO: THIS WHOLE FILE
// ######################################################
// ######################################################
// ######################################################

template<int X, int Y, int Z>
__global__ static void updateGridWater(WaterCell* grid, WaterCell* tempGrid, int T, int M)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = X * Y * Z;

	for (int i = index; i < n; i += stride)
	{
		//bool thisRock = grid[i].Rock;
		bool nextState;
		glm::ivec3 thisPos = expand<X, Y>(i);
		//int env = thisRock ? -1 : 0; // un-count self if alive

		// iterate each of 26 neighbors, checking their living status if they exist
		for (int z = -M; z <= M; z++)
		{
			//if (!inBound(z, GRID_SIZE_Z)) continue;
			//int zPart = z * GRID_SIZE_Y * GRID_SIZE_Z;
			for (int y = -M; y <= M; y++)
			{
				//if (!inBound(y, GRID_SIZE_Y)) continue;
				//int yzPart = y * GRID_SIZE_Y + zPart;
				for (int x = -M; x <= M; x++)
				{
					//if (!inBound(x, GRID_SIZE_X)) continue;
					//int fIndex = x + yzPart;
					glm::ivec3 nPos = thisPos + glm::ivec3(x, y, z);
					if (!inBoundary<X, Y, Z>(nPos)) continue;
					int fIndex = flatten<X, Y>(nPos);
					
				}
			}
		}

		//if (env >= T)
		//	nextState = true;
		//else
		//	nextState = false;

		//tempGrid[i].Rock = nextState;
	}
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Init()
{
	// populate the grid with water and walls
	for (int z = 0; z < Z; z++)
	{
		int zPart = z * Y * Z;
		for (int y = 0; y < Y; y++)
		{
			int yzPart = y * Y + zPart;
			for (int x = 0; x < X; x++)
			{
				// compute final part of flattened index
				int index = x + yzPart;

				this->Grid[index].depth = Utils::get_random(0, 1);
			}
		}
	}

	// init grids to be same
	for (int i = 0; i < X * Z * Y; i++)
		this->TGrid[i] = this->Grid[i];

	genMesh();
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Update()
{
	updateGridWater<X, Y, Z><<<numBlocks, blockSize>>>(this->Grid, this->TGrid, 1, 1);
	cudaDeviceSynchronize();

	// TGrid contains updated grid values after update
	// we no longer care about values in Grid anymore
	// a simple swap of pointers will suffice
	std::swap(this->Grid, this->TGrid);
	this->UpdateMesh = true;
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Render()
{
	if (this->UpdateMesh)
		genMesh(), this->UpdateMesh = false;

	ShaderPtr sr = Shader::shaders["flatPhong"];
	sr->Use();
	sr->setMat4("u_proj", Render::GetCamera()->GetProj());
	sr->setMat4("u_view", Render::GetCamera()->GetView());
	sr->setMat4("u_model", glm::mat4(1));
	sr->setVec3("u_color", { .2, .7, .9 });
	sr->setVec3("u_viewpos", Render::GetCamera()->GetPos());

	this->mesh_->Draw();

	{
		// DANGEROUS IF AUTOMATON IS NOT A CAVEGEN
		ImGui::Begin("Piped Water Simulation");
		ImGui::End();
	}
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::genMesh()
{
	delete this->mesh_;
	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;

	auto skip = [](const WaterCell& elem)->bool
	{
		return elem.depth == 0;
	};
	auto height = [](const WaterCell& elem)->float
	{
		return elem.depth;
	};
	mesh_ = GenVoxelMesh(this->Grid, X, Y, Z, skip, height);
}
