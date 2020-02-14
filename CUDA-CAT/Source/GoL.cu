#include "stdafx.h"
#include "GoL.h"

#include "shader.h"
#include "camera.h"
#include "pipeline.h"
#include "mesh.h"

#include "CAMesh.h"
#include "CommonDevice.cuh"

template class GameOfLife<50, 50, 50>;
template class GameOfLife<200, 200, 1>;

template<int X, int Y, int Z>
__global__ static void updateGridGOL(GoLCell* grid, GoLCell* tempGrid, GoLRule r)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = X * Y * Z;

	for (int i = index; i < n; i += stride)
	{
		bool thisAlive = grid[i].Alive;
		bool nextState;
		glm::ivec3 thisPos = expand<X, Y>(i);
		int env = thisAlive ? -1 : 0; // un-count self if alive
		
		// iterate each of 26 neighbors, checking their living status if they exist
		for (int z = -1; z <= 1; z++)
		{
			//if (!inBound(z, GRID_SIZE_Z)) continue;
			//int zPart = z * GRID_SIZE_Y * GRID_SIZE_Z;
			for (int y = -1; y <= 1; y++)
			{
				//if (!inBound(y, GRID_SIZE_Y)) continue;
				//int yzPart = y * GRID_SIZE_Y + zPart;
				for (int x = -1; x <= 1; x++)
				{
					//if (!inBound(x, GRID_SIZE_X)) continue;
					//int fIndex = x + yzPart;
					glm::ivec3 nPos = thisPos + glm::ivec3(x, y, z);
					if (!inBoundary<X, Y, Z>(nPos)) continue;
					int fIndex = flatten<X, Y>(nPos);
					if (grid[fIndex].Alive == true)
						env++;
				}
			}
		}

		if (thisAlive)
		{
			if (env >= r.eL && env <= r.eH)
				nextState = true;
			else
				nextState = false;
		}
		else // this cell is currently dead
		{
			if (env >= r.fL && env <= r.fH)
				nextState = true;
			else
				nextState = false;
		}

		tempGrid[i].Alive = nextState;
	}
}


template<int X, int Y, int Z>
void GameOfLife<X, Y, Z>::Init()
{
	// populate the grid with water and walls
	for (int z = 0; z < Z; z++)
	{
		int zPart = z * Y * X;
		for (int y = 0; y < Y; y++)
		{
			int yzPart = y * X + zPart;
			for (int x = 0; x < X; x++)
			{
				// compute final part of flattened index
				int index = x + yzPart;

				this->Grid[index].Alive = rand() % 2 == 0;
			}
		}
	}

	// init grids to be same
	for (int i = 0; i < X * Z * Y; i++)
		this->TGrid[i] = this->Grid[i];

	genMesh();
}


template<int X, int Y, int Z>
void GameOfLife<X, Y, Z>::Update()
{
	updateGridGOL<X, Y, Z> <<<numBlocks, blockSize>>>(this->Grid, this->TGrid, currRule);
	cudaDeviceSynchronize();

	// TGrid contains updated grid values after update
	// we no longer care about values in Grid anymore
	// a simple swap of pointers will suffice
	std::swap(this->Grid, this->TGrid);
	this->UpdateMesh = true;
}


template<int X, int Y, int Z>
void GameOfLife<X, Y, Z>::Render()
{
	if (this->UpdateMesh)
		genMesh(), this->UpdateMesh = false;

	ShaderPtr sr = Shader::shaders["flatPhong"];
	sr->Use();
	sr->setMat4("u_proj", Render::GetCamera()->GetProj());
	sr->setMat4("u_view", Render::GetCamera()->GetView());
	sr->setMat4("u_model", glm::mat4(1));
	sr->setVec3("u_color", { .9, .4, .4 });
	sr->setVec3("u_viewpos", Render::GetCamera()->GetPos());

	this->mesh_->Draw();

	{
		ImGui::Begin("Game of Life");
		ImGui::PushItemWidth(150);
		ImGui::SliderInt("Low Env", &currRule.eL, 0, 26);
		ImGui::SliderInt("High Env", &currRule.eH, 0, 26);
		ImGui::SliderInt("Low Fert", &currRule.fL, 0, 26);
		ImGui::SliderInt("High Fert", &currRule.fH, 0, 26);
		ImGui::Separator();
		ImGui::Text("Presets");
		// https://wpmedia.wolfram.com/uploads/sites/13/2018/02/01-3-1.pdf
		if (ImGui::Button("10, 21, 10, 21"))
			currRule = r1;
		if (ImGui::Button("4, 5, 2, 6"))
			currRule = r2;
		if (ImGui::Button("5, 7, 6, 6"))
			currRule = r3;
		if (ImGui::Button("4, 5, 5, 5"))
			currRule = r4;
		if (ImGui::Button("3, 3, 3, 3"))
			currRule = r5;
		if (ImGui::Button("2, 3, 3, 3"))
			currRule = rog;
		ImGui::End();
	}
}

template<int X, int Y, int Z>
void GameOfLife<X, Y, Z>::genMesh()
{
	delete this->mesh_;
	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;

	auto skip = [](const GoLCell& elem)->bool
	{
		return !elem.Alive;
	};
	auto height = [](const GoLCell& elem)->float
	{
		return static_cast<float>(!!elem.Alive); // 1 or 0 height
	};
	mesh_ = GenVoxelMesh(this->Grid, X, Y, Z, skip, height);
}