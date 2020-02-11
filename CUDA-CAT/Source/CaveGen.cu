#include "stdafx.h"
#include "CaveGen.h"

#include "shader.h"
#include "camera.h"
#include "pipeline.h"
#include "mesh.h"
#include "utilities.h"

#include "CommonDevice.cuh"
#define FLATTEN(x, y, z) (x + y * X + z * X * Y)
#define FLATTENV(p) (FLATTEN(p.x, p.y, p.z))

template class CaveGen<50, 50, 1>;
template class CaveGen<200, 200, 1>;

template<int X, int Y, int Z>
__global__ static void updateGridCave(CaveCell* grid, CaveCell* tempGrid, int T, int M)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = X * Y * Z;

	for (int i = index; i < n; i += stride)
	{
		bool thisRock = grid[i].Rock;
		bool nextState;
		glm::ivec3 thisPos = expand<X, Y>(i);
		int env = thisRock ? -1 : 0; // un-count self if alive

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
					if (grid[fIndex].Rock == true)
						env++;
				}
			}
		}

		if (env >= T)
			nextState = true;
		else
			nextState = false;

		tempGrid[i].Rock = nextState;
	}
}


template<int X, int Y, int Z>
void CaveGen<X, Y, Z>::Init()
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

				
				this->Grid[index].Rock = Utils::get_random(0, 1) < r;
				//this->Grid[index].Rock = rand() % 50 == 0;
			}
		}
	}

	// init grids to be same
	for (int i = 0; i < X * Z * Y; i++)
		this->TGrid[i] = this->Grid[i];

	genMesh();

	//for (int i = 0; i < n; i++)
	//	Update();
}


template<int X, int Y, int Z>
void CaveGen<X, Y, Z>::Update()
{
	updateGridCave<X, Y, Z><<<numBlocks, blockSize>>>(this->Grid, this->TGrid, T, M);
	cudaDeviceSynchronize();

	// TGrid contains updated grid values after update
	// we no longer care about values in Grid anymore
	// a simple swap of pointers will suffice
	std::swap(this->Grid, this->TGrid);
	this->UpdateMesh = true;
}


template<int X, int Y, int Z>
void CaveGen<X, Y, Z>::Render()
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
		// DANGEROUS IF AUTOMATON IS NOT A CAVEGEN
		ImGui::Begin("Cave Generator");
		auto caver = this;
		ImGui::PushItemWidth(150);
		ImGui::SliderFloat("r (starting rock %)", &caver->r, 0, 1, "%.2f");
		ImGui::SliderInt("n (num iterations)", &caver->n, 0, 6);
		ImGui::InputInt("T (threshold)", &caver->T);
		ImGui::SameLine();
		if (ImGui::Button("Suggest"))
		{
			switch (caver->M)
			{
			case 1: caver->T = 5; break;
			case 2: caver->T = 13; break;
			case 3: caver->T = 25; break;
			case 4: caver->T = 41; break;
			default: caver->T = 0; break;
			}
		}
		ImGui::SliderInt("M (neighborhood)", &caver->M, 1, 4);
		ImGui::Separator();
		if (ImGui::Button("Generate"))
		{
			caver->Init();
			for (int i = 0; i < caver->n; i++)
				caver->Update();
		}
		ImGui::End();
	}
}


template<int X, int Y, int Z>
void CaveGen<X, Y, Z>::genMesh()
{
	delete this->mesh_;
	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;

	for (int z = 0; z < Z; z++)
	{
		int zpart = z * X * Y;
		for (int y = 0; y < Y; y++)
		{
			int yzpart = zpart + y * X;
			for (int x = 0; x < X; x++)
			{
				int index = x + yzpart;

				// skip empty cells
				if (this->Grid[index].Rock == false)
					continue;

				// iterate faces of cells
				for (int f = 0; f < 6; f++)
				{
					const auto& normal = faces[f];

					// near cell pos
					glm::ivec3 ncp(x + normal.x, y + normal.y, z + normal.z);
					if ( // generate quad if near cell would be OOB
						ncp.x < 0 || ncp.x > X - 1 ||
						ncp.y < 0 || ncp.y > Y - 1 ||
						ncp.z < 0 || ncp.z > Z - 1)
					{
						AddQuad({ x, y, z }, f, vertices, indices);
						continue;
					}
					const auto& nearCell = this->Grid[FLATTENV(ncp)];
					if (nearCell.Rock == true) // skip opposing faces
						continue;
					AddQuad({ x, y, z }, f, vertices, indices);
				}
			}
		}
	}

	this->mesh_ = new Mesh(vertices, indices, std::vector<Texture>());
}
