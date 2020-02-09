#include "stdafx.h"
#include "GoL.h"

#include "shader.h"
#include "camera.h"
#include "pipeline.h"
#include "mesh.h"

#include "CommonDevice.cuh"
#define FLATTEN(x, y, z) (x + y * X + z * X * Y)
#define FLATTENV(p) (FLATTEN(p.x, p.y, p.z))

template class GameOfLife<50, 50, 50>;

void GoLCell::Update(glm::ivec3 pos, Cell* grid, glm::ivec3 bound)
{
	auto GGrid = reinterpret_cast<GoLCell*>(grid);
}

template<int X, int Y, int Z>
__global__ static void updateGridGOL(GoLCell* grid, GoLCell* tempGrid, int n, GoLRule r)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += stride)
	{
		bool thisAlive = grid[i].Alive;
		bool nextState;
		glm::ivec3 thisPos = expand<X, Y>(i);
		int env = -1; // un-count self
		
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

		// A living cell remains living if it has between 2 and 3 living neighbors,
		// A dead cell will become alive if it has between 3 and 3 living neighbors.
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

		//Cell swag(grid[i]); swag.fill_ = nextState;
		//tempGrid[i] = swag;
		tempGrid[i].Alive = nextState;
	}
}


template<int X, int Y, int Z>
void GameOfLife<X, Y, Z>::Init()
{
	cudaMallocManaged(&this->Grid, X * Y * Z * sizeof(GoLCell));
	cudaMallocManaged(&this->TGrid, X * Y * Z * sizeof(GoLCell));

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
	GoLRule r1 = { 10, 21, 10, 21 };
	GoLRule r2 = { 4, 5, 2, 6 };
	GoLRule r3 = { 5, 7, 6, 6 };
	GoLRule r4 = { 4, 5, 5, 5 };
	GoLRule r5 = { 3, 3, 3, 3 };
	updateGridGOL<X, Y, Z> <<<numBlocks, blockSize>>>(this->Grid, this->TGrid, X * Y * Z, r5);
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
	sr->setVec3("u_color", { 0, .4, .9 });
	sr->setVec3("u_viewpos", Render::GetCamera()->GetPos());

	this->mesh_->Draw();
}

// TODO: make a mesher class or something and put this (and AddQuad) in it
static const glm::ivec3 faces[6] =
{
	{ 0, 0, 1 }, // 'far' face    (-z direction)
	{ 0, 0,-1 }, // 'near' face   (+z direction)
	{-1, 0, 0 }, // 'left' face   (-x direction)
	{ 1, 0, 0 }, // 'right' face  (+x direction)
	{ 0, 1, 0 }, // 'top' face    (+y direction)
	{ 0,-1, 0 }, // 'bottom' face (-y direction)
};
void AddQuad(glm::ivec3 pos, int face,
	std::vector<Vertex>& vertices, std::vector<GLuint>& indices)
{
	// add adjusted indices of the quad we're about to add
	int endIndices = (face + 1) * 6;
	for (int i = face * 6; i < endIndices; i++)
		indices.push_back(Render::cube_indices_light_cw[i] + vertices.size());

	// add vertices for the quad
	const GLfloat* data = Render::cube_vertices_light;
	int endQuad = (face + 1) * 12;
	for (int i = face * 12; i < endQuad; i += 3)
	{
		Vertex v;
		v.Position = { data[i], data[i + 1], data[i + 2] };
		v.Position += pos; // + .5f;
		v.Normal = faces[face];
		vertices.push_back(v);
	}
}

template<int X, int Y, int Z>
void GameOfLife<X, Y, Z>::genMesh()
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
				if (this->Grid[index].Alive == 0)
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
					const GoLCell& nearCell = this->Grid[FLATTENV(ncp)];
					if (nearCell.Alive != 0) // skip opposing faces
						continue;
					AddQuad({ x, y, z }, f, vertices, indices);
				}
			}
		}
	}

	this->mesh_ = new Mesh(vertices, indices, std::vector<Texture>());
}