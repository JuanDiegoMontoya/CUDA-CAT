#include "stdafx.h"
#include "PipeWater.h"

#include "shader.h"
#include "camera.h"
#include "pipeline.h"
#include "mesh.h"
#include "utilities.h"

#include "CAMesh.h"
#include "CommonDevice.cuh"

//template class PipeWater<200, 1, 200>;
//template class PipeWater<1, 1, 10>;
//template class PipeWater<10, 1, 1>;
template class PipeWater<10, 1, 10>;

// ######################################################
// ######################################################
// ######################################################
//               TODO: THIS WHOLE FILE
// ######################################################
// ######################################################
// ######################################################

template<int X, int Y, int Z>
__global__ static void updateGridWater(WaterCell* grid, WaterCell* tempGrid, Pipe* hPGrid, Pipe* vPGrid)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = X * Y * Z;

	for (int i = index; i < n; i += stride)
	{
		float depth = grid[i].depth;

		// almost certainly using pipe grid index (pipe->vec2)
		glm::ivec2 tp = expand<X>(i);

		// d += -dt*(SUM(Q)/(dx)^2)
		// add to depth flow of adjacent pipes
		//depth += hPGrid[tp.z][tp.x].flow;
		//depth -= hPGrid[tp.z][tp.x + 1].flow;
		//depth += vPGrid[tp.x][tp.z].flow;
		//depth -= vPGrid[tp.x][tp.z + 1].flow;
		float sumflow = 0;

		// LEFT TO RIGHT FLOW
		// tp.y is Z POSITION (vec2 constraint)
		// add flow from left INTO this cell
		// (vec2->pipe)
		//sumflow += hPGrid[flatten<X+1>({ tp.y, tp.x })].flow;
		// subtract flow TO right cell
		//sumflow -= hPGrid[flatten<X+1>({ tp.y + 1, tp.x })].flow;
		//sumflow += vPGrid[flatten<Z+1>({tp.y, tp.x})].flow;
		//sumflow -= vPGrid[flatten<Z+1>({tp.y, tp.x + 1})].flow;
		sumflow += hPGrid[flatten<X + 1>({ tp.y, tp.x })].flow;
		sumflow -= hPGrid[flatten<X + 1>({ tp.y + 1, tp.x })].flow;
		sumflow += vPGrid[flatten<Z + 1>({ tp.x, tp.y })].flow;
		sumflow -= vPGrid[flatten<Z + 1>({ tp.x + 1, tp.y })].flow;

		float dt = .125;
		tempGrid[i].depth	= depth + sumflow * -dt;
	}
}


template<int X, int Y, int Z>
__global__ static void updateHPipes(WaterCell* grid,  Pipe* hPGrid, Pipe* thPGrid)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = (X+1) * (Z);

	for (int i = index; i < n; i += stride)
	{
		float flow = hPGrid[i].flow;

		// PIPE GRID ACCESS (X + 1) (pipe->vec2)
		glm::ivec2 pipePos = expand<X+1>(i);
		swap(pipePos.x, pipePos.y); // confusion

		if (pipePos.x == 0 || pipePos.x == X)
		{
			thPGrid[i].flow = 0;
			continue;
		}

		/*
		0   1   2  <-- PIPE INDEX
		| 0 | 1 |  <-- CELL INDEX
		This is why we need to do pipePos - { 1, 0 } to get the left cell,
		but not for the right cell.
		*/
		// (vec2->normal!) USE NORMAL GRID INDEX
		float leftHeight = grid[flatten<X>(pipePos - glm::ivec2(1, 0))].depth;
		float rightHeight = grid[flatten<X>(pipePos)].depth;

		// A = cross section
		// A = d (w/ line above) * dx       # OPTIONAL
		// d (w/ line above) = upwind depth # OPTIONAL
		// dh = surface height difference
		// dh_(x+.5,y) = h_(x+1,y) - h_(x,y)
		// dt = optional scalar
		// Q += A*(g/dx)*dh*dt
		float A = 1;
		float g = 9.8;
		float dt = .125;
		float dx = 1; // CONSTANT (length of pipe)
		float dh = rightHeight - leftHeight; // diff left->right

		// flow from left to right
		thPGrid[i].flow = A * (g / dx) * dh * dt;
	}
}


template<int X, int Y, int Z>
__global__ static void updateVPipes(WaterCell* grid, Pipe* vPGrid, Pipe* tvPGrid)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = (X) * (Z+1);

	for (int i = index; i < n; i += stride)
	{
		float flow = vPGrid[i].flow;

		glm::ivec2 pipePos = expand<Z + 1>(i);
		//swap(pipePos.x, pipePos.y); // confusion

		if (pipePos.y == 0 || pipePos.y == Z)
		{
			tvPGrid[i].flow = 0;
			continue;
		}

		float downheight = grid[flatten<Z>(pipePos - glm::ivec2(0, 1))].depth;
		float upheight = grid[flatten<Z>(pipePos)].depth;
		float A = 1;
		float g = 9.8;
		float dt = .125;
		float dx = 1;
		float dh = upheight - downheight;

		tvPGrid[i].flow = A * (g / dx) * dh * dt;
	}
}


template<int X, int Y, int Z>
static void updateHPipesCPU(WaterCell* grid, Pipe* hPGrid, Pipe* thPGrid)
{
	int n = (X + 1) * (Z);

	for (int i = 0; i < n; i++)
	{
		float flow = hPGrid[i].flow;

		// PIPE GRID ACCESS (X + 1) (pipe->vec2)
		glm::ivec2 pipePos = expand<X+1>(i);
		std::swap(pipePos.x, pipePos.y); // confusion

		//ASSERT(pipePos.x < 10);
		if (pipePos.x == 0 || pipePos.x == X)
		{
			thPGrid[i].flow = 0;
			continue;
		}

		/*
		0   1   2  <-- PIPE INDEX
		| 0 | 1 |  <-- CELL INDEX
		This is why we need to do pipePos - { 1, 0 } to get the left cell,
		but not for the right cell.
		*/
		// (vec2->normal!) USE NORMAL GRID INDEX
		float leftHeight = grid[flatten<X>(pipePos - glm::ivec2(1, 0))].depth;
		float rightHeight = grid[flatten<X>(pipePos)].depth;

		// A = cross section
		// A = d (w/ line above) * dx       # OPTIONAL
		// d (w/ line above) = upwind depth # OPTIONAL
		// dh = surface height difference
		// dh_(x+.5,y) = h_(x+1,y) - h_(x,y)
		// dt = optional scalar
		// Q += A*(g/dx)*dh*dt
		float A = 1;
		float g = 9.8;
		float dt = .125;
		float dx = 1; // CONSTANT (length of pipe)
		float dh = rightHeight - leftHeight; // diff left->right

		// flow from left to right
		thPGrid[i].flow = A * (g / dx) * dh * dt;
	}
}


template<int X, int Y, int Z>
static void updateVPipesCPU(WaterCell* grid, Pipe* vPGrid, Pipe* tvPGrid)
{
	int n = (X) * (Z + 1);

	for (int i = 0; i < n; i++)
	{
		float flow = vPGrid[i].flow;

		glm::ivec2 pipePos = expand<Z + 1>(i);
		//std::swap(pipePos.x, pipePos.y); // confusion

		if (pipePos.y == 0 || pipePos.y == Z)
		{
			tvPGrid[i].flow = 0;
			continue;
		}

		float downheight = grid[flatten<Z>(pipePos - glm::ivec2(0, 1))].depth;
		float upheight = grid[flatten<Z>(pipePos)].depth;
		float A = 1;
		float g = 9.8;
		float dt = .125;
		float dx = 1;
		float dh = upheight - downheight;

		tvPGrid[i].flow = A * (g / dx) * dh * dt;
	}
}

// THIS FUNCTION IS CORRECT FOR 2D
template<int X, int Y, int Z>
static void updateGridWaterCPU(WaterCell* grid, WaterCell* tempGrid, Pipe* hPGrid, Pipe* vPGrid)
{
	int n = X * Y * Z;

	for (int i = 0; i < n; i ++)
	{
		float depth = grid[i].depth;

		// almost certainly NORMAL grid index
		glm::ivec2 tp = expand<X>(i);
		float sumflow = 0;

		sumflow += hPGrid[flatten<X + 1>({ tp.y, tp.x })].flow;
		sumflow -= hPGrid[flatten<X + 1>({ tp.y + 1, tp.x })].flow;
		sumflow += vPGrid[flatten<Z + 1>({ tp.x, tp.y })].flow;
		sumflow -= vPGrid[flatten<Z + 1>({ tp.x + 1, tp.y })].flow;

		float dt = .125;
		tempGrid[i].depth = depth + sumflow * -dt;
	}
}


template<int X, int Y, int Z>
PipeWater<X, Y, Z>::PipeWater()
{
	auto e1 = cudaMallocManaged(&hPGrid, (X + 1) * (Z) * sizeof(Pipe));
	auto e2 = cudaMallocManaged(&thPGrid, (X + 1) * (Z) * sizeof(Pipe));
	auto e3 = cudaMallocManaged(&vPGrid, (X) * (Z + 1) * sizeof(Pipe));
	auto e4 = cudaMallocManaged(&tvPGrid, (X) * (Z + 1) * sizeof(Pipe));
	//std::cout << e1 << '\n' << e2 << '\n' << e3 << '\n' << e4 << '\n';
	//vPGrid  = new Pipe[((X) * (Z + 1))];
	//tvPGrid = new Pipe[((X) * (Z + 1))];
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Init()
{
	// populate the grid with water and walls
	for (int z = 0; z < Z; z++)
	{
		int zpart = z * X * Y;
		for (int y = 0; y < Y; y++)
		{
			int yzpart = zpart + y * X;
			for (int x = 0; x < X; x++)
			{
				// compute final part of flattened index
				int index = x + yzpart;

				this->Grid[index].depth = Utils::get_random(0, 10);
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
	updateHPipes<X, Y, Z><<<hPNumBlocks, PBlockSize>>>(this->Grid, hPGrid, thPGrid);
	//updateHPipesCPU<X, Y, Z>(this->Grid, hPGrid, thPGrid);
	updateVPipes<X, Y, Z><<<vPNumBlocks, PBlockSize>>>(this->Grid, vPGrid, tvPGrid);
	//updateVPipesCPU<X, Y, Z>(this->Grid, vPGrid, tvPGrid);
	cudaDeviceSynchronize();
	std::swap(hPGrid, thPGrid);
	std::swap(vPGrid, tvPGrid);
	updateGridWater<X, Y, Z><<<numBlocks, blockSize>>>(this->Grid, this->TGrid, hPGrid, vPGrid);
	//updateGridWaterCPU<X, Y, Z>(this->Grid, this->TGrid, hPGrid, vPGrid);
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
	sr->setMat4("u_model", glm::translate(glm::mat4(1), glm::vec3(150, 40, 80)));
	sr->setVec3("u_color", { .2, .7, .9 });
	sr->setVec3("u_viewpos", Render::GetCamera()->GetPos());

	this->mesh_->Draw();

	{
		ImGui::Begin("Piped Water Simulation");
		float sum = 0;
		for (int i = 0; i < X * Y * Z; i++)
			sum += this->Grid[i].depth;
		ImGui::Text("Sum of water: %.2f", sum);
		//ImGui::Text("Avg height: %.2f", sum / (X * Y * Z));
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
