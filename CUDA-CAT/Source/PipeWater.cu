#include "stdafx.h"
#include "PipeWater.h"

#include "shader.h"
#include "camera.h"
#include "pipeline.h"
#include "mesh.h"
#include "utilities.h"

#include "CAMesh.h"
#include "CommonDevice.cuh"
#include "cuda_gl_interop.h"
#include "vendor/helper_cuda.h"
#include <map>

//template class PipeWater<200, 1, 200>;
//template class PipeWater<1, 1, 10>;
//template class PipeWater<10, 1, 1>;
//template class PipeWater<10, 1, 10>;
template class PipeWater<100, 1, 100>;
template class PipeWater<500, 1, 500>;
template class PipeWater<2000, 1, 2000>;

surface<void, 2> surfRef;

/*################################################################
##################################################################
										      KERNEL CODE
##################################################################
################################################################*/

void printTex(int x, int y, GLuint texID)
{
	int numElements = x * y;
	float* data = new float[numElements];

	glBindTexture(GL_TEXTURE_2D, texID);
	{
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	std::map<float, int> unique;
	for (int i = 0; i < numElements; i++)
	{
		unique[data[i]]++;
	}
	// print how many times f.first appears
	for (auto f : unique)
		std::cout << f.first << " :: " << f.second << '\n';

	delete[] data;
}


// init surface
template<int X, int Y, int Z>
__global__ static void perturbGrid()
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = X * Y * Z;

	for (int i = index; i < n; i += stride)
	{
		glm::ivec2 tp = expand<X>(i);

		float h = glm::distance(glm::vec2(tp.x, tp.y), glm::vec2(0, 0)) / 5.0f +
			cosf(tp.x / 50.f) * 5 + sinf(tp.y / 50.f) * 5 + 10;
		surf2Dwrite(h, surfRef, tp.x * sizeof(float), tp.y);
	}
}

template<int X, int Y, int Z>
__global__ static void updateGridWater(WaterCell* grid, Pipe* hPGrid, Pipe* vPGrid, float dt)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = X * Y * Z;

	for (int i = index; i < n; i += stride)
	{
		glm::ivec2 tp = expand<X>(i);
		float depth;// = grid[i].depth;
		surf2Dread(&depth, surfRef, tp.x * sizeof(float), tp.y);


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

		//float dt = .125;
		//tempGrid[i].depth	= depth + (sumflow * -dt);
		grid[i].depth	= depth + (sumflow * -dt);
		surf2Dwrite(depth + (sumflow * -dt), surfRef, tp.x * sizeof(float), tp.y);
		//surf2Dwrite(float(blockIdx.x) / 100, surfRef, tp.x * sizeof(float), tp.y);
	}
}


template<int X, int Y, int Z>
__global__ static void updateHPipes(WaterCell* grid,  Pipe* hPGrid, PipeUpdateArgs args)
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
			hPGrid[i].flow = 0;
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
		//thPGrid[i].flow = flow + (A * (g / dx) * dh * dt);
		hPGrid[i].flow = flow + (A * (args.g / args.dx) * dh) * args.dt;
	}
}


template<int X, int Y, int Z>
__global__ static void updateVPipes(WaterCell* grid, Pipe* vPGrid, PipeUpdateArgs args)
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
			vPGrid[i].flow = 0;
			continue;
		}

		float downheight = grid[flatten<Z>(pipePos - glm::ivec2(0, 1))].depth;
		float upheight = grid[flatten<Z>(pipePos)].depth;
		float A = 1;
		float g = 9.8;
		float dt = .125;
		float dx = 1;
		float dh = upheight - downheight;

		//tvPGrid[i].flow = flow + (A * (g / dx) * dh * dt);
		vPGrid[i].flow = flow + (A * (args.g / args.dx) * dh) * args.dt;
	}
}


/*################################################################
##################################################################
										    END KERNEL CODE
##################################################################
################################################################*/


template<int X, int Y, int Z>
PipeWater<X, Y, Z>::PipeWater()
{
	auto e1 = cudaMallocManaged(&hPGrid, (X + 1) * (Z) * sizeof(Pipe));
	auto e3 = cudaMallocManaged(&vPGrid, (X) * (Z + 1) * sizeof(Pipe));

	// will at least tell us something is wrong
	auto ef = e1 | e3;
	if (ef != cudaSuccess)
		std::cout << "Error initializing shared memory!\n";

	//initPlanarMesh();
	initDepthTex();
}


template<int X, int Y, int Z>
PipeWater<X, Y, Z>::~PipeWater()
{
	cudaGraphicsUnregisterResource(imageResource);
}


#define _USE_MATH_DEFINES
#include <math.h>
template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Init()
{
	return;
	// populate the grid with water
	for (int z = 0; z < Z; z++)
	{
		int zpart = z * X * Y;
		for (int y = 0; y < Y; y++)
		{
			int yzpart = zpart + y * X;
			for (int x = 0; x < X; x++)
			{
				int index = x + yzpart;

				//this->Grid[index].depth = Utils::get_random(7, 10);
				this->Grid[index].depth = glm::distance(glm::vec2(x, z), glm::vec2(0, 0)) / 5.0f +
					cosf(x / 50.f) * 5 + sinf(z / 50.f) * 5 + 10;
				//this->Grid[index].depth = sinf(z * M_PI / 5) + cosf(x * M_PI / 5) + 2;
				//this->Grid[index].depth = cosf(z * M_PI / 5) + cosf(x * M_PI / 5) + 2;
			}
		}
	}

	for (int i = 0; i < (X + 1) * Z; i++)
		hPGrid[i].flow = 0;
	for (int i = 0; i < (Z + 1) * X; i++)
		vPGrid[i].flow = 0;

	this->UpdateMesh = true;
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Update()
{
	updateHPipes<X, Y, Z><<<hPNumBlocks, PBlockSize>>>(this->Grid, hPGrid, args);
	//updateHPipesCPU<X, Y, Z>(this->Grid, hPGrid, thPGrid);
	updateVPipes<X, Y, Z><<<vPNumBlocks, PBlockSize>>>(this->Grid, vPGrid, args);
	//updateVPipesCPU<X, Y, Z>(this->Grid, vPGrid, tvPGrid);
	cudaDeviceSynchronize();


	if (cudaGraphicsMapResources(1, &imageResource, 0))
		printf("1ERROR\n");
	if (cudaGraphicsSubResourceGetMappedArray(&arr, imageResource, 0, 0))
		printf("2ERROR\n");
	int err = cudaBindSurfaceToArray(surfRef, arr);
	if(err)
		printf("3ERROR\n");
	

	//printTex(X, Z, HeightTex);

	updateGridWater<X, Y, Z><<<numBlocks, blockSize>>>(this->Grid, hPGrid, vPGrid, args.dt);
	//updateGridWaterCPU<X, Y, Z>(this->Grid, this->TGrid, hPGrid, vPGrid);
	cudaDeviceSynchronize();
	err = cudaGraphicsUnmapResources(1, &imageResource, 0);
	if (err)
		printf("4ERROR\n");

	//printTex(X, Z, HeightTex);

	this->UpdateMesh = true;
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Render()
{
	//if (this->UpdateMesh)
	//	updatePlanarMesh(), this->UpdateMesh = false;
	//if (this->UpdateMesh)
	//	genMesh(), this->UpdateMesh = false;

	//ShaderPtr sr = Shader::shaders["flatPhong"];
	ShaderPtr sr = Shader::shaders["height"];
	sr->Use();
	glm::mat4 model(1);
	model = glm::translate(model, glm::vec3(150, 40, 80));
	model = glm::scale(model, glm::vec3(.1, .1, .1));
	sr->setMat4("u_proj", Render::GetCamera()->GetProj());
	sr->setMat4("u_view", Render::GetCamera()->GetView());
	sr->setMat4("u_model", model);
	sr->setVec3("u_color", { .2, .7, .9 });
	sr->setVec3("u_viewpos", Render::GetCamera()->GetPos());
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, HeightTex);
	sr->setInt("heightTex", 0);

	//this->mesh_->Draw();
	glDisable(GL_CULL_FACE);
	pVao->Bind();
	glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, nullptr);
	pVao->Unbind();
	glEnable(GL_CULL_FACE);

	{
		ImGui::Begin("Piped Water Simulation");
		ImGui::Text("Dimensions: X = %d, Z = %d", X, Z);
		ImGui::Separator();
		ImGui::Text("Changing settings may lead \n to explosive results");
		ImGui::SliderFloat("dt", &args.dt, 0, 1, "%.2f s");
		ImGui::SliderFloat("dx", &args.dx, 0, 5, "%.2f m");
		ImGui::SliderFloat("g", &args.g, 0, 50, "%.2f m/s^2");
		ImGui::Checkbox("Calculate Normals", &calcNormals);
		//float sum = 0;
		//for (int i = 0; i < X * Y * Z; i++)
		//	sum += this->Grid[i].depth;
		//ImGui::Text("Sum of water: %.2f", sum);
		//ImGui::Text("Avg height: %.2f", sum / (X * Y * Z));
		ImGui::End();
	}

	//printTex(X, Z, HeightTex);
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


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::initPlanarMesh()
{
	vertices = std::vector<glm::vec3>(X * Z * 2, glm::vec3(0)); // num cells * attributes (pos + normal)
	indices.reserve((X-1) * (Z-1) * 2 * 3); // num cells * num tris per cell * num verts per tri

	// init indices
	for (int x = 0; x < X - 1; x++)
	{
		// for each cell
		for (int z = 0; z < Z - 1; z++)
		{
			GLuint one = flatten<X>(glm::ivec2(x, z));
			GLuint two = flatten<X>(glm::ivec2(x + 1, z));
			GLuint three = flatten<X>(glm::ivec2(x + 1, z + 1));
			GLuint four = flatten<X>(glm::ivec2(x, z + 1));

			indices.push_back(one);
			indices.push_back(two);
			indices.push_back(three);
			
			indices.push_back(one);
			indices.push_back(three);
			indices.push_back(four);
		}
	}

	pVbo = new VBO(&vertices[0][0], vertices.size() * sizeof(glm::vec3), GL_DYNAMIC_DRAW);
	VBOlayout layout;
	layout.Push<float>(3);
	layout.Push<float>(3);
	pVao = new VAO();
	pVao->AddBuffer(*pVbo, layout);
	pIbo = new IBO(indices.data(), indices.size());
	pVao->Unbind();
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::updatePlanarMesh()
{
	vertices.clear();
	//for (int i = 0; i < X * Z; i++)
	//{
	//	auto xz = expand<X>(i);
	//	auto xzUp = xz; xzUp.y++;
	//	auto xzRight = xz; xzRight.x++;
	//	glm::vec3 up = { xzUp.x, this->Grid[flatten<X>(xzUp)].depth, xzUp.y };
	//	glm::vec3 right = { xzRight.x, this->Grid[flatten<X>(xzRight)].depth, xzRight.y };
	//	glm::vec3 d = { xz.x, this->Grid[i].depth, xz.y };
	//	vertices.push_back(d);
	//	//vertices.push_back(glm::normalize(glm::cross(xz.y < Z ? up - d : d - up, xz.x < X ? right - d : d - right)));
	//	// TODO: fix normals
	//	vertices.push_back(glm::cross(up - d, right - d));
	//	//vertices.push_back({ 0, 1, 0 });
	//}

	for (int x = 0; x < X; x++)
	{
		for (int z = 0; z < Z; z++)
		{
			int ind = flatten<X>({ x, z });
			int indup = flatten<X>({ x, z + 1 });
			int indright = flatten<X>({ x + 1, z });
			if (indup > X * Z) indup = ind;
			if (indright > X * Z) indright = ind;
			glm::vec3 cur = { x, Grid[ind].depth, z };
			glm::vec3 right = { x + 1, Grid[indright].depth, z };
			glm::vec3 up = { x, Grid[indup].depth, z + 1};

			vertices.push_back(cur);
			if (calcNormals)
				vertices.push_back(glm::cross(up - cur, right - cur));
			else
				vertices.push_back({ 0, 1, 0 });
		}
	}

	pVbo->Bind();
	glBufferSubData(GL_ARRAY_BUFFER,
		NULL,
		vertices.size() * sizeof(glm::vec3),
		vertices.data());
	pVbo->Unbind();
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::initDepthTex()
{
	//vertices2d = std::vector<glm::vec2>(X * Z * 2, glm::vec2(0)); // num cells * attributes (pos + normal)
	vertices2d.reserve(X * Z * 2);
	indices.reserve((X - 1) * (Z - 1) * 2 * 3); // num cells * num tris per cell * num verts per tri
	
	for (int x = 0; x < X; x++)
	{
		for (int z = 0; z < Z; z++)
		{
			vertices2d.push_back({ (float)x, float(z) }); // pos xz
			vertices2d.push_back({ float(x) / float(X), float(z) / float(Z) }); // texcoord
		}
	}

	// init indices
	for (int x = 0; x < X - 1; x++)
	{
		// for each cell
		for (int z = 0; z < Z - 1; z++)
		{
			GLuint one = flatten<X>(glm::ivec2(x, z));
			GLuint two = flatten<X>(glm::ivec2(x + 1, z));
			GLuint three = flatten<X>(glm::ivec2(x + 1, z + 1));
			GLuint four = flatten<X>(glm::ivec2(x, z + 1));

			indices.push_back(one);
			indices.push_back(two);
			indices.push_back(three);

			indices.push_back(one);
			indices.push_back(three);
			indices.push_back(four);
		}
	}

	pVbo = new VBO(&vertices2d[0], vertices2d.size() * sizeof(glm::vec2), GL_DYNAMIC_DRAW);
	VBOlayout layout;
	layout.Push<float>(2); // pos xz
	layout.Push<float>(2); // texCoord
	pVao = new VAO();
	pVao->AddBuffer(*pVbo, layout);
	pIbo = new IBO(indices.data(), indices.size());
	pVao->Unbind();

	// Generate 2D texture with 1 float element
	glGenTextures(1, &HeightTex);
	glBindTexture(GL_TEXTURE_2D, HeightTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, X, Z, 0, GL_RED, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glBindTexture(GL_TEXTURE_2D, 0);

	GLfloat height = 6.9;
	glClearTexImage(HeightTex, 0, GL_RED, GL_FLOAT, &height);

	auto err = cudaGraphicsGLRegisterImage(&imageResource, HeightTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if (err != cudaSuccess)
		std::cout << "Error registering CUDA image: " << err << std::endl;

	// not sure if these 2 lines are necessary
	//cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	//err = cudaMallocArray(&arr, &channelDesc, X, Z, cudaArraySurfaceLoadStore);
	//if (err != cudaSuccess)
	//	std::cout << "Error mallocing cuda array: " << err << std::endl;
	

	if (cudaGraphicsMapResources(1, &imageResource, 0))
		printf("1ERROR\n");
	if (cudaGraphicsSubResourceGetMappedArray(&arr, imageResource, 0, 0))
		printf("2ERROR\n");
	err = cudaBindSurfaceToArray(surfRef, arr);
	if (err)
		printf("3ERROR\n");
	perturbGrid<X, Y, Z><<<numBlocks, blockSize>>>();
	err = cudaGraphicsUnmapResources(1, &imageResource, 0);
	if (err)
		printf("4ERROR\n");
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::updateDepthTex()
{
}


/*################################################################
##################################################################
               CPU SIMULATION CODE FOR DEBUGGING
##################################################################
################################################################*/


template<int X, int Y, int Z>
static void updateHPipesCPU(WaterCell* grid, Pipe* hPGrid, Pipe* thPGrid)
{
	int n = (X + 1) * (Z);

	for (int i = 0; i < n; i++)
	{
		float flow = hPGrid[i].flow;

		glm::ivec2 pipePos = expand<X + 1>(i);
		std::swap(pipePos.x, pipePos.y); // confusion

		//ASSERT(pipePos.x < 10);
		if (pipePos.x == 0 || pipePos.x == X)
		{
			thPGrid[i].flow = 0;
			continue;
		}

		float leftHeight = grid[flatten<X>(pipePos - glm::ivec2(1, 0))].depth;
		float rightHeight = grid[flatten<X>(pipePos)].depth;

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


template<int X, int Y, int Z>
static void updateGridWaterCPU(WaterCell* grid, WaterCell* tempGrid, Pipe* hPGrid, Pipe* vPGrid)
{
	int n = X * Y * Z;

	for (int i = 0; i < n; i++)
	{
		float depth = grid[i].depth;

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


/*################################################################
##################################################################
					           END CPU SIMULATION CODE
##################################################################
################################################################*/