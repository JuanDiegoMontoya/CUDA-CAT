#include "stdafx.h"
#include "PipeWater.h"

#include "shader.h"
#include "camera.h"
#include "Pipeline.h"
#include "utilities.h"
#include "Renderer.h"

#include "CAMesh.h" // voxel mesh
#include "CommonDevice.cuh" // helpers, cudaCheck
#include "cuda_gl_interop.h"// gl interoperability
#include <map> // texture print


template class PipeWater<100, 1, 100>;
template class PipeWater<500, 1, 500>;
template class PipeWater<2000, 1, 2000>;

// texture storing the depth information
surface<void, 2> surfRef;

/*################################################################
##################################################################
										      KERNEL CODE
##################################################################
################################################################*/


// prints how many time each unique element of texture appears in it
void printTex(int x, int y, GLuint texID)
{
	int numElements = x * y;
	float* data = new float[numElements];

	glBindTexture(GL_TEXTURE_2D, texID);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, data);
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


// makes a splash
template<int X, int Y, int Z>
__global__ static void perturbGrid(SplashArgs args)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = X * Y * Z;

	for (int i = index; i < n; i += stride)
	{
		glm::ivec2 tp = expand<X>(i);

		float h = 0;
		surf2Dread(&h, surfRef, tp.x * sizeof(float), tp.y);
		// DHM
		// y = Ae^(-bt)
		float t = glm::distance({ tp.x, tp.y }, args.pos);
		h += args.A * glm::pow(glm::e<float>(), -args.b * t);
		surf2Dwrite(h, surfRef, tp.x * sizeof(float), tp.y);
	}
}


// throttles pipes that would cause an update to lower water depth below 0
// TODO: this function is broken and not super necessary
template<int X, int Y, int Z>
__global__ static void clampGridPipes(Pipe* hPGrid, Pipe* vPGrid, float dt)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int n = X * Y * Z;

	for (int i = index; i < n; i += stride)
	{
		glm::ivec2 tp = expand<X>(i);
		float depth;
		surf2Dread(&depth, surfRef, tp.x * sizeof(float), tp.y);

		float sumflow = 0;

		float flows[4];
		flows[0] = hPGrid[flatten<X + 1>({ tp.y, tp.x })].flow; // flow from left +
		flows[1] = hPGrid[flatten<X + 1>({ tp.y + 1, tp.x })].flow; // flow to right -
		flows[2] = vPGrid[flatten<Z + 1>({ tp.x, tp.y })].flow; // flow from below +
		flows[3] = vPGrid[flatten<Z + 1>({ tp.x + 1, tp.y })].flow; // flow to top -

		sumflow += flows[0];
		sumflow -= flows[1];
		sumflow += flows[2];
		sumflow -= flows[3];
		float finalDepth = depth + (sumflow * -dt);

		if (finalDepth < 0)
		{
			float scalar = depth / (sumflow * -dt);
			if (fabs(scalar) > 1)
			{
				//printf("meme: %.3f\n", scalar);
				//continue;
			}
			if (flows[0] < 0)
			{
				//printf("divisor: %.3f flow0: %.3f\n", divisor);
				hPGrid[flatten<X + 1>({ tp.y, tp.x })].flow = flows[0] * scalar;
			}
			if (flows[1] > 0)
			{
				hPGrid[flatten<X + 1>({ tp.y + 1, tp.x })].flow = flows[1] * scalar;
			}
			if (flows[2] < 0)
			{
				vPGrid[flatten<Z + 1>({ tp.x, tp.y })].flow = flows[2] * scalar;
			}
			if (flows[3] > 0)
			{
				vPGrid[flatten<Z + 1>({ tp.x + 1, tp.y })].flow = flows[3] * scalar;
			}
		}
	}
}


// update the height of each water cell based on its current height and the flow
// to and from neighboring pipes
template<int X, int Y, int Z>
__global__ static void updateGridWater(Pipe* hPGrid, Pipe* vPGrid, float dt)
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
		float sumflow = 0;

		// LEFT TO RIGHT FLOW
		// tp.y is Z POSITION (vec2 constraint)
		// add flow from left INTO this cell
		// (vec2->pipe)
		sumflow += hPGrid[flatten<X + 1>({ tp.y, tp.x })].flow; // flow from left
		sumflow -= hPGrid[flatten<X + 1>({ tp.y + 1, tp.x })].flow; // flow to right
		sumflow += vPGrid[flatten<Z + 1>({ tp.x, tp.y })].flow; // flow from below
		sumflow -= vPGrid[flatten<Z + 1>({ tp.x + 1, tp.y })].flow; // flow to top

		surf2Dwrite(depth + (sumflow * -dt), surfRef, tp.x * sizeof(float), tp.y);
	}
}


// traverse the pipes that deliver water horizontally and update them based
// on height of neighboring water cells to left and right
template<int X, int Y, int Z>
__global__ static void updateHPipes(Pipe* hPGrid, PipeUpdateArgs args)
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
		float leftHeight;
		float rightHeight;
		surf2Dread(&leftHeight, surfRef, pipePos.y * sizeof(float), pipePos.x - 1);
		surf2Dread(&rightHeight, surfRef, pipePos.y * sizeof(float), pipePos.x);

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


// traverse the pipes that deliver water vertically and update them based
// on height of neighboring water cells to top and bottom
template<int X, int Y, int Z>
__global__ static void updateVPipes(Pipe* vPGrid, PipeUpdateArgs args)
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

		float downheight;
		float upheight;
		//surf2Dread(&downheight, surfRef, pipePos.x * sizeof(float), pipePos.y - 1);
		//surf2Dread(&upheight, surfRef, pipePos.x * sizeof(float), pipePos.y);
		surf2Dread(&downheight, surfRef, (pipePos.y -1) * sizeof(float), pipePos.x );
		surf2Dread(&upheight, surfRef, pipePos.y * sizeof(float), pipePos.x);
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
	cudaCheck(cudaMallocManaged(&hPGrid, (X + 1) * (Z) * sizeof(Pipe)));
	cudaCheck(cudaMallocManaged(&vPGrid, (X) * (Z + 1) * sizeof(Pipe)));
	cudaCheck(cudaMallocManaged(&temphPGrid, (X + 1) * (Z) * sizeof(Pipe)));
	cudaCheck(cudaMallocManaged(&tempvPGrid, (X) * (Z + 1) * sizeof(Pipe)));
}


template<int X, int Y, int Z>
PipeWater<X, Y, Z>::~PipeWater()
{
	//cudaCheck(cudaGraphicsUnregisterResource(imageResource));
}


#define _USE_MATH_DEFINES
#include <math.h>
template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Init()
{
	initDepthTex();

	// reset flow of 
	for (int i = 0; i < (X + 1) * Z; i++)
		hPGrid[i].flow = 0;
	for (int i = 0; i < (Z + 1) * X; i++)
		vPGrid[i].flow = 0;
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Update()
{
	cudaCheck(cudaGraphicsMapResources(1, &imageResource, 0));
	cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, imageResource, 0, 0));
	cudaCheck(cudaBindSurfaceToArray(surfRef, arr));

	// update pipes' flow
	updateHPipes<X, Y, Z><<<hPNumBlocks, PBlockSize>>>(hPGrid, args);
	updateVPipes<X, Y, Z><<<vPNumBlocks, PBlockSize>>>(vPGrid, args);
	cudaDeviceSynchronize();

	//clampGridPipes<X, Y, Z><<<numBlocks, blockSize>>>(hPGrid, vPGrid, args.dt);
	//cudaDeviceSynchronize();

	// update water depth
	updateGridWater<X, Y, Z><<<numBlocks, blockSize>>>(hPGrid, vPGrid, args.dt);
	cudaDeviceSynchronize();
	cudaCheck(cudaGraphicsUnmapResources(1, &imageResource, 0));
}


template<int X, int Y, int Z>
void PipeWater<X, Y, Z>::Render()
{
	//ShaderPtr sr = Shader::shaders["flatPhong"];
	ShaderPtr sr = Shader::shaders["heightWater"];
	sr->Use();
	glm::mat4 model(1);
	model = glm::translate(model, glm::vec3(150, 40, 80));
	model = glm::scale(model, glm::vec3(.01, .01, .01));
	sr->setMat4("u_proj", Renderer::GetPipeline()->GetCamera(0)->GetProj());
	sr->setMat4("u_view", Renderer::GetPipeline()->GetCamera(0)->GetView());
	sr->setMat4("u_model", model);
	//sr->setVec3("u_color", { .2, .7, .9 });
	sr->setVec3("u_viewpos", Renderer::GetPipeline()->GetCamera(0)->GetPos());
	sr->setVec3("sun.ambient", { .1, .1, .1 });
	sr->setVec3("sun.diffuse", { .8, .8, .8 });
	sr->setVec3("sun.specular", { .8, .8, .8 });
	sr->setVec3("sun.direction", { 0, -1, 0 });
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, HeightTex);
	sr->setInt("heightTex", 0);
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
		if (ImGui::Button("Splash water"))
		{
			cudaCheck(cudaGraphicsMapResources(1, &imageResource, 0));
			cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, imageResource, 0, 0));
			cudaCheck(cudaBindSurfaceToArray(surfRef, arr));
			perturbGrid<X, Y, Z><<<numBlocks, blockSize>>>(splash);
			//for (int i = 0; i < 10; i++)
			//{
			//	perturbGrid<X, Y, Z> << <numBlocks, blockSize >> > (splashLoc + glm::ivec2(i * 25, i * 25));
			//}
			cudaCheck(cudaGraphicsUnmapResources(1, &imageResource, 0));
		}
		if (ImGui::Button("Random Splash"))
		{
			SplashArgs sp = splash;
			sp.pos = { Utils::get_random(0, X), Utils::get_random(0, Z) };
			cudaCheck(cudaGraphicsMapResources(1, &imageResource, 0));
			cudaCheck(cudaGraphicsSubResourceGetMappedArray(&arr, imageResource, 0, 0));
			cudaCheck(cudaBindSurfaceToArray(surfRef, arr));
			perturbGrid<X, Y, Z><<<numBlocks, blockSize>>>(sp);
			cudaCheck(cudaGraphicsUnmapResources(1, &imageResource, 0));
		}
		ImGui::Separator();
		ImGui::Text("Splash Settings");
		ImGui::InputFloat2("Location", &splash.pos[0]);
		ImGui::InputFloat("Amplitude", &splash.A);
		ImGui::InputFloat("Falloff", &splash.b);
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
void PipeWater<X, Y, Z>::initDepthTex()
{
	vertices2d.clear();
	indices.clear();

	//vertices2d = std::vector<glm::vec2>(X * Z * 2, glm::vec2(0)); // num cells * attributes (pos + normal)
	vertices2d.reserve(X * Z * 2);
	indices.reserve((X - 1) * (Z - 1) * 2 * 3); // num cells * num tris per cell * num verts per tri
	
	for (int x = 0; x < X; x++)
	{
		for (int z = 0; z < Z; z++)
		{
			glm::dvec2 p(x, z);
			glm::dvec2 P(X, Z);
			vertices2d.push_back(p); // pos xz
			//vertices2d.push_back({ float(x) / float(X), float(z) / float(Z) }); // texcoord
			vertices2d.push_back(p / P); // texcoord
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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

	GLfloat height = 0;
	glClearTexImage(HeightTex, 0, GL_RED, GL_FLOAT, &height);

	auto err = cudaGraphicsGLRegisterImage(&imageResource, HeightTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if (err != cudaSuccess)
		std::cout << "Error registering CUDA image: " << err << std::endl;
}