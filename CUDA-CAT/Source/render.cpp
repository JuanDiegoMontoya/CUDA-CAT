#include "stdafx.h"
#include "vbo.h"
#include "vao.h"
#include "ibo.h"
#include "shader.h"
#include "camera.h"
#include "pipeline.h"
#include "settings.h"
#include "input.h"
#include "render.h"

#include "GoL.h"

extern float dt__;

Renderer::Renderer() {}


// initializes the gBuffer and its attached textures
void Renderer::Init()
{
	initCubeMap();
	automaton->Init();
	//for (int i = 0; i < 500; i++)
	//	automaton->Update();
	//std::exit(0);
}


// draws everything at once, I guess?
// this should be fine
void Renderer::DrawAll()
{
  PERF_BENCHMARK_START;
	glEnable(GL_FRAMEBUFFER_SRGB); // gamma correction

	glDepthMask(GL_FALSE);
	ShaderPtr sr = Shader::shaders["skybox"];
	sr->Use();
	sr->setInt("skybox", 1);
	sr->setMat4("proj", Render::GetCamera()->GetProj());
	glm::mat4 view = Render::GetCamera()->GetView();
	view = glm::mat4(glm::mat3(view));
	sr->setMat4("view", view);
	cubeVao->Bind();
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_CUBE_MAP, cubeTex);
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glDepthMask(GL_TRUE);

	//for (auto& obj : objects)
	//	obj->Update();
	if (!pauseSimulation)
	{
		timeCount += dt__;

		if (timeCount > updateFrequency)
		{
			//WCA::UpdateWCA();
			automaton->Update();
			timeCount = 0;
		}
	}
	//WCA::RenderWCA();
	automaton->Render();

	drawAxisIndicators();


	glDisable(GL_FRAMEBUFFER_SRGB);
  PERF_BENCHMARK_END;
}


void Renderer::Clear() // unused
{
	glClearColor(0, 0, 0, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}


void Renderer::drawAxisIndicators()
{
	static VAO* axisVAO;
	static VBO* axisVBO;
	if (axisVAO == nullptr)
	{
		float indicatorVertices[] =
		{
			// positions			// colors
			0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // x-axis
			1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, // y-axis
			0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, // z-axis
			0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f
		};

		axisVAO = new VAO();
		axisVBO = new VBO(indicatorVertices, sizeof(indicatorVertices), GL_STATIC_DRAW);
		VBOlayout layout;
		layout.Push<float>(3);
		layout.Push<float>(3);
		axisVAO->AddBuffer(*axisVBO, layout);
	}
	/* Renders the axis indicator (a screen-space object) as though it were
		one that exists in the world for simplicity. */
	ShaderPtr currShader = Shader::shaders["axis"];
	currShader->Use();
	Camera* cam = Render::GetCamera();
	currShader->setMat4("u_model", glm::translate(glm::mat4(1), cam->GetPos() + cam->front * 10.f)); // add scaling factor (larger # = smaller visual)
	currShader->setMat4("u_view", cam->GetView());
	currShader->setMat4("u_proj", cam->GetProj());
	glClear(GL_DEPTH_BUFFER_BIT); // allows indicator to always be rendered
	axisVAO->Bind();
	glLineWidth(2.f);
	glDrawArrays(GL_LINES, 0, 6);
	axisVAO->Unbind();
}


// draws a single quad over the entire viewport
void Renderer::drawQuad()
{
	static unsigned int quadVAO = 0;
	static unsigned int quadVBO;
	if (quadVAO == 0)
	{
		float quadVertices[] =
		{
			// positions        // texture Coords
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
		};
		// setup plane VAO
		glGenVertexArrays(1, &quadVAO);
		glGenBuffers(1, &quadVBO);
		glBindVertexArray(quadVAO);
		glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
	}
	glBindVertexArray(quadVAO);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}


#include <stb_image.h>
void Renderer::initCubeMap()
{
	glGenTextures(1, &cubeTex);
	glBindTexture(GL_TEXTURE_CUBE_MAP, cubeTex);

	cubeVao = new VAO();
	cubeVbo = new VBO(Render::skyboxVertices, sizeof(Render::skyboxVertices));
	VBOlayout layout;
	layout.Push<float>(3);
	cubeVao->AddBuffer(*cubeVbo, layout);

	std::vector<std::string> faces =
	{
		"./resources/Textures/skybox/right.jpg",
		"./resources/Textures/skybox/left.jpg",
		"./resources/Textures/skybox/top.jpg",
		"./resources/Textures/skybox/bottom.jpg",
		"./resources/Textures/skybox/front.jpg",
		"./resources/Textures/skybox/back.jpg"
	};
	int width, height, nrChannels;
	for (unsigned int i = 0; i < faces.size(); i++)
	{
		unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
		if (data)
		{
			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
				0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data
			);
			stbi_image_free(data);
		}
		else
		{
			std::cout << "Cubemap tex failed to load at path: " << faces[i] << std::endl;
			stbi_image_free(data);
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}