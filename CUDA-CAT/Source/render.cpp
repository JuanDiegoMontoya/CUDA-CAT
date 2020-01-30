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

#include "tester.h"

#define VISUALIZE_MAPS 0


Renderer::Renderer() {}


// initializes the gBuffer and its attached textures
void Renderer::Init()
{
	initPPBuffers();
	testFunc();
}


// draws everything at once, I guess?
// this should be fine
void Renderer::DrawAll()
{
  PERF_BENCHMARK_START;
	glEnable(GL_FRAMEBUFFER_SRGB); // gamma correction

	//for (auto& obj : objects)
	//	obj->Update();

	auto view = Render::GetCamera()->GetView();
	auto proj = Render::GetCamera()->GetProj();
	auto skyView = glm::mat4(glm::mat3(view));
	drawNormal(view, proj, Render::GetCamera()->GetPos());
	drawAxisIndicators();
	drawDepthMapsDebug();

	postProcess();

	glDisable(GL_FRAMEBUFFER_SRGB);
  PERF_BENCHMARK_END;
}


void Renderer::Clear() // unused
{
	glClearColor(0, 0, 0, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}


static VAO* blockHoverVao = nullptr;
static VBO* blockHoverVbo = nullptr;
void Renderer::DrawCube()
{
	if (blockHoverVao == nullptr)
	{
		blockHoverVao = new VAO();
		blockHoverVbo = new VBO(Render::cube_vertices, sizeof(Render::cube_vertices));
		VBOlayout layout;
		layout.Push<float>(3);
		blockHoverVao->AddBuffer(*blockHoverVbo, layout);
	}
	//glClear(GL_DEPTH_BUFFER_BIT);
	//glDisable(GL_CULL_FACE);
	blockHoverVao->Bind();
	glDrawArrays(GL_TRIANGLES, 0, 36);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	//glEnable(GL_CULL_FACE);
}


// draw sky objects (then clear the depth buffer)
void Renderer::drawSky()
{
	//activeSun_->Render();
}


// draw solid objects in each chunk
void Renderer::drawNormal(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& viewpos)
{
	DrawCB preDrawCB =
		[&view, &proj, &viewpos, this]()
	{
		ShaderPtr shdr = Shader::shaders["phong"];
		shdr->Use();
		shdr->setMat4("u_view", view);
		shdr->setMat4("u_proj", proj);
	};

	ModelCB drawCB =
		[&](const GameObject* obj)
	{
		ShaderPtr shdr = Shader::shaders["phong"];
		shdr->setMat4("u_model", obj->model);
	};

	DrawCB postDrawCB = [](){}; // does nothing (yet)

	//drawMeshes(preDrawCB, drawCB, postDrawCB, (std::vector<void*>&)objects);
}


void Renderer::drawMeshes(
	DrawCB predraw_cb, 
	ModelCB draw_cb, 
	DrawCB postdraw_cb,
	std::vector<void*>& cont)
{
	predraw_cb();

	for (GameObject* obj : (std::vector<GameObject*>&)cont)
	{
		draw_cb(obj);
		obj->Draw();
	}

	postdraw_cb();
}


void Renderer::drawBillboard(VAO * vao, size_t count, DrawCB uniform_cb)
{
	// TODO: set this up for actual use
}


void Renderer::drawDepthMapsDebug()
{}


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


void Renderer::initPPBuffers()
{
	int scrX = Settings::Graphics.screenX;
	int scrY = Settings::Graphics.screenY;

	glGenFramebuffers(1, &pBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, pBuffer);

	// color buffer
	glGenTextures(1, &pColor);
	glBindTexture(GL_TEXTURE_2D, pColor);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, scrX, scrY, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pColor, 0);

	// depth attachment
	glGenTextures(1, &pDepth);
	glBindTexture(GL_TEXTURE_2D, pDepth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, scrX, scrY, 0, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, pDepth, 0);

	glDrawBuffer(GL_COLOR_ATTACHMENT0);
	
	ASSERT_MSG(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "Incomplete framebuffer!");
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


void Renderer::postProcess()
{
	glBindFramebuffer(GL_FRAMEBUFFER, pBuffer);
	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	int scrX = Settings::Graphics.screenX;
	int scrY = Settings::Graphics.screenY;

	// copy default framebuffer contents into pBuffer
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, pBuffer);
	//glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glBlitFramebuffer(
		0, 0, scrX, scrY, 
		0, 0, scrX, scrY, 
		GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, pColor);
	ShaderPtr shader = Shader::shaders["postprocess"];
	shader->Use();

	shader->setInt("colorTex", 0);
	shader->setBool("sharpenFilter", ppSharpenFilter);
	shader->setBool("edgeDetection", ppEdgeDetection);
	shader->setBool("chromaticAberration", ppChromaticAberration);
	shader->setBool("blurFilter", ppBlurFilter);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_FRAMEBUFFER_SRGB);
	drawQuad();
	glEnable(GL_FRAMEBUFFER_SRGB);
	glEnable(GL_DEPTH_TEST);
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