#pragma once
#include <functional>
#include "mesh.h"
#include "line.h"
#include "game_object.h"

class VAO;
class IBO;
class Shader;
typedef std::function<void()> DrawCB;
typedef std::function<void(const GameObject*)> ModelCB;

// responsible for making stuff appear on the screen
class Renderer
{
public:
	Renderer();
	void Init();

	// interaction
	void DrawAll();
	void Clear();

	static void DrawCube();

	bool renderShadows = true;
	bool doGeometryPass = true; // for ssr currently
	//bool renderSSR = true;

	// other stuff
	bool revolve = true;

	// pp effects
	bool ppSharpenFilter = false;
	bool ppBlurFilter = false;
	bool ppEdgeDetection = false;
	bool ppChromaticAberration = false;

	// CA
	bool pauseSimulation = false;
private:
	// broad-phase rendering
	void drawSky();
	void drawNormal(const glm::mat4& view, const glm::mat4& proj, const glm::vec3& viewpos); // draw what we see

	// narrow-phase rendering
	void drawMeshes(
		DrawCB predraw_cb, 
		ModelCB draw_cb, 
		DrawCB postdraw_cb,
		std::vector<void*>& cont);
	void drawBillboard(VAO* vao, size_t count, DrawCB uniform_cb);
	void drawQuad();

	// debug
	void drawDepthMapsDebug();
	void drawAxisIndicators();

	// post processing
	void initPPBuffers();
	void postProcess();

	// pp
	unsigned pBuffer;
	unsigned pColor;
	unsigned pDepth;
};