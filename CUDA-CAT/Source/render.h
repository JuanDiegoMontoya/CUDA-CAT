#pragma once
#include <functional>
#include "mesh.h"
#include "line.h"
#include "game_object.h"
#include "GoL.h"

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
	bool pauseSimulation = true;
	float updateFrequency = .1f; // seconds
	float timeCount = 0;
	GameOfLife<50, 50, 50> GoL;
	CellularAutomata* automaton = GoL;

private:
	void drawQuad();

	// debug
	void drawAxisIndicators();
};