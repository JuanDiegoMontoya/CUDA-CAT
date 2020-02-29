#pragma once
#include <functional>
#include "mesh.h"
#include "GoL.h"
#include "CaveGen.h"
#include "PipeWater.h"

class VAO;
class IBO;
class Shader;
typedef std::function<void()> DrawCB;

// responsible for making stuff appear on the screen
class Renderer
{
public:
	Renderer();
	void Init();

	// interaction
	void DrawAll();
	void Clear();

	// pp effects
	bool ppSharpenFilter = false;
	bool ppBlurFilter = false;
	bool ppEdgeDetection = false;
	bool ppChromaticAberration = false;

	// CA
	bool pauseSimulation = true;
	float updateFrequency = .02f; // seconds
	float timeCount = 0;
	GameOfLife<50, 50, 50> GoL;
	GameOfLife<200, 200, 1> GoL2;
	CaveGen<200, 200, 1> Caver;
	CaveGen<100, 100, 100> Caver4;
	PipeWater<100, 1, 100> Water5;
	PipeWater<500, 1, 500> Water6;
	PipeWater<2000, 1, 2000> Water7;
	CAInterface* automaton = reinterpret_cast<CAInterface*>(&Water7);

private:
	void drawQuad();

	// debug
	void drawAxisIndicators();

	GLuint cubeTex;
	VAO* cubeVao = nullptr;
	VBO* cubeVbo = nullptr;
	void initCubeMap();
};