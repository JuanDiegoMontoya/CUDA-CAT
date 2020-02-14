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
	float updateFrequency = .1f; // seconds
	float timeCount = 0;
	//GameOfLife<50, 50, 50> GoL;
	//GameOfLife<200, 200, 1> GoL2;
	//CaveGen<200, 200, 1> Caver;
	//CaveGen<50, 50, 50> Caver2;
	//CaveGen<200, 50, 200> Caver3;
	CaveGen<100, 100, 100> Caver4;
	//PipeWater<200, 1, 200> Water;
	PipeWater<1, 1, 10> Water2;
	PipeWater<10, 1, 1> Water3;
	PipeWater<10, 1, 10> Water4;
	CAInterface* automaton = reinterpret_cast<CAInterface*>(&Water4);

private:
	void drawQuad();

	// debug
	void drawAxisIndicators();
};