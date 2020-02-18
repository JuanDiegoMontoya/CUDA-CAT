#pragma once
#include "CellularAutomata.h"

//https://tutcris.tut.fi/portal/files/4312220/kellomaki_1354.pdf p74
struct WaterCell : public Cell
{
	float depth = 0; // d += -dt*(SUM(Q)/(dx)^2) # sum of all adjacent pipes
};

struct Pipe
{
	float flow = 0; // Q += A*(g/dx)*dh*dt
};

struct PipeUpdateArgs
{
	float g = 9.8;
	float dx = 1; // length of pipe
	float dt = .125;
};

template<int X, int Y, int Z>
class PipeWater : public CellularAutomata<WaterCell, X, Y, Z>
{
public:
	PipeWater();
	virtual void Init() override;
	virtual void Update() override;
	virtual void Render() override;

private:
	virtual void genMesh() override;

	// alternative to generating a mesh from scratch every update,
	// just update what is already there with much less CPU overhead
	void initPlanarMesh();
	void updatePlanarMesh();
	std::vector<glm::vec3> vertices; // order doesn't change
	std::vector<GLuint> indices; // immutable basically
	class IBO* pIbo = nullptr;
	class VBO* pVbo = nullptr;
	class VAO* pVao = nullptr;

	// use 
	void initDepthTex();
	void updateDepthTex();
	GLuint depthTex;

	const int PBlockSize = 128;
	const int hPNumBlocks = ((X+1) * Z + PBlockSize - 1) / PBlockSize;
	const int vPNumBlocks = (X * (Z+1) + PBlockSize - 1) / PBlockSize;
	Pipe* hPGrid = nullptr; // horizontal (x axis)
	Pipe* vPGrid = nullptr; // vertical (z axis)

	PipeUpdateArgs args;
	bool calcNormals = true;
};