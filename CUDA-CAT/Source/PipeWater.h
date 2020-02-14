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

// TODO: TEST 2D VERSION FIRST
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

	const int PBlockSize = 128;
	const int hPNumBlocks = ((X+1) * Z + PBlockSize - 1) / PBlockSize;
	const int vPNumBlocks = (X * (Z+1) + PBlockSize - 1) / PBlockSize;
	Pipe* hPGrid = nullptr; // horizontal (x axis)
	Pipe* thPGrid = nullptr; // temp
	Pipe* vPGrid = nullptr; // vertical (z axis)
	Pipe* tvPGrid = nullptr; // temp

	Mesh* pMesh_ = nullptr;
};