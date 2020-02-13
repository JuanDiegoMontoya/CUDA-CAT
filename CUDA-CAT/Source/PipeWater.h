#pragma once
#include "CellularAutomata.h"

//https://tutcris.tut.fi/portal/files/4312220/kellomaki_1354.pdf p74
struct WaterCell : public Cell
{
	// A = cross section
	// A = d (w/ line above) * dx       # OPTIONAL
	// d (w/ line above) = upwind depth # OPTIONAL
	// dh = surface height difference
	// dh_(x+.5,y) = h_(x+1,y) - h_(x,y)
	// dt = optional scalar
	float flow = 0; // Q += A*(g/dx)*dh*dt
	float depth = 0; // d += -dt*(SUM(Q)/(dx)^2) # sum of all adjacent pipes
};

// TODO: TEST 2D VERSION FIRST
template<int X, int Y, int Z>
class PipeWater : public CellularAutomata<WaterCell, X, Y, Z>
{
public:
	virtual void Init() override;
	virtual void Update() override;
	virtual void Render() override;

private:
	virtual void genMesh() override;
};