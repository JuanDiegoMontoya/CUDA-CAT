#pragma once
#include "CellularAutomata.h"

struct CaveCell : public Cell
{
	bool Rock = false;
};

template<int X, int Y, int Z>
class CaveGen : public CellularAutomata<CaveCell, X, Y, Z>
{
public:
	virtual void Init() override;
	virtual void Update() override;
	virtual void Render() override;

	float r = 0.5f; // initial percentage of rock cells
	int n = 4;      // num iterations
	int T = 13;      // moore neighborhood rock threshold
	int M = 2;      // moore neighborhood size
private:
	virtual void genMesh() override;

};