#pragma once
#include "CellularAutomata.h"

struct GoLCell : public Cell
{
	bool Alive = false;

#ifdef __CUDACC__
	virtual void Update(glm::ivec3 pos, Cell* grid, glm::ivec3 bound) override;
#endif
};

struct GoLRule
{
	int eL, eH; // environment low, high (living dies otherwise)
	int fL, fH; // fertility low, high (otherwise nothing)
	// normal GOL (2D!) rules are 2,3,3,3 
};

template<int X, int Y, int Z>
class GameOfLife : public CellularAutomata<GoLCell, X, Y, Z>
{
public:
	virtual void Init() override;
	virtual void Update() override;
	virtual void Render() override;

private:
	virtual void genMesh() override;
};

