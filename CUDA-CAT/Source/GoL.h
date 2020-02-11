#pragma once
#include "CellularAutomata.h"

struct GoLCell : public Cell
{
	bool Alive = false;

#ifdef N__CUDACC__
	template<int X, int Y, int Z>
	virtual void Update(glm::ivec3 pos, Cell* grid) override;
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
	//GameOfLife() : CellularAutomata(X, Y, Z)
	virtual void Init() override;
	virtual void Update() override;
	virtual void Render() override;

private:
	virtual void genMesh() override;

	const GoLRule r1 = { 10, 21, 10, 21 };
	const GoLRule r2 = { 4, 5, 2, 6 };
	const GoLRule r3 = { 5, 7, 6, 6 };
	const GoLRule r4 = { 4, 5, 5, 5 };
	const GoLRule r5 = { 3, 3, 3, 3 };
	const GoLRule rog = { 2, 3, 3, 3 };
	GoLRule currRule = rog;
};