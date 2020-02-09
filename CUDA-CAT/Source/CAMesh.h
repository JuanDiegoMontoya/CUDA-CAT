#pragma once
#include "mesh.h"

namespace WCA
{
	Mesh GenWallMesh(Cell* grid);
	Mesh GenWaterMesh(Cell* grid);
	Mesh GenGoLMesh(Cell* grid);
}