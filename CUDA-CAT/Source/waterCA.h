#pragma once
//#include <glm/vec3.hpp> // just for visual error

// unfortunately we need these defines to easily have this data
// visible at the global scope- both on the host and the device
#define GRID_SIZE_X 15
#define GRID_SIZE_Y 15
#define GRID_SIZE_Z 15
#define CELL_COUNT (GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z)
#define MAX_WATER 1.0f

#define WMOV_INC 0.0078125f // how much water to move incrementally

namespace WCA
{
	struct Cell
	{
		Cell() : isWall_(false), fill_(0), velocity_(0) {}
		bool isWall_;
		float fill_; // 0-1 normal fill range
		glm::ivec3 velocity_; // axis aligned
	};

	// normals of cube, or cell in this instance
	const glm::ivec3 dirs[6] =
	{
		{-1, 0, 0 },
		{ 1, 0, 0 },
		{ 0,-1, 0 },
		{ 0, 1, 0 },
		{ 0, 0,-1 },
		{ 0, 0, 1 }
	};


	//const glm::uvec3 GRID_SIZE = { 10, 10, 10 };
	//const unsigned CELL_COUNT = GRID_SIZE.x * GRID_SIZE.y * GRID_SIZE.z;

	const int blockSize = 256;
	const int numBlocks = (CELL_COUNT + blockSize - 1) / blockSize;

	extern Cell* Grid;
	extern Cell* TGrid;
	extern bool UpdateWallMesh;

	void UpdateWCA();
	void RenderWCA();
	void InitWCA();
}