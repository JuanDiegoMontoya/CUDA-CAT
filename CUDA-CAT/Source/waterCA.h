#pragma once
#include <glm/vec3.hpp> // just for visual error

namespace WCA
{
	struct Cell
	{
		Cell() : wallHere_(false), fill_(0), velocity_(0) {}
		bool wallHere_;
		float fill_; // 0-1 normal fill range
		glm::ivec3 velocity_; // axis aligned
	};

	constexpr inline glm::uvec3 GRID_SIZE(10,10,10);

	inline Cell Grid[GRID_SIZE.x * GRID_SIZE.y * GRID_SIZE.z];
}