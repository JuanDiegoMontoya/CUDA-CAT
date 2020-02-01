#include "stdafx.h"
#include "waterCA.h"
#include "CAMesh.h"
#include "pipeline.h"
#include "utilities.h"

#define FLATTEN(x, y, z) (x + y * GRID_SIZE_X + z * GRID_SIZE_X * GRID_SIZE_Z)
#define FLATTENV(p) (FLATTEN(p.x, p.y, p.z))

// in the future, might make the mesh on the GPU (which should be fairly simple)

namespace WCA
{
	static const glm::ivec3 faces[6] =
	{
		{ 0, 0, 1 }, // 'far' face    (-z direction)
		{ 0, 0,-1 }, // 'near' face   (+z direction)
		{-1, 0, 0 }, // 'left' face   (-x direction)
		{ 1, 0, 0 }, // 'right' face  (+x direction)
		{ 0, 1, 0 }, // 'top' face    (+y direction)
		{ 0,-1, 0 }, // 'bottom' face (-y direction)
	};


	void AddQuad(glm::ivec3 pos, int face,
		std::vector<Vertex>& vertices, std::vector<GLuint>& indices)
	{
		// add adjusted indices of the quad we're about to add
		int endIndices = (face + 1) * 6;
		for (int i = face * 6; i < endIndices; i++)
			indices.push_back(Render::cube_indices_light_cw[i] + vertices.size());

		// add vertices for the quad
		const GLfloat* data = Render::cube_vertices_light;
		int endQuad = (face + 1) * 12;
		for (int i = face * 12; i < endQuad; i += 3)
		{
			Vertex v;
			v.Position = { data[i], data[i + 1], data[i + 2] };
			v.Position += pos; // + .5f;
			v.Normal = faces[face];
			vertices.push_back(v);
		}
	}


	// a much "dumber" mesh because it is constantly changing
	Mesh GenWaterMesh(Cell* grid)
	{
		std::vector<Vertex> vertices;
		std::vector<GLuint> indices;

		for (int z = 0; z < GRID_SIZE_Z; z++)
		{
			int zpart = z * GRID_SIZE_X * GRID_SIZE_Y;
			for (int y = 0; y < GRID_SIZE_Y; y++)
			{
				int yzpart = zpart + y * GRID_SIZE_X;
				for (int x = 0; x < GRID_SIZE_X; x++)
				{
					int index = x + yzpart;
					glm::vec3 pos(x, y, z);
					Cell cell = grid[index];

					// skip wall cells...
					if (cell.isWall_ == true)
						continue;
					if (cell.fill_ <= 0)
						continue;

					for (int f = 0; f < 6; f++)
					{
						// indices
						int endIndices = (f + 1) * 6;
						for (int i = f * 6; i < endIndices; i++)
							indices.push_back(Render::cube_indices_light_cw[i] + vertices.size());

						// vertices
						const GLfloat* data = Render::cube_vertices_light;
						int endQuad = (f + 1) * 12;
						for (int i = f * 12; i < endQuad; i += 3)
						{
							Vertex v;
							v.Position = { data[i], data[i + 1], data[i + 2] };
							if (v.Position.y > 0 && cell.fill_ < 1.0f)
							{
								v.Position.y += .5f;
								v.Position.y *= cell.fill_;
								v.Position.y -= .5f;
							}
							v.Position += pos; // + .5f;
							v.Normal = faces[f];
							vertices.push_back(v);
						}
					}
				}
			}
		}

		return Mesh(vertices, indices, std::vector<Texture>());
	}


	Mesh GenWallMesh(Cell* grid)
	{
		std::vector<Vertex> vertices;
		std::vector<GLuint> indices;

		for (int z = 0; z < GRID_SIZE_Z; z++)
		{
			int zpart = z * GRID_SIZE_X * GRID_SIZE_Y;
			for (int y = 0; y < GRID_SIZE_Y; y++)
			{
				int yzpart = zpart + y * GRID_SIZE_X;
				for (int x = 0; x < GRID_SIZE_X; x++)
				{
					int index = x + yzpart;
					
					// skip non wall cells...
					if (grid[index].isWall_ == false)
						continue;

					// iterate faces of cells
					for (int f = 0; f < 6; f++)
					{
						const auto& normal = faces[f];

						// near cell pos
						glm::ivec3 ncp(x + normal.x, y + normal.y, z + normal.z);
						if ( // generate quad if near cell would be OOB
							ncp.x < 0 || ncp.x > GRID_SIZE_X - 1 ||
							ncp.y < 0 || ncp.y > GRID_SIZE_Y - 1 ||
							ncp.z < 0 || ncp.z > GRID_SIZE_Z - 1)
						{
							AddQuad({ x, y, z }, f, vertices, indices);
							continue;
						}
						const Cell& nearCell = grid[FLATTENV(ncp)];
						if (nearCell.isWall_) // skip opposing faces
							continue;
						AddQuad({ x, y, z }, f, vertices, indices);
					}
				}
			}
		}

		return Mesh(vertices, indices, std::vector<Texture>());
	}
}