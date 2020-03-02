#pragma once
#include "mesh.h"
#include "Vertices.h"

#define FLATTEN(x, y, z) (x + y * X + z * X * Y)
#define FLATTENV(p) (FLATTEN(p.x, p.y, p.z))

static const glm::ivec3 faces[6] =
{
	{ 0, 0, 1 }, // 'far' face    (-z direction)
	{ 0, 0,-1 }, // 'near' face   (+z direction)
	{-1, 0, 0 }, // 'left' face   (-x direction)
	{ 1, 0, 0 }, // 'right' face  (+x direction)
	{ 0, 1, 0 }, // 'top' face    (+y direction)
	{ 0,-1, 0 }, // 'bottom' face (-y direction)
};
enum Face { Far, Near, Left, Right, Top, Bottom };

inline void AddQuad(glm::ivec3 pos, int face,
	std::vector<Vertex>& vertices, std::vector<GLuint>& indices, float heightModifier)
{
	// add adjusted indices of the quad we're about to add
	int endIndices = (face + 1) * 6;
	for (int i = face * 6; i < endIndices; i++)
		indices.push_back(Vertices::cube_indices_light_cw[i] + vertices.size());

	// add vertices for the quad
	const GLfloat* data = Vertices::cube_vertices_light;
	int endQuad = (face + 1) * 12;
	for (int i = face * 12; i < endQuad; i += 3)
	{
		Vertex v;
		v.Position = { data[i], data[i + 1], data[i + 2] };
		if (v.Position.y > 0) // TODO: make sure this works
		{
			v.Position.y += .5f;
			v.Position.y *= heightModifier;
			v.Position.y -= .5f;
		}
		v.Position += pos;
		v.Normal = faces[face];
		vertices.push_back(v);
	}
}


// bool SkipCompare(const CellType& elem) -- returns true if elem's mesh should not be generated
// float HeightCheck(const CellType& elem) -- returns fractional height of elem
template<class CellType, class SkipCheck, class HeightCheck>
Mesh* GenVoxelMesh(CellType* Grid, int X, int Y, int Z, SkipCheck skip, HeightCheck height)
{
	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;

	for (int z = 0; z < Z; z++)
	{
		int zpart = z * X * Y;
		for (int y = 0; y < Y; y++)
		{
			int yzpart = zpart + y * X;
			for (int x = 0; x < X; x++)
			{
				int index = x + yzpart;

				if (skip(Grid[index]))
					continue;

				float thisHeight = height(Grid[index]);

				// iterate faces of cells
				for (int f = 0; f < 6; f++)
				{
					const auto& normal = faces[f];

					// near cell pos
					glm::ivec3 ncp(x + normal.x, y + normal.y, z + normal.z);
					if ( // generate quad if near cell would be OOB
						ncp.x < 0 || ncp.x > X - 1 ||
						ncp.y < 0 || ncp.y > Y - 1 ||
						ncp.z < 0 || ncp.z > Z - 1)
					{
						AddQuad({ x, y, z }, f, vertices, indices, thisHeight);
						continue;
					}

					const CellType& nearCell = Grid[FLATTENV(ncp)];
					float theirHeight = height(nearCell);
					if (f == Face::Top && thisHeight >= 1 && theirHeight > 0) continue;
					if (f == Face::Bottom && theirHeight >= 1 && thisHeight > 0) continue;
					if (f != Face::Top && f != Face::Bottom && thisHeight <= theirHeight) // skip our face if lower or same as theirs
						continue;
					AddQuad({ x, y, z }, f, vertices, indices, thisHeight);
				}
			}
		}
	}

	return new Mesh(vertices, indices, std::vector<Texture>());
}