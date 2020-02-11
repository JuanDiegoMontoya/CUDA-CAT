#pragma once
#include "texture.h"
#include "vao.h"
#include "ibo.h"
#include "vbo.h"
#include "pipeline.h"

struct Vertex
{
	Vertex(glm::vec3 p = glm::vec3(0), glm::vec3 n = glm::vec3(0), 
		glm::vec2 t = glm::vec2(0)) 
		: Position(p), Normal(n), TexCoords(t)/*, index(0)*/ {}
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoords;
	//int index;
};

class Mesh
{
public:

	explicit Mesh(std::vector<Vertex> vertices, 
		std::vector<unsigned> indices, 
		std::vector<Texture> textures);
	Mesh(const Mesh& other);
	~Mesh();

	void Draw();

	const std::vector<Vertex>& GetVertices() const { return vertices; }
	const std::vector<GLuint>& GetIndices() const { return indices; }
	const std::vector<Texture>& GetTextures() const { return textures; }

	// unadjusted min, max, and center bounding coords
	glm::vec3 min, max, mid;
protected:
	Mesh() = delete;

	VAO* vao = nullptr;
	VBO* vbo = nullptr;
	IBO* ibo = nullptr;

  unsigned vertCount;
	bool indexed; // whether to use DrawElements or DrawArrays
	void setupBuffers();

	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;
	std::vector<Texture> textures;
};


// TODO: make a mesher class or something and put this (and AddQuad) in it
// OR put this in the mesh class itself
static const glm::ivec3 faces[6] =
{
	{ 0, 0, 1 }, // 'far' face    (-z direction)
	{ 0, 0,-1 }, // 'near' face   (+z direction)
	{-1, 0, 0 }, // 'left' face   (-x direction)
	{ 1, 0, 0 }, // 'right' face  (+x direction)
	{ 0, 1, 0 }, // 'top' face    (+y direction)
	{ 0,-1, 0 }, // 'bottom' face (-y direction)
};
inline void AddQuad(glm::ivec3 pos, int face,
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

//
//template<int X, int Y, int Z>
//void GameOfLife<X, Y, Z>::genVoxelMesh()
//{
//	delete this->mesh_;
//	std::vector<Vertex> vertices;
//	std::vector<GLuint> indices;
//
//	for (int z = 0; z < Z; z++)
//	{
//		int zpart = z * X * Y;
//		for (int y = 0; y < Y; y++)
//		{
//			int yzpart = zpart + y * X;
//			for (int x = 0; x < X; x++)
//			{
//				int index = x + yzpart;
//
//				// skip empty cells
//				if (this->Grid[index].Alive == 0)
//					continue;
//
//				// iterate faces of cells
//				for (int f = 0; f < 6; f++)
//				{
//					const auto& normal = faces[f];
//
//					// near cell pos
//					glm::ivec3 ncp(x + normal.x, y + normal.y, z + normal.z);
//					if ( // generate quad if near cell would be OOB
//						ncp.x < 0 || ncp.x > X - 1 ||
//						ncp.y < 0 || ncp.y > Y - 1 ||
//						ncp.z < 0 || ncp.z > Z - 1)
//					{
//						AddQuad({ x, y, z }, f, vertices, indices);
//						continue;
//					}
//					const GoLCell& nearCell = this->Grid[FLATTENV(ncp)];
//					if (nearCell.Alive != 0) // skip opposing faces
//						continue;
//					AddQuad({ x, y, z }, f, vertices, indices);
//				}
//			}
//		}
//	}
//
//}