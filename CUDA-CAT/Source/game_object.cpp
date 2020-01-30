#include "stdafx.h"
#include "mesh.h"
#include "line.h"
#include "game_object.h"
#include "shader.h"
#include "camera.h"
#include "render.h"
#include "pipeline.h"


GameObject::GameObject(std::function<void(GameObject*)> func)
	: updateFunc_(func)
{}

GameObject::~GameObject()
{
	if (mesh_)
		delete mesh_;
}

void GameObject::Update()
{
	calcTransform();
	updateFunc_(this);
}

void GameObject::Draw()
{
	mesh_->Draw();
}

void GameObject::DrawNormals(bool drawFLines, bool drawVLines)
{
	// draw dem lines
	ShaderPtr currShader = Shader::shaders["line"];
	currShader->Use();
	currShader->setMat4("u_model", model);
	currShader->setMat4("u_view", Render::GetCamera()->GetView());
	currShader->setMat4("u_proj", Render::GetCamera()->GetProj());
	//if (drawVLines)
	//	for (const auto& line : vtxNmlLines)
	//		line->Draw();
	//if (drawFLines)
	//	for (const auto& line : fceNmlLines)
	//		line->Draw();

	if (drawVLines)
		if (vtxLines)
			vtxLines->Draw();
	if (drawFLines)
		if (fceLines)
			fceLines->Draw();
}

void GameObject::SetMesh(Mesh* mesh)
{
	mesh_ = mesh;
	std::vector<Vertex> vertices;
	std::vector<unsigned> indices;
	std::vector<Texture> textures;

	std::vector<glm::vec3> wpos;
	std::vector<glm::vec3> dir;
	std::vector<glm::vec3> tclr;
	std::vector<glm::vec3> bclr;
	for (const auto& v : mesh->GetVertices())
	{
		//vtxNmlLines.push_back(new Line(v.Position, v.Normal / 7.f, { 0, 1, 0 }, { 1, 0, 0 }));

		wpos.push_back(v.Position);
		dir.push_back(v.Normal / 7.f);
		tclr.push_back({ 0, 1, 0 });
		bclr.push_back({ 1, 0, 0 });
	}
	vtxLines = new LinePool(wpos, dir, tclr, bclr);
	wpos.clear(); dir.clear(); tclr.clear(); bclr.clear();

	for (size_t i = 0; i < mesh->GetIndices().size(); i += 3)
	{
		const Vertex& v1 = mesh->GetVertices()[mesh->GetIndices()[i + 0]];
		const Vertex& v2 = mesh->GetVertices()[mesh->GetIndices()[i + 1]];
		const Vertex& v3 = mesh->GetVertices()[mesh->GetIndices()[i + 2]];
		glm::vec3 nml = (v1.Normal + v2.Normal + v3.Normal) / 3.f;
		glm::vec3 pos = (v1.Position + v2.Position + v3.Position) / 3.f;

		//fceNmlLines.push_back(new Line(pos, nml / 7.f, { 0, 0, 1 }, { 1, 0, 0 }));

		wpos.push_back(pos);
		dir.push_back(nml / 7.f);
		tclr.push_back({ 0, 0, 1 });
		bclr.push_back({ 1, 0, 0 });
	}
	fceLines = new LinePool(wpos, dir, tclr, bclr);
}
