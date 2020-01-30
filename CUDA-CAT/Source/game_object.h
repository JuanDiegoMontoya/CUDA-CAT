#pragma once
#include <functional>


// a renderable object
class GameObject
{
public:
	GameObject(std::function<void(GameObject*)> func = [](GameObject* n){});
	~GameObject();

	virtual void Update();
	void Draw();
	void DrawNormals(bool drawFLines = false, bool drawVLines = false);
	void SetMesh(Mesh* mesh);
	Mesh* GetMesh() { return mesh_; }
	const Mesh* GetMesh() const { return mesh_; }

	// renderer
	glm::vec3 diffuse;

	// transform stuff
	glm::vec3 rpos = { 0, 0, 0 }; // final position (mutable)
	glm::vec3 bpos = { 0, 0, 0 }; // base position
	//glm::vec3 rot = { 0, 0, 0 }; // euler angles
	glm::vec3 scl = { 1, 1, 1 };
	float angle = 0;
	glm::mat4 model;
	bool visible = true;

	// normal visualizations
	std::vector<Line*> vtxNmlLines;
	std::vector<Line*> fceNmlLines;
	LinePool* vtxLines;
	LinePool* fceLines;
private:
	friend class Light;
	std::function<void(GameObject*)> updateFunc_;
	Mesh* mesh_;

	virtual void calcTransform()
	{
		model = glm::mat4(1);
		model = glm::translate(model, rpos);
		model = glm::scale(model, scl);
		//transform_ *= rotation;
	}
};

