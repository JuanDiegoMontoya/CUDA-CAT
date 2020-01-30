#pragma once
#include "camera.h"
#include "game.h"
#include "render.h"

typedef class Level
{
public:
	Level(std::string name); // will load from file in the future (maybe)
	~Level();

	void Init();
	void Update(float dt);

	void DrawImGui();

	inline void SetBgColor(glm::vec3 c) { bgColor_ = c; }

	inline GamePtr Game() { return game_; }
	inline const glm::vec3& GetBgColor() const { return bgColor_; }

	friend class Game;
private:
	Renderer renderer_;

	std::string name_; // name of file
	GamePtr game_;
	std::vector<Camera*> cameras_;			 // all cameras in the scene
	//glm::vec3 bgColor_ = glm::vec3(.529f, .808f, .922f); // sky blue
	glm::vec3 bgColor_ = glm::vec3(0, .2, .4); // midnight blue
	
	bool activeCursor = false;
}Level, *LevelPtr;