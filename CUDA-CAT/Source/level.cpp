#include "stdafx.h"
#include "shader.h"
#include "camera.h"
#include "level.h"
#include "pipeline.h"
#include "mesh.h"
#include "texture.h"
#include <chrono>
#include <execution>
#include "input.h"
#include "settings.h"
#include <functional>
#include <filesystem>
#include "waterCA.h"

using namespace std::chrono;
using namespace std::filesystem;
extern bool revolve; // CANCER GLOBAL

Level::Level(std::string name)
{
	name_ = name;
}

Level::~Level()
{
	for (auto& cam : cameras_)
		delete cam;
}

// for now this function is where we declare objects
void Level::Init()
{
	cameras_.push_back(new Camera(kControlCam));
	Render::SetCamera(cameras_[0]);

	high_resolution_clock::time_point benchmark_clock_ = high_resolution_clock::now();
	
	renderer_.Init();
	
	duration<double> benchmark_duration_ = duration_cast<duration<double>>(high_resolution_clock::now() - benchmark_clock_);
	//std::cout << benchmark_duration_.count() << std::endl;
}

// update every object in the level
void Level::Update(float dt)
{
	if (Input::Keyboard().pressed[GLFW_KEY_GRAVE_ACCENT])
		activeCursor = !activeCursor;
	glfwSetInputMode(game_->GetWindow(), GLFW_CURSOR, activeCursor ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);

	// update each camera
	if (!activeCursor)
		for (auto& cam : cameras_)
			cam->Update(dt);

	renderer_.DrawAll();
	DrawImGui();
}

void Level::DrawImGui()
{
	{
		ImGui::Begin("Sun");

		//int shadow = sun_.GetShadowSize().x;
		//if (ImGui::InputInt("Shadow Scale", &shadow, 1024, 1024))
		//{
		//	glm::clamp(shadow, 0, 16384);
		//	sun_.SetShadowSize(glm::ivec2(shadow));
		//}
		if (ImGui::Button("Recompile Debug Map"))
		{
			delete Shader::shaders["debug_map3"];
			Shader::shaders["debug_map3"] = new Shader("debug_map.vs", "debug_map.fs");
		}
		if (ImGui::Button("Recompile Postprocess Shader"))
		{
			//delete Shader::shaders["postprocess"];
			Shader::shaders["postprocess"] = new Shader("postprocess.vs", "postprocess.fs");
		}
		if (ImGui::Button("Recompile Phong Shader"))
		{
			delete Shader::shaders["phong"];
			Shader::shaders["phong"] = new Shader("phong.vs", "phong.fs");
		}

		ImGui::End();
	}

	{
		ImGui::Begin("Info");

		ImGui::Text("FPS: %.0f (%.1f ms)", 1.f / game_->GetDT(), 1000. * game_->GetDT());
		ImGui::NewLine();
		glm::vec3 pos = Render::GetCamera()->GetPos();
		//ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
		if (ImGui::InputFloat3("Camera Position", &pos[0], 2))
			Render::GetCamera()->SetPos(pos);
		pos = Render::GetCamera()->front;
		ImGui::Text("Camera Direction: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
		pos = pos * .5f + .5f;
		ImGui::SameLine();
		ImGui::ColorButton("visualization", ImVec4(pos.x, pos.y, pos.z, 1.f));

		ImGui::NewLine();
		ImGui::Text("Flying: %s", activeCursor ? "False" : "True");
		//ImGui::Text("Vertex Count: %d", renderer_.vtxNmlLines.size());
		//ImGui::Text("Face Count: %d", renderer_.fceNmlLines.size());

		ImGui::End();
	}

	{
		ImGui::Begin("CA Simulation");
		ImGui::Checkbox("Pause Simulation", &renderer_.pauseSimulation);
		if (ImGui::Button("Re-init Simulation"))
			WCA::InitWCA();
		if (ImGui::Button("1x Update Simulation"))
			WCA::UpdateWCA();
		if (ImGui::Button("20x Update Simulation"))
			for (int i = 0; i < 20; i++)
				WCA::UpdateWCA();
		if (ImGui::Button("200x Update Simulation"))
			for (int i = 0; i < 200; i++)
				WCA::UpdateWCA();
		ImGui::End();
	}
}