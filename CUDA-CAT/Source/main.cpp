#include "stdafx.h"

#include "Pipeline.h"
#include "Engine.h"
#include "Renderer.h"
#include "Interface.h"

GLFWwindow* win = nullptr;

int main()
{
	EngineConfig cfg;
	cfg.verticalSync = true;
	Engine::Init(cfg);
	Renderer::Init();
	Interface::Init();

	Engine::Run();

	Engine::Cleanup();

	return 0;
}