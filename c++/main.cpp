#include "Raycast.h"
#include "Render.h"

//Fps counter
float fps;
sf::Clock fps_clock = sf::Clock::Clock();
sf::Time previous_time = fps_clock.getElapsedTime();
sf::Time current_time;



int main() {

	//Init raycast object
	Raycast raycast;
	Render render;

	if (!raycast.window->hasFocus()) {
		raycast.window->requestFocus();
	}

	//Game loop
	while (raycast.running()) {
		//FPS
		current_time = fps_clock.getElapsedTime();
		raycast.fps = int(1.f / (current_time.asSeconds() - previous_time.asSeconds()));
		//std::cout << fps << '\n';
		previous_time = current_time;


		//
		//Update
		//
		raycast.update();
		
		for (auto& r : raycast.rays) {
			render.ray_angles.push_back(r.angle);
		}

		render.player_pos = raycast.get_player_center();
		render.CONE_ANGLE = raycast.CONE_ANGLE;
		render.DEG_STEP = raycast.DEG_STEP;
		render.place_objects = raycast.place_objects;
		render.closest_collisions = raycast.closest_collisions;
		render.update();

		//
		//Render
		//
		raycast.render();
		render.render();
	}
}