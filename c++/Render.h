#pragma once

#include <SFML/Graphics.hpp>
#include <vector>
#include "Raycast.h"

//Class for top down raycast view
class Render
{
private:
	//Variables
	//Window
	sf::RenderWindow* window;
	sf::VideoMode videoMode;
	sf::Event ev;
	

	//Private functions
	void initVariables();
	void initWindow();

public:

	//Variables
	std::vector<sf::Vector2f> closest_collisions;
	sf::Vector2f player_pos;
	//Boolean
	bool place_objects;

	//Constants
	int CONE_ANGLE = 30;
	int DEG_STEP = 3;

	std::vector<int> ray_angles;

	//Constructors and destructors
	Render();
	virtual ~Render();


	//Accsessors
	const bool running() const;

	//Functions
	void floors();
	void walls();
	int map(int value, int leftMin, int leftMax, int rightMin, int rightMax);
	void pollEvents();
	void update();
	void render();
};