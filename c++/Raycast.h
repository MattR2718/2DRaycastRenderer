#pragma once

#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

//Class for top down raycast view
class Raycast
{
private:
	//Variables
	//Window
	sf::Event ev;

	sf::Font font;

	//Constants
	int BLOCKSIZE = 20;
	float MOVEMENT_SPEED;
	bool LOAD_GRID = true;

	//Grid
	std::vector<std::vector<int>> grid;

	//Mouse position
	sf::Vector2i mouse_pos;

	//Objects
	sf::RectangleShape player;

	//Delta time clock
	sf::Clock delta_clock;

	//Private functions
	void initVariables();
	void initWindow();
	void initPlayer();
	void initRays();
	void loadGrid();
	bool checkWhole(float num);

public:

	//Window
	sf::RenderWindow* window;
	sf::VideoMode videoMode;

	//Variables
	//Collisions vector
	std::vector<sf::Vector2f> collisions;
	std::vector<sf::Vector2f> closest_collisions;
	int fps;
	//Boolean
	bool place_objects;

	//Rays
	struct Ray {
		float angle;
		float c;
		float m;
		sf::Vector2f origin;
		sf::Vector2f point;
	};
	std::vector<Ray> rays;

	//Constants
	int CONE_ANGLE = 120;
	int DEG_STEP = 1;


	//Constructors and destructors
	Raycast();
	virtual ~Raycast();


	//Accsessors
	const bool running() const;

	//Functions
	sf::Vector2f get_player_center();
	void border();
	void draw_grid();
	void draw_walls();
	void draw_rays();
	void update_rays();
	sf::Vector2f rotate_rays(sf::Vector2f origin, sf::Vector2f point, float angle);
	void check_collisions();
	bool same_side(sf::Vector2f point, Ray ray);
	sf::Vector2f find_closest_collision(std::vector<sf::Vector2f>& collisions);
	void draw_collisions();
	void mouse_line();
	void updateMousePosition();
	void move_player();
	void displayFPS();
	void saveGrid();
	void pollEvents();
	void update();
	void render();
};