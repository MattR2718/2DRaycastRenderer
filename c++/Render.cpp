#include "Render.h"

//Private functions
void Render::initVariables()
{
	this->window = nullptr;
}

void Render::initWindow()
{
	sf::ContextSettings settings;
	settings.antialiasingLevel = 8;

	this->videoMode.width = 800;
	this->videoMode.height = 600;
	this->window = new sf::RenderWindow(this->videoMode, "Render", sf::Style::Titlebar, settings);
	this->window->setPosition(sf::Vector2i(videoMode.width, 0));
	this->window->setFramerateLimit(60);
}


//Constructors and Destructors
Render::Render()
{
	this->initVariables();
	this->initWindow();
}

Render::~Render()
{
	delete this->window;
}

//Accessors
const bool Render::running() const
{
	return this->window->isOpen();
}

void Render::floors()
{
	sf::RectangleShape floor(sf::Vector2f(this->videoMode.width, int(this->videoMode.height / 2)));
	floor.setPosition(sf::Vector2f(0, int(this->videoMode.height / 2)));
	floor.setFillColor(sf::Color(0, 0, 200, 255));

	sf::RectangleShape ceiling(sf::Vector2f(this->videoMode.width, int(this->videoMode.height / 2)));
	ceiling.setPosition(sf::Vector2f(0, 0));
	ceiling.setFillColor(sf::Color(100, 100, 100, 240));

	this->window->draw(floor);
	this->window->draw(ceiling);
}

void Render::walls()
{
	float center_y = int(this->videoMode.height / 2);
	for (int i = 0; i < closest_collisions.size(); i++) {
		int dist = int(sqrt(pow(this->closest_collisions[i].x - this->player_pos.x, 2) + pow(this->closest_collisions[i].y - this->player_pos.y, 2)) * cos(this->ray_angles[i] * (3.1415926/180)));
		float mapped_value = this->map(dist, 0, this->videoMode.width, int((2.01/3) * this->videoMode.height), 50);

		int top_x = int(i * (this->videoMode.width / (this->CONE_ANGLE / this->DEG_STEP)));
		int top_y = int(center_y - mapped_value / 2);
		int width = int(this->videoMode.width / (this->CONE_ANGLE / this->DEG_STEP));
		int height = int(mapped_value);

		//Map colour based off of distance between 'wall' and player
		//Simulates light dimming the further an object is away
		int map_colour = 255 - int(map(mapped_value, 0, int((2.01 / 3) * this->videoMode.height), 200, 0));
		sf::RectangleShape wall(sf::Vector2f(width, height));
		wall.setPosition(sf::Vector2f(top_x, top_y));
		wall.setFillColor(sf::Color(0, map_colour, 100, 255));
		this->window->draw(wall);
	}
}

int Render::map(int value, int leftMin, int leftMax, int rightMin, int rightMax)
{
	/*# Figure out how 'wide' each range is
	leftSpan = leftMax - leftMin
	rightSpan = rightMax - rightMin

	# Convert the left range into a 0-1 range (float)
	valueScaled = float(value - leftMin) / float(leftSpan)

	# Convert the 0-1 range into a value in the right range.
	return rightMin + (valueScaled * rightSpan)*/

	float leftSpan = leftMax - leftMin;
	float rightSpan = rightMax - rightMin;
	float value_scaled = float(value - leftMin) / float(leftSpan);
	return rightMin + (value_scaled * rightSpan);
	//return 0;
}

//Functions
void Render::pollEvents()
{
	//Event polling
	while (this->window->pollEvent(this->ev)) {
		switch (this->ev.type) {
		case sf::Event::Closed:
			this->window->close();
			break;

		case sf::Event::KeyPressed:
			if (this->ev.key.code == sf::Keyboard::Return) {
				this->window->close();
			}
			break;
		}
	}
}

void Render::update()
{
	//Everything thats not drawing

	this->pollEvents();

}

void Render::render()
{
	//Drawing
	this->window->clear(sf::Color::Blue);
	
	if (!this->place_objects) {
		this->floors();
		this->walls();
	}

	//Draw everything
	this->window->display();
}