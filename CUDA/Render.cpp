#include "Render.h"

Render::Render(){
    this->initWindow();
}

Render::~Render(){
    delete this->window;
}

void Render::initWindow(){
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;
    this->videomode.width = 800;
    this->videomode.height = 600;
    this->window = new sf::RenderWindow(this->videomode, "Render", sf::Style::Titlebar, settings);
    this->window->setPosition(sf::Vector2i(videomode.width, 0));
    this->window->setFramerateLimit(60);
}

const bool Render::running() const{
    return this->window->isOpen();
}

void Render::update(){
    this->pollEvents();
}

void Render::render(){
    this->window->clear(sf::Color::Blue);
    this->floors();
    this->walls();
    this->window->display();
}

void Render::pollEvents(){
    while (this->window->pollEvent(this->ev)){
        switch(this->ev.type){
            case sf::Event::Closed:
                this->window->close();
                break;
            case sf::Event::KeyPressed:
                if (this->ev.key.code == sf::Keyboard::Return){
                    this->window->close();
                }
                break;
        }
    }
}

void Render::floors(){
    sf::RectangleShape floor(sf::Vector2f(this->videomode.width, int(this->videomode.height / 2)));
	floor.setPosition(sf::Vector2f(0, int(this->videomode.height / 2)));
	floor.setFillColor(sf::Color(0, 0, 200, 255));

	sf::RectangleShape ceiling(sf::Vector2f(this->videomode.width, int(this->videomode.height / 2)));
	ceiling.setPosition(sf::Vector2f(0, 0));
	ceiling.setFillColor(sf::Color(100, 100, 100, 240));

	this->window->draw(floor);
	this->window->draw(ceiling);
}

void Render::walls(){

    //for (auto& c : this->collisions){
    //    std::cout<<c<<'\n';
    //}

    float center_y = int(this->videomode.height / 2);
    int num = this->collisions.size();
	for (int i = 0; i < num; i++) {
		//int dist = int(this->collisions[i]) * cos(this->ray_angles[i] * (3.1415926/180)));
        int dist = int(this->collisions[i]);
		float mapped_value = this->map(dist, 0, this->videomode.width, int((2.01/3) * this->videomode.height), 50);

		int top_x = int(i * (this->videomode.width / (this->CONEANGLE / this->DEGSTEP)));
		int top_y = int(center_y - mapped_value / 2);
		int width = int(this->videomode.width / (this->CONEANGLE / this->DEGSTEP));
		int height = int(mapped_value);

		//Map colour based off of distance between 'wall' and player
		//Simulates light dimming the further an object is away
		int map_colour = 255 - int(map(mapped_value, 0, int((2.01 / 3) * this->videomode.height), 200, 0));
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