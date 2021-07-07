#include "Ray.h"
#include "Raycast.h"

void Ray::initVariables()
{
	this->c = 0;
	this->m = 0;
	this->origin_player = sf::Vector2f(0.f, 0.f);
	this->point = sf::Vector2f(0.f, 0.f);
}

//Constructors and Destructors
Ray::Ray(float angle)
{
	this->initVariables();
	this->angle = angle;
}

Ray::~Ray()
{
}

void Ray::set_origin()
{

}

void Ray::update()
{
	
}

void Ray::render()
{
}
