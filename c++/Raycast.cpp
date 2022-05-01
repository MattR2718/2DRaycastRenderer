#include "Raycast.h"


//Private functions
void Raycast::initVariables()
{
	this->window = nullptr;
	this->place_objects = true;
	this->MOVEMENT_SPEED = 200.f;

	this->font;
	if (!font.loadFromFile("../../c++/arial.ttf"))
	{
		// error...
	}
}

void Raycast::initWindow()
{
	sf::ContextSettings settings;
	settings.antialiasingLevel = 8;

	this->videoMode.width = 800;
	this->videoMode.height = 600;
	this->window = new sf::RenderWindow(this->videoMode, "Raycast", sf::Style::Close | sf::Style::Titlebar, settings);
	this->window->setPosition(sf::Vector2i(0, 0));
	this->window->setFramerateLimit(60);

	//Create 2d vector of 0's to store grid
	std::vector<int> filler(this->videoMode.width / this->BLOCKSIZE, 0);
	for (int y = 0; y < this->videoMode.height / this->BLOCKSIZE; y++) {
		this->grid.push_back(filler);
	}
	this->border();
	if (this->LOAD_GRID){
		this->loadGrid();
	}
}

void Raycast::initPlayer()
{
	this->player.setPosition(sf::Vector2f(this->BLOCKSIZE, this->BLOCKSIZE));
	this->player.setSize(sf::Vector2f(this->BLOCKSIZE, this->BLOCKSIZE));
	this->player.setFillColor(sf::Color::Green);	
}

void Raycast::initRays()
{
	sf::Vector2f player_center = this->player.getPosition();
	player_center.x += this->BLOCKSIZE / 2;
	player_center.y += this->BLOCKSIZE / 2;
	for (int i = int(360 - this->CONE_ANGLE / 2); i < 360; i += this->DEG_STEP) {
		Ray temp_ray;
		temp_ray.angle = i;
		temp_ray.origin = player_center;
		this->rays.push_back(temp_ray);
	}
	for (int i = 0; i < int(this->CONE_ANGLE / 2) + 1; i += this->DEG_STEP) {
		Ray temp_ray;
		temp_ray.angle = i;
		temp_ray.origin = player_center;
		this->rays.push_back(temp_ray);
	}
	for (auto& r : rays) {
		r.m = (r.origin.y - r.point.y) / (r.origin.x - r.point.x);
		//y= mx + c
		//c = y - mx
		r.c = r.origin.y = r.m * r.origin.x;
	}
}

void Raycast::loadGrid()
{
	std::ifstream file("../../c++/grid.txt");
	std::string linetxt;
	int y = 0;
	while (std::getline(file, linetxt)) {
		for (int x = 0; x < this->grid[0].size(); x++) {
			char c = linetxt[x];
			grid[y][x] = c - '0';
		}
		y++;
	}
}

bool Raycast::checkWhole(float num)
{
	if (fmod(num, 1) == 0) {
		return true;
	}
	else { return false; }
}

//Constructors and Destructors
Raycast::Raycast()
{
	this->initVariables();
	this->initWindow();
	this->initPlayer();
	this->initRays();
}

Raycast::~Raycast()
{
	delete this->window;
}

//Accessors
const bool Raycast::running() const
{
	return this->window->isOpen();
}

sf::Vector2f Raycast::get_player_center()
{
	return sf::Vector2f(int(this->player.getPosition().x + this->BLOCKSIZE / 2), int(this->player.getPosition().y + this->BLOCKSIZE / 2));
}

//Functions
void Raycast::border()
{
	for (int x = 0; x < this->videoMode.width / this->BLOCKSIZE; x++) {
		grid[0][x] = 1;
		grid[(this->videoMode.height / this->BLOCKSIZE) - 1][x] = 1;
	}
	for (int y = 0; y < this->videoMode.height / this->BLOCKSIZE; y++) {
		grid[y][0] = 1;
		grid[y][(this->videoMode.width / this->BLOCKSIZE) - 1] = 1;
	}
}

void Raycast::draw_grid()
{
	sf::RectangleShape vertical(sf::Vector2f(1.f, this->videoMode.height));
	sf::RectangleShape horizontal(sf::Vector2f(this->videoMode.width, 1.f));
	vertical.setFillColor(sf::Color::Black);
	horizontal.setFillColor(sf::Color::Black);
	for (int x = 0; x < this->videoMode.width; x += this->BLOCKSIZE) {
		vertical.setPosition(sf::Vector2f(x, 0.f));
		this->window->draw(vertical);
	}
	for (int y = 0; y < this->videoMode.height; y += this->BLOCKSIZE) {
		horizontal.setPosition(sf::Vector2f(0.f, y));
		this->window->draw(horizontal);
	}
}

void Raycast::draw_walls()
{
	sf::RectangleShape wall(sf::Vector2f(this->BLOCKSIZE, this->BLOCKSIZE));
	wall.setFillColor(sf::Color::Blue);
	for (int y = 0; y < this->grid.size(); y++) {
		for (int x = 0; x < this->grid[0].size(); x++) {
			if (grid[y][x] == 1) {
				wall.setPosition(sf::Vector2f(x * this->BLOCKSIZE, y * this->BLOCKSIZE));
				this->window->draw(wall);
			}
		}
	}
}

void Raycast::draw_rays()
{
	int count = 0;
	for (auto& r : rays) {
		r.point.x = this->closest_collisions[count].x;
		r.point.y = this->closest_collisions[count].y;
		count++;
	}


	//Draw rays on window
	count = 0;
	for (auto& r : this->rays) {
		sf::Vertex line[] = {
			sf::Vertex(sf::Vector2f(r.point.x, r.point.y), sf::Color::Yellow),
			sf::Vertex(sf::Vector2f(r.origin.x, r.origin.y), sf::Color::Yellow)
		};
		this->window->draw(line, 2, sf::LineStrip);
		count++;
	}
}

void Raycast::update_rays()
{
	//Update ray locations
	sf::Vector2f player_center = this->player.getPosition();
	player_center.x += this->BLOCKSIZE / 2;
	player_center.y += this->BLOCKSIZE / 2;
	
	this->updateMousePosition();

	for (auto& r : this->rays) {
		r.origin = sf::Vector2f(player_center.x, player_center.y);
		r.point = sf::Vector2f(this->mouse_pos.x, this->mouse_pos.y);
		r.point = this->rotate_rays(r.origin, r.point, r.angle);

		r.m = (r.origin.y - r.point.y) / (r.origin.x - r.point.x);
		if (r.m == 0 || std::isinf(r.m)) {
			r.m = 0.0001;
		}
		r.c = r.origin.y - r.m * r.origin.x;
	}

}

sf::Vector2f Raycast::rotate_rays(sf::Vector2f origin, sf::Vector2f point, float angle)
{
	float pi = 3.14159265359;

	angle *= pi / 180;

	float ox = origin.x;
	float oy = origin.y;
	float px = point.x;
	float py = point.y;

	float qx = ox + cos(angle) * (px - ox) - sin(angle) * (py - oy);
	float qy = oy + sin(angle) * (px - ox) + cos(angle) * (py - oy);
	return sf::Vector2f(qx, qy);

}

void Raycast::check_collisions()
{
	//this->collisions.clear();
	this->closest_collisions.clear();
	for (auto& r : rays) {
		this->collisions.clear();
		for (int y = 0; y < this->videoMode.height / this->BLOCKSIZE; y++) {
			for (int x = 0; x < this->videoMode.width / this->BLOCKSIZE; x++) {
				if (this->grid[y][x] == 1) {
					//Define borders of wall
					//Left x, right x, top y, bottom y
					float lx = x * this->BLOCKSIZE;
					float rx = x * this->BLOCKSIZE + this->BLOCKSIZE;
					float ty = y * this->BLOCKSIZE;
					float by = y * this->BLOCKSIZE + this->BLOCKSIZE;
					//for (auto& r : rays) {
					//Check x axis collisions
					float ly = r.m * lx + r.c;
					float ry = r.m * rx + r.c;
					if (r.m > 0) {
						if (ly <= ty && ry >= ty) {
							float itx = (ty - r.c) / r.m;
							if (same_side(sf::Vector2f(itx, ty), r)) {
								this->collisions.push_back(sf::Vector2f(int(itx), ty));
							}
						}
						if (ly <= by && ry >= by) {
							float ibx = (by - r.c) / r.m;
							if (same_side(sf::Vector2f(ibx, by), r)) {
								this->collisions.push_back(sf::Vector2f(int(ibx), by));
							}
						}
						//For negative m, flip comparison operator
					}
					else if (r.m < 0) {
						if (ly >= ty && ry <= ty) {
							float itx = (ty - r.c) / r.m;
							if (same_side(sf::Vector2f(itx, ty), r)) {
								this->collisions.push_back(sf::Vector2f(int(itx), ty));
							}
						}
						if (ly >= by && ry <= by) {
							float ibx = (by - r.c) / r.m;
							if (same_side(sf::Vector2f(ibx, by), r)) {
								this->collisions.push_back(sf::Vector2f(int(ibx), by));
							}
						}
					}
					else { std::cout << "\nERROR WITH X AXIS COLLISIONS M\n"; }
					float tx = (ty - r.c) / r.m;
					float bx = (by - r.c) / r.m;
					if (r.m > 0) {
						if (tx <= lx && bx >= lx) {
							float ily = r.m * lx + r.c;
							if (same_side(sf::Vector2f(lx, int(ily)), r)) {
								this->collisions.push_back(sf::Vector2f(lx, int(ily)));
							}
						}
						if (tx <= rx && bx >= rx) {
							float iry = r.m * rx + r.c;
							if (same_side(sf::Vector2f(rx, int(iry)), r)) {
								this->collisions.push_back(sf::Vector2f(rx, int(iry)));
							}
						}
						//For negative m, flip comparison operator
					}
					else if (r.m < 0) {
						if (tx >= lx && bx <= lx) {
							float ily = r.m * lx + r.c;
							if (same_side(sf::Vector2f(lx, int(ily)), r)) {
								this->collisions.push_back(sf::Vector2f(lx, int(ily)));
							}
						}
						if (tx >= rx && bx <= rx) {
							float iry = r.m * rx + r.c;
							if (same_side(sf::Vector2f(rx, int(iry)), r)) {
								this->collisions.push_back(sf::Vector2f(rx, int(iry)));
							}
						}
					}
					else { std::cout << "\nERROR WITH Y AXIS COLLISIONS M\n"; }
					//}
				}
			}
		}
		this->closest_collisions.push_back(this->find_closest_collision(collisions));
	}

}

bool Raycast::same_side(sf::Vector2f point, Ray ray)
{
	sf::Vector2f player_center = this->player.getPosition();
	player_center.x += this->BLOCKSIZE / 2;
	player_center.y += this->BLOCKSIZE / 2;

	this->updateMousePosition();

	//Mouse line
	//m = y2 - y1 / x2 - x1
	float mouse_m = -1 / ((this->mouse_pos.y - player_center.y)/(this->mouse_pos.x - player_center.x));
	//y = mx + c
	//c = y - mx
	float mouse_c = player_center.y - mouse_m * player_center.x;

	float p1 = mouse_m * point.x + mouse_c;
	float p2 = mouse_m * this->mouse_pos.x + mouse_c;

	sf::Vertex line[] = {
		sf::Vertex(sf::Vector2f(0, mouse_c), sf::Color::Magenta),
		sf::Vertex(sf::Vector2f(800, mouse_m * 800 + mouse_c), sf::Color::Magenta)
	};
	this->window->draw(line, 2, sf::LineStrip);

	//Check if both points are on same side
	if ((this->mouse_pos.y >= mouse_m * this->mouse_pos.x + mouse_c && point.y >= mouse_m * point.x + mouse_c) || (this->mouse_pos.y <= mouse_m * mouse_pos.x + mouse_c && point.y <= mouse_m * point.x + mouse_c)) {
		return true;
	}
	else { 
		return false;
	}
}

sf::Vector2f Raycast::find_closest_collision(std::vector<sf::Vector2f>& collisions)
{
	sf::Vector2f player_center = this->player.getPosition();
	player_center.x += this->BLOCKSIZE / 2;
	player_center.y += this->BLOCKSIZE / 2;

	float short_dist = 100000.f;
	sf::Vector2f shortest = sf::Vector2f(0, 0);

	for (auto& hits : collisions) {
		float dist = sqrt(pow((hits.x - player_center.x), 2) + pow((hits.y - player_center.y), 2));
		if (dist < short_dist) {
			short_dist = dist;
			shortest = sf::Vector2f(hits.x, hits.y);
		}
	}
	return shortest;
}

void Raycast::draw_collisions()
{
	for (auto& col : this->closest_collisions) {
		sf::CircleShape shape(3);
		shape.setFillColor(sf::Color::Red);
		shape.setPosition(col.x, col.y);
		this->window->draw(shape);
	}
}

void Raycast::mouse_line()
{
	this->updateMousePosition();
	sf::Vector2f player_center = this->player.getPosition();
	player_center.x += this->BLOCKSIZE / 2;
	player_center.y += this->BLOCKSIZE / 2;
	sf::Vertex line[] = {
		sf::Vertex(sf::Vector2f(this->mouse_pos.x, this->mouse_pos.y), sf::Color::Red),
		sf::Vertex(player_center, sf::Color::Red)
	};
	this->window->draw(line, 2, sf::LineStrip);
}

void Raycast::updateMousePosition()
{
	//Takes reference to window not window (*)
	this->mouse_pos = sf::Mouse::getPosition(*this->window);
}

void Raycast::move_player()
{
	sf::Time dt = this->delta_clock.restart();
	//Keyboard input
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
		this->player.move(0.f, -this->MOVEMENT_SPEED * dt.asSeconds());
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
		this->player.move(0.f, this->MOVEMENT_SPEED * dt.asSeconds());
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
		this->player.move(-this->MOVEMENT_SPEED * dt.asSeconds(), 0.f);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
		this->player.move(this->MOVEMENT_SPEED * dt.asSeconds(), 0.f);
	}

	//Window bound collision
	if (this->player.getGlobalBounds().left < 0 + this->BLOCKSIZE) {
		this->player.move(abs(this->BLOCKSIZE - this->player.getGlobalBounds().left), 0.f);
	}
	if (this->player.getGlobalBounds().top < 0 + this->BLOCKSIZE) {
		this->player.move(0.f, abs(this->BLOCKSIZE - this->player.getGlobalBounds().top));
	}
	if (this->player.getGlobalBounds().left + this->player.getGlobalBounds().width > this->videoMode.width - this->BLOCKSIZE) {
		this->player.move(-((this->player.getGlobalBounds().left + this->player.getGlobalBounds().width) - (this->videoMode.width - this->BLOCKSIZE)), 0.f);
	}
	if (this->player.getGlobalBounds().top + this->player.getGlobalBounds().height > this->videoMode.height - this->BLOCKSIZE) {
		this->player.move(0.f, -((this->player.getGlobalBounds().top + this->player.getGlobalBounds().height) - (this->videoMode.height - this->BLOCKSIZE)));
	}

}

void Raycast::displayFPS()
{
	sf::Text text;

	// select the font
	text.setFont(this->font);
	// set the string to display
	text.setString(std::to_string(this->fps));
	// set the character size
	text.setCharacterSize(30);
	// set the color
	text.setFillColor(sf::Color::Red);
	//Set position
	text.setPosition(sf::Vector2f(0, 0));
	this->window->draw(text);

}

void Raycast::saveGrid()
{
	std::ofstream file("grid.txt");
	for (int y = 0; y < this->grid.size(); y++) {
		for (int x = 0; x < this->grid[0].size(); x++) {
			file << grid[y][x];
		}
		file << '\n';
	}
	file.close();
}

void Raycast::pollEvents()
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
			//Switching between place and casting rays
			if (this->ev.key.code == sf::Keyboard::Space) {
				this->place_objects = false;
				this->border();
				this->grid[1][1] = 0;
				this->saveGrid();
			}
			break;
		case sf::Event::MouseButtonPressed:
			if (this->place_objects) {
				this->updateMousePosition();
				if (this->grid[this->mouse_pos.y / this->BLOCKSIZE][this->mouse_pos.x / this->BLOCKSIZE] == 0) {
					this->grid[this->mouse_pos.y / this->BLOCKSIZE][this->mouse_pos.x / this->BLOCKSIZE] = 1;
				}
				else if (this->grid[this->mouse_pos.y / this->BLOCKSIZE][this->mouse_pos.x / this->BLOCKSIZE] == 1) {
					this->grid[this->mouse_pos.y / this->BLOCKSIZE][this->mouse_pos.x / this->BLOCKSIZE] = 0;
				}
			}
			break;
		}
	}
}

void Raycast::update()
{
	//Everything thats not drawing
	
	//Poll events
	this->pollEvents();

	//Player movement
	if (!this->place_objects) {
		this->move_player();
		this->update_rays();
		
	}
}

void Raycast::render()
{
	//Drawing
	this->window->clear(sf::Color(100, 100, 100, 255));

	//Draw everything
	this->draw_grid();

	this->draw_walls();

	if (!this->place_objects) {
		this->mouse_line();
		this->window->draw(this->player);
		this->check_collisions();
		this->draw_collisions();
		this->draw_rays();
	}

	this->displayFPS();

	this->window->display();
}