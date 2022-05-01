#include "Raycast.h"

Raycast::Raycast(){
    this->initWindow();
    this->initGrid();
    if (this->LOADGRID) { this->loadGrid(); }
    this->border();
    this->initPlayer();
    this->initRays();
    this->deltaClock.restart();
    this->prevTime = this->deltaClock.getElapsedTime();
    if (!this->font.loadFromFile("../arial.ttf")){
        
    }
}

Raycast::~Raycast(){
    delete this->window;
}

const bool Raycast::running() const{
    return this->window->isOpen();
}

void Raycast::update(){
    this->pollEvents();
    this->updatePlayer();
    this->updateRays();
    int n = (sizeof(this->rays)/sizeof(*this->rays));
    rotateRays(n, rays);
    //rotateRaysCPU(n, rays);
}

void Raycast::render(){
    this->window->clear(this->backgroundColour);
    this->drawGrid();
    this->window->draw(this->player);
    this->drawRays();
    this->displayFPS();
    this->window->display();
}

void Raycast::initWindow(){
    sf::ContextSettings settings;
    settings.antialiasingLevel = 8;

    this->videoMode.width = this->WIDTH;
    this->videoMode.height = this->HEIGHT;
    this->window = new sf::RenderWindow(this->videoMode, "Raycast", sf::Style::Close | sf::Style::Titlebar, settings);
    this->window->setPosition(sf::Vector2i(0, 0));
    this->window->setFramerateLimit(60);

}

constexpr void Raycast::initGrid(){
    for (auto& g : this->grid){
        g = 0;
    }
}

void Raycast::border(){
    for (int i = 0; i < (sizeof(this->grid)/sizeof(*this->grid)); i++){
        if (i < (this->WIDTH / this->BLOCKSIZE)){
            this->grid[i] = 1;
            this->grid[(sizeof(this->grid)/sizeof(*this->grid)) - i - 1] = 1;
        }
        if ((i % (this->WIDTH / this->BLOCKSIZE)) == 0){ 
            this->grid[i] = 1;
            this->grid[i + (this->WIDTH / this->BLOCKSIZE) - 1] = 1;
        }
    }
}

void Raycast::initPlayer(){
    this->player.setPosition(sf::Vector2f(this->BLOCKSIZE, this->BLOCKSIZE));
    this->player.setSize(sf::Vector2f(this->BLOCKSIZE, this->BLOCKSIZE));
    this->player.setFillColor(this->playerColour);
}

void Raycast::initRays(){
    sf::Vector2f playerCentre = this->player.getPosition();
    this->getMousePos();
    playerCentre.x += this->BLOCKSIZE / 2;
    playerCentre.y += this->BLOCKSIZE / 2;
    int rayCount = 0;
    Ray tempRay;
    tempRay.ox = playerCentre.x;
    tempRay.oy = playerCentre.y;
    tempRay.px = this->mouse_pos.x;
    tempRay.py = this->mouse_pos.y;
    tempRay.m = 0;
    tempRay.c = 0;
    for (float i = (360 - (this->CONEANGLE/ 2)); i < 360; i += this->DEGSTEP){
        tempRay.angle = i;
        this->rays[rayCount++] = tempRay;
    }
    for (float i = 0; i < ((this->CONEANGLE / 2) + 1); i += this->DEGSTEP){
        tempRay.angle = i;
        this->rays[rayCount++] = tempRay;
    }
}

void Raycast::pollEvents(){
    while(this->window->pollEvent(this->ev)){
        switch (this->ev.type){
            case sf::Event::Closed:
                this->window->close();
                break;
        }
    }
}

void Raycast::drawGrid(){
    sf::RectangleShape cell(sf::Vector2f(this->BLOCKSIZE, this->BLOCKSIZE));
    for (int i = 0; i < (sizeof(this->grid)/sizeof(*this->grid)); i++){
        cell.setPosition(sf::Vector2f((i % (this->WIDTH/this->BLOCKSIZE)) * this->BLOCKSIZE, (i / (this->WIDTH/this->BLOCKSIZE)) * this->BLOCKSIZE));
        if (this->grid[i]){ cell.setFillColor(this->wallColour); }
        else { cell.setFillColor(this->backgroundColour); }
        this->window->draw(cell);
    }
}

void Raycast::loadGrid(){
    std::ifstream file("../../CUDA/grid.txt");
    std::string linetxt;
    int row = 0;
    while (std::getline(file, linetxt)){
        int col = 0;
        for (auto& l : linetxt){
            this->grid[(row * (this->WIDTH / this->BLOCKSIZE)) + col++] = l - '0';
            if (((row * (this->WIDTH / this->BLOCKSIZE)) + col) > (sizeof(this->grid)/sizeof(*this->grid))){ std::cout<<"C TOO BIG\n"; break;}
        }
        row++;
    }
}

void Raycast::getMousePos(){
    this->mouse_pos = sf::Mouse::getPosition(*this->window);
}

void Raycast::updateRays(){
    sf::Vector2f playerCentre = this->player.getPosition();
    this->getMousePos();
    playerCentre.x += this->BLOCKSIZE / 2;
    playerCentre.y += this->BLOCKSIZE / 2;
    this->getMousePos();
    for (auto& r : this->rays){
        r.ox = playerCentre.x;
        r.oy = playerCentre.y;
        r.px = this->mouse_pos.x;
        r.py = this->mouse_pos.y;
    }
}

void Raycast::drawRays(){
    int f = 0;
    for (auto& r : this->rays){
        sf::Vertex line[] = {
            sf::Vertex(sf::Vector2f(r.ox, r.oy), sf::Color::Red), 
            sf::Vertex(sf::Vector2f(r.px, r.py), sf::Color::Red)
            //sf::Vertex(sf::Vector2f(r.ox + f, r.oy + f), sf::Color::Red)
        };
        f++;
        this->window->draw(line, 2, sf::Lines);
    }
}

void Raycast::updatePlayer(){
    sf::Time dt = this->deltaClock.restart();
	//Keyboard input
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
		this->player.move(0.f, -this->MOVEMENTSPEED * dt.asSeconds());
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
		this->player.move(0.f, this->MOVEMENTSPEED * dt.asSeconds());
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
		this->player.move(-this->MOVEMENTSPEED * dt.asSeconds(), 0.f);
	}
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
		this->player.move(this->MOVEMENTSPEED * dt.asSeconds(), 0.f);
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

void Raycast::displayFPS(){
    sf::Text text;
    text.setFont(this->font);
    text.setString(std::to_string(this->fps));
    text.setCharacterSize(30);
    text.setFillColor(sf::Color::Black);
    text.setPosition(sf::Vector2f(0, 0));
    this->window->draw(text);
}