#include "Raycast.h"

Raycast::Raycast(){
    this->initWindow();
    this->initGrid();
    if (this->LOADGRID) { this->loadGrid(); }
    this->border();
    this->initPlayer();
    this->getMousePos();
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
    //rotateRays(n, rays);
    //rotateRaysCPU(n, rays);
    int N = sizeof(this->grid) / sizeof(*this->grid);
    //getCollisionDistance(this->WIDTH/this->BLOCKSIZE, this->BLOCKSIZE, n, this->rays, this->grid, this->collisions);
    //for (int i = 0; i < n; i++){
    //    this->collisions[i] = i + 20.0;
    //}
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

float Raycast::angleBetween(sf::Vector2i two, sf::Vector2i three, sf::Vector2f one){
    /*
    sf::Vector2f a = sf::Vector2f(mp1.x - pp.x, mp1.y - pp.y);
    sf::Vector2f b = sf::Vector2f(mp2.x - pp.x, mp2.y - pp.y);
    float dot = (a.x*b.x + a.y*b.y);
    //std::cout<<dot<<'\n';
    float maga = sqrt(a.x*a.x+a.y*a.y);
    //std::cout<<maga<<'\n';
    float magb = sqrt(b.x*b.x+b.y*b.y);
    //std::cout<<magb<<'\n';
    float val = dot / (maga * magb);
    //float o = 1.0;
    //val = std::modf(val, o);
    float angle = std::acos(val);
    std::cout<<angle<<'\n';
    return angle;
    */
   float p12 = sqrt((one.x - two.x) * (one.x - two.x) + (one.y - two.y) * (one.y - two.y));
   float p13 = sqrt((one.x - three.x) * (one.x - three.x) + (one.y - three.y) * (one.y - three.y));
   float p23 = sqrt((two.x - three.x) * (two.x - three.x) + (two.y - three.y) * (two.y - three.y));
   float angle = acos((p12*p12 + p13*p13 - p23*p23) / ( 2 * p12 * p13));
   float posOrNeg = (three.x - two.x) * (three.y - two.y);
   //angle *= (posOrNeg > 0) ? 1 : -1;
   return angle;
}

void Raycast::updateRays(){
    sf::Vector2i mp = this->mouse_pos;
    this->getMousePos();
    float angle = this->angleBetween(mp, this->mouse_pos, this->getPlayerCentre());
    angle = this->radToDeg(angle);
    for (auto& r : this->rays){
        r.angle += angle;
    }
}

/*
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
*/

float Raycast::degToRad(float& angle){
    return (angle * PI) / 180;
}

float Raycast::radToDeg(float& angle){
    return (angle * 180) / PI;
}

void Raycast::drawRays(){
    int f = 0;
    sf::Vector2f playerPos = getPlayerCentre();
    for (auto& r : this->rays){
        sf::Vector2f rayEnd = playerPos + sf::Vector2f(r.mag * cos(this->degToRad(r.angle)), r.mag * sin(this->degToRad(r.angle)));
        sf::Vertex line[] = {
            sf::Vertex(playerPos, sf::Color::Red), 
            sf::Vertex(rayEnd, sf::Color::Red)
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

std::vector<float> Raycast::getCollisions(){
    std::vector<float> t;
    for (auto& c : this->collisions){ t.push_back(c); }
    return t;
}

float Raycast::getDegStep(){
    return this->DEGSTEP;
}

float Raycast::getConeAngle(){
    return this->CONEANGLE;
}

sf::Vector2f Raycast::getPlayerCentre(){
    sf::Vector2f pos = this->player.getPosition();
    return sf::Vector2f(pos.x + this->BLOCKSIZE / 2, pos.y + this->BLOCKSIZE / 2);
}