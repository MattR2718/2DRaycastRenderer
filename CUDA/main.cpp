#include <iostream>

#include "Raycast.h"
#include "Render.h"

sf::Clock fpsClock = sf::Clock();
sf::Time prev = fpsClock.getElapsedTime();
sf::Time cur;

int main(){ 
    Raycast raycast;

    while (raycast.running()){

        cur = fpsClock.getElapsedTime();
        raycast.fps = int(1.f / (cur.asSeconds() - prev.asSeconds()));
        prev = cur;

        raycast.update();
        raycast.render();

    }
}