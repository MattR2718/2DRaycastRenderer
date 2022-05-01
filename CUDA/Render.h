#ifndef RENDER
#define RENDER

#include <SFML/Graphics.hpp>
#include <vector>
#include "Raycast.h"

class Render{
    public:
        Render();
        virtual ~Render();
        const bool running() const;
        void update();
        void render();
        std::vector<float> collisions;

        float CONEANGLE;
        float DEGSTEP;

    private:
        sf::RenderWindow* window;
        sf::VideoMode videomode;
        sf::Event ev;

        

        void initWindow();
        void floors();
        void walls();
        void pollEvents();
        int map(int value, int leftMin, int leftMax, int rightMin, int rightMax);
};

#endif