#ifndef RAYCAST
#define RAYCAST

#include <SFML/Graphics.hpp>
#include <vector>
#include <fstream>
#include <iostream>

#include "kernel.cuh"
#include "ray.h"

class Raycast{
    public:
        sf::RenderWindow* window;
        sf::VideoMode videoMode;

        Raycast();
        virtual ~Raycast();

        const bool running() const;

        void update();
        void render();

        

    private:
        sf::Event ev;
        sf::Font font;

        constexpr static int BLOCKSIZE = 20;
        constexpr static float MOVEMENTSPEED = 10.0;
        constexpr static bool LOADGRID = true;

        constexpr static int WIDTH = 800;
        constexpr static int HEIGHT = 600;


        bool grid[(WIDTH / BLOCKSIZE) * (HEIGHT / BLOCKSIZE)];
        sf::Vector2i mouse_pos;

        sf::RectangleShape player;

        sf::Clock delta_clock;

        sf::Color backgroundColour = sf::Color(100, 100, 100, 255);
        sf::Color playerColour = sf::Color::Green;
        sf::Color wallColour = sf::Color::Blue;


        constexpr static float CONEANGLE = 120.0;
        constexpr static float DEGSTEP = 1.5;
        Ray rays[(int)(CONEANGLE / DEGSTEP)];

        void initWindow();
        constexpr void initGrid();
        void border();
        void initPlayer();
        void initRays();
        void pollEvents();
        void drawGrid();
        void loadGrid();
        void getMousePos();
        void updateRays();
        void drawRays();
};
#endif