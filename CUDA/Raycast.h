#ifndef RAYCAST
#define RAYCAST

#include <SFML/Graphics.hpp>
#include <vector>
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>

#include "kernel.cuh"
#include "ray.h"

#define PI 3.14159

class Raycast{
    public:
        sf::RenderWindow* window;
        sf::VideoMode videoMode;

        int fps;

        Raycast();
        virtual ~Raycast();

        const bool running() const;

        void update();
        void render();

        std::vector<float> getCollisions();
        float getDegStep();
        float getConeAngle();
        

    private:
        sf::Event ev;
        sf::Font font;

        constexpr static int BLOCKSIZE = 20;
        constexpr static float MOVEMENTSPEED = 100.0;
        constexpr static bool LOADGRID = true;

        constexpr static int WIDTH = 800;
        constexpr static int HEIGHT = 600;


        bool grid[(WIDTH / BLOCKSIZE) * (HEIGHT / BLOCKSIZE)];

        

        sf::Vector2i mouse_pos;

        sf::RectangleShape player;

        sf::Clock deltaClock;
        sf::Time prevTime;
        sf::Time currentTime;

        sf::Color backgroundColour = sf::Color(100, 100, 100, 255);
        sf::Color playerColour = sf::Color::Green;
        sf::Color wallColour = sf::Color::Blue;


        constexpr static float CONEANGLE = 120.0;
        constexpr static float DEGSTEP = 1.5;
        Ray rays[(int)(CONEANGLE / DEGSTEP)];
        float collisions[(int)(CONEANGLE / DEGSTEP)];

        void initWindow();
        constexpr void initGrid();
        void border();
        void initPlayer();
        void initRays();
        void pollEvents();
        void drawGrid();
        void loadGrid();
        void getMousePos();
        float angleBetween(sf::Vector2i mp1, sf::Vector2i mp2, sf::Vector2f pp);
        void updateRays();
        float degToRad(float& angle);
        float radToDeg(float& angle);
        void drawRays();
        void updatePlayer();
        void displayFPS();
        sf::Vector2f getPlayerCentre();
};
#endif