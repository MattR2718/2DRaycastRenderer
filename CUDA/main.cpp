#include <iostream>

#include "Raycast.h"
#include "Render.h"

int main(){ 
    Raycast raycast;

    while (raycast.running()){
        raycast.update();
        raycast.render();

    }
}