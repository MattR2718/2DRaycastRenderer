cmake_minimum_required(VERSION 3.0.0)

add_executable(C++RaycastRenderer main.cpp Render.cpp Raycast.cpp)
target_link_libraries(C++RaycastRenderer sfml-graphics sfml-window sfml-system)