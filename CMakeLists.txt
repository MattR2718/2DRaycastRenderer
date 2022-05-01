cmake_minimum_required(VERSION 3.0.0)
project(RaycastRender)

include_directories(/usr/include/SFML/)
link_directories(/usr/lib/aarch64-linux-gnu/)

add_subdirectory(c++)
add_subdirectory(CUDA)

