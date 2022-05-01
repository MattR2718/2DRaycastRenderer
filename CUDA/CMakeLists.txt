cmake_minimum_required(VERSION 3.0.0)
project(CudaRaycast VERSION 0.1.0 LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

include(CheckLanguage)
check_language(CUDA)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_executable(CudaRaycast main.cpp Raycast.cpp Render.cpp kernel.cu ray.h)
target_link_libraries(CudaRaycast sfml-graphics sfml-window sfml-system)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
