#ifndef RAYCAST_KERNEL
#define RAYCAST_KERNEL
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include "ray.h"


void rotateRays(int& n, Ray *rays);
void rotateRaysCPU(int& n, Ray *rays);

#endif