#include "kernel.cuh"

void rotateRaysCPU(int& n, Ray *rays){
    const float PI = 3.1415;
    for (int i = 0; i < n; i++){
        std::cout<<i<<'\n';
        float angle = rays[i].angle * PI / 180;
        float qx = rays[i].ox + cos(angle) * (rays[i].px - rays[i].ox) - sin(angle) * (rays[i].py - rays[i].oy);
        float qy = rays[i].oy + sin(angle) * (rays[i].px - rays[i].ox) + cos(angle) * (rays[i].py - rays[i].oy);
        rays[i].px = qx;
        rays[i].py = qy;
    }
}

__global__ void rotate(int n, Ray *rays){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float PI = 3.1415;
    //printf("IN KERNEL");
    for (int i = index; i < n; i += stride){
        //printf("ANGLE %d: %f\n", i, rays[i].angle);
        //printf("OX %d: %d\n", i, rays[i].ox);
        float angle = rays[i].angle * PI / 180;
        float qx = rays[i].ox + cos(angle) * (rays[i].px - rays[i].ox) - sin(angle) * (rays[i].py - rays[i].oy);
        float qy = rays[i].oy + sin(angle) * (rays[i].px - rays[i].ox) + cos(angle) * (rays[i].py - rays[i].oy);
        rays[i].px = round(qx);
        rays[i].py = round(qy);
    }
}

void rotateRays(int& n, Ray *rays){
    int N = n;
    Ray *rs;
    cudaMallocManaged(&rs, N*sizeof(Ray));
    for (int i = 0; i < n ; i++){
        rs[i] = rays[i];
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    rotate<<<numBlocks, blockSize>>>(n, rs);
    cudaDeviceSynchronize();

    for (int i = 0; i < n ; i++){
        rays[i] = rs[i];
    }

    cudaFree(rs);
}