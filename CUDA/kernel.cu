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

__device__ float intersection(int ax, int ay, int bx, int by, int cx, int cy, int dx, int dy){
    float a = (float)by - ay;
    float b = (float)ax - bx;
    float c = a * ax + b * ay;
    float d = (float)dy - cy;
    float e = (float)cx - dx;
    float f = d * cx + e * cy;
    float det = a * e - d * b;
    if (det == 0){
        return 1000000.0;
    } else{
        float x = (e * c - b * f) / det;
        float y = (a * f - d * c) / det;
        return sqrt((x * x) + (y * y));
    }
}


__global__ void collisions(int n, int N, int width, int bs, Ray *rays, bool *grid, float *cols){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride){
        Ray r = rays[i];
        float closest = 10000000.0;
        for (int g = 0; g < n; g++){
            if(grid[g]){
                int x = (g % width) * bs;
                int y = (g / width) * bs;
                float t = intersection(r.ox, r.oy, r.px, r.py, x, y, x + bs, y);
                float b = intersection(r.ox, r.oy, r.px, r.py, x, y + bs, x + bs, y + bs);
                float l = intersection(r.ox, r.oy, r.px, r.py, x, y, x, y + bs);
                float ri = intersection(r.ox, r.oy, r.px, r.py, x + bs, y, x + bs, y + bs);
                if (t < closest){ closest = t; }
                if (b < closest){ closest = b; }
                if (l < closest){ closest = l; }
                if (ri < closest){ closest = ri; }
            }
        }
        cols[i] = closest;
    }
}

void getCollisionDistance(int& n, int& width, int bs, Ray *rays, bool *grid, float *collis){
    int N = sizeof(rays) / sizeof(*rays);
    float *cols;
    cudaMallocManaged(&cols, N*sizeof(float));
    Ray *rs;
    cudaMallocManaged(&rs, N*sizeof(Ray));
    for (int i = 0; i < N ; i++){
        rs[i] = rays[i];
    }
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    collisions<<<numBlocks, blockSize>>>(n, N, width, bs, rs, grid, cols);
    cudaDeviceSynchronize();
    for (int i = 0; i < N ; i++){
        cols[i] = collis[i];
    }
    cudaFree(rs);
    cudaFree(cols);

}