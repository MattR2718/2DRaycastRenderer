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

__device__ float collidex(int width, Ray r, int bs, bool *grid, int step){
    printf("IN COLX %d \n", step);
    int c = r.ox + (bs / 2) * step;
    int y;
    bool collided = false;
    while ((c < ((width - 1) * bs)) && (c > (bs)) && (!collided)){
        y = round(r.m * c + r.c);
        int y2 = floor((float)y / bs);
        if (grid[width * y2 + c]){ collided = true; }
        c += step;
    }
    printf("RETURN\n");
    return sqrt((float)(y * y) + ((c * bs) * (c * bs)));
}

__device__ float collidey(int width, int height, Ray r, int bs, bool *grid, int step){
    printf("IN COLY %d \n", step);
    int c = r.oy + (bs / 2) * step;
    int x;
    bool collided = false;
    while ((c < ((height - 1) * bs)) && (c > bs) && (!collided)){
        x = round((c - r.c) / r.m);
        int x2 = x / bs;
        if (grid[width * c + x2]){ collided = true; }
        c += step;
    }
    float dist = sqrt((float)(x * x) + ((c * bs) * (c * bs)));
    printf("%f\n", dist);
    //printf("dhfiosfhgo");
    return dist;
}


__global__ void collisions(int wGrid, int nRays, int bs, Ray *rays, bool *grid, float *cols){
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("IN KERNEL%d\n", index);
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < nRays; i += stride){
        printf("BEFORE RAY\n");
        Ray r = rays[i];
        printf("AFTER RAY\n");
        //float closest = 10000000.0;
        float closestx = 100;
        float closesty = 10;
        
        if ((r.ox - r.px) > 1){
            closestx = collidex(wGrid, r, bs, grid, -1);
        } else{ 
            closestx = collidex(wGrid, r, bs, grid, 1);
        }
        //int hGrid = (sizeof(grid)/sizeof(*grid))/wGrid;
        printf("OUT OF X\n");
        int hGrid = 30;
        printf("GOING TO Y\n");
        if ((r.oy - r.py) > 1){
            closesty = collidey(wGrid, hGrid, r, bs, grid, -1);
        } else{ 
            closesty = collidey(wGrid, hGrid, r, bs, grid, 1);
        }
        
        if (closestx <= closesty) { cols[i] = closestx; }
        else { cols[i] = closesty; }
        //cols[i] = (float)i;
        //printf("%f\n", closestx);
        //printf("hdihfguhfug");
    }
}

void getCollisionDistance(int wGrid, int bs, int nRays, Ray *rays, bool *grid, float *collis){
    //int N = sizeof(rays) / sizeof(*rays);
    int N = nRays;
    //printf("%d\n", N);
    
    float *cols;
    //printf("MALLOC COLS\n");
    cudaMallocManaged(&cols, N*sizeof(float));
    Ray *rs;
    //printf("MALLOC RS\n");
    cudaMallocManaged(&rs, N*sizeof(Ray));
    //printf("SET RAYS\n");
    for (int i = 0; i < N ; i++){
        rs[i] = rays[i];
        cols[i] = 0.0;
    }
    //printf("BLOCKSIZE\n");
    int blockSize = 256;
    //printf("NUMBLOCKS\n");
    int numBlocks = (N + blockSize - 1) / blockSize;
    //printf("GOING INTO KERNEL\n");
    collisions<<<numBlocks, blockSize>>>(wGrid, N, bs, rs, grid, cols);
    cudaDeviceSynchronize();
    //printf("SYNCHRONIZED\n");
    
    //printf("LOOP\n");
    for (int i = 0; i < N ; i++){
        collis[i] = cols[i];
        //collis[i] = 100 + i;
        printf("%f\n", cols[i]);
        //collis[i] = 100;
    }
    //printf("OUT OF LOOP\n");
    cudaFree(rs);
    //printf("FREEING\n");
    cudaFree(cols);
    //printf("FREED\n");
    //printf("odijoidfjgoijog");
}