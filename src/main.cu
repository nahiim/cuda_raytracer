
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "math.h"
#include "types.h"
#include "kernels.h"


uint8_t* framebuffer;
const int width  = 1080;
const int height = 1080;

Vec3 origin = Vec3(0.0f, 0.0f, 0.0f);
Vec3 front = Vec3(0.0f, 0.0f, -1.0f);

Sphere* spheres;


void saveImage(const char* filename, uint8_t* pixels, int width, int height)
{
    int success = stbi_write_png(filename, width, height, 4, pixels, width * 4);
}




int main()
{
    cudaMallocManaged(&framebuffer, width * height * 4);

    Sphere hostSpheres[5];
    hostSpheres[0] = { Vec3(0, -1, -5), 1.0f, Vec3(1.0f, 0.71f, 0.29f), 0.3f, 0.0f }; 
    hostSpheres[1] = { Vec3(2, 0, -6), 1.0f, Vec3(0.9f, 0.9f, 0.9f), 0.2f, 0.3f }; 
    hostSpheres[2] = { Vec3(-2, 0, -6), 1.0f, Vec3(0.5f, 0.8f, 1.0f), 0.1f, 0.0f }; 
    hostSpheres[3] = { Vec3(0, -5001, -5), 5000.0f, Vec3(0.8f, 0.8f, 0.8f), 0.0f, 0.0f }; 
    hostSpheres[4] = { Vec3(0, 2, -5), 0.5f, Vec3(1.0f, 0.0f, 0.0f), 0.2f, 0.0f }; 

    cudaMalloc(&spheres, sizeof(Sphere) * 5);
    cudaMemcpy(spheres, hostSpheres, sizeof(Sphere) * 5, cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);
    render << <numBlocks, threadsPerBlock >> > (framebuffer, width, height, origin, front, spheres);
    cudaDeviceSynchronize();

    saveImage("render_result/output.png", framebuffer, width, height);

    cudaFree(framebuffer);

    return 0;
}