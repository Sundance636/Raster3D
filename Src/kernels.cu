#include "kernels.h"


__global__ void render(vec3 *frameBuffer, int pixels_x, int pixels_y , vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, entity** world) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= pixels_x) || (j >= pixels_y)) return;

    int pixel_index = j*pixels_x + i;

    float u = float(i) / float(pixels_x);//ratio representing the position of u
    float v = float(j) / float(pixels_y);//ratio representing the position of v

    //frameBuffer[pixel_index] = colour(r,world);
}








void renderBuffer(vec3* d_fb, int tx, int ty, entity** d_world) {
    // Render our buffer
    dim3 blocks(nx/tx+1,NY/ty+1);
    dim3 threads(tx,ty);

    //to a 4 by 3 aspect ratio window/ 3d space
    render<<<blocks, threads>>>(d_fb, nx, NY,
                                vec3(-4.0, -3.0, -4.0),//lowest left point of 3d space
                                vec3(8.0, 0.0, 0.0),//the width of space (pos and neg)
                                vec3(0.0, 6.0, 0.0),//height of the space
                                vec3(0.0, 0.0, 0.0),
                                d_world);// where the origin is defined

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

}


