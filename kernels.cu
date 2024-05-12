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



void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


void* allocateFb(vec3* d_fb, int width, int height) {
    int num_pixels = width*height;
    size_t fb_size = num_pixels*sizeof(vec3);

    checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));

    //std::cout << d_fb << "\n";
    //checkCudaErrors(cudaMallocManaged((void **)&d_fb, fb_size));

    return d_fb;
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

void freeGPU(vec3* d_fb,entity** d_list, entity** d_world) {
    checkCudaErrors(cudaFree(d_fb));
    
    //free_world<<<1,1>>>(d_list,d_world);

}

void transferMem(vec3* h_fb,vec3* d_fb) {
    int num_pixels = nx*NY;
    size_t fb_size = 3*num_pixels*sizeof(float);
    std::cout << "Device frame buffer address: " << d_fb << "\n";
    std::cout << "Host frame buffer address: " << h_fb << "\n";
    std::cout << "FrameBuffer Size: " << fb_size << "\n";

    checkCudaErrors(cudaMemcpy(h_fb,d_fb,fb_size, cudaMemcpyDeviceToHost));



}

