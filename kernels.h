#ifndef __kernels_h__
#define __kernels_h__

#include <iostream>
#include "entity.h"
#include "vec3.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define nx 640
#define NY 480
#define ASPECT_RATIO 4.0f/3.0f


void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

void* allocateFb(vec3*, int, int);

void renderBuffer(vec3*,int,int,entity**);
void freeGPU(vec3*,entity**,entity**);
void transferMem(vec3*,vec3*);

__global__ void render(vec3 *frameBuffer, int pixels_x, int pixels_y , vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, entity** world);

#endif