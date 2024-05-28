#ifndef __gpu_h__
#define __gpu_h__

#include <iostream>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "entity_list.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

#define nx 640
#define ny 480

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

void* allocateFb(vec3*, int, int);
void** allocateList(entity**,size_t);
void** allocateWorld(entity**,size_t);

void renderBuffer(vec3*,int,int,entity**);
void freeGPU(vec3*,entity**,entity**);
void transferMem(vec3*,vec3*);
void initializeScenes(entity** &d_list, entity** &d_world);

__global__ void create_world(entity**, entity**);
__device__ vec3 colour(const ray&,entity**);
__device__ float hit_sphere(const vec3&, float, ray);
__global__ void free_world(entity **d_list, entity **d_world);
__global__ void render(vec3 *frameBuffer, int pixels_x, int pixels_y , vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin, entity** world);

#endif