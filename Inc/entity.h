#ifndef __entity_h__
#define __entity_h__

#include "triangle.h"

#include <fstream>
#include <sstream>
#include <vector>

class entity {

    private:
        std::vector<triangle> tris;
        int triCount;

    public:
        entity();
        entity(std::vector<triangle>);
        entity(entity&);
        __host__ __device__ triangle& operator[](const int index);

        __host__ __device__ void translateEntity(vec4);
        __host__ __device__ void rotateEntityX(float);
        __host__ __device__ void scaleEntity(vec4);
        __host__ __device__ void rotateEntityY(float);
        __host__ __device__ void rotateEntityZ(float);

        __host__ __device__ void setTriCount(int);
        __host__ __device__ int getTriCount();
        __host__ void loadObj(std::string);

        __host__ __device__ triangle* getTriangles();




};
__global__ void scaleK(vec4 inputVec, triangle*, int);
__global__ void translationK(vec4 inputVec, triangle*, int);
__global__ void rotationXK(float radians,  triangle* , int );
__global__ void rotationYK(float radians, triangle*, int);



#endif