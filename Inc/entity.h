#ifndef __entity_h__
#define __entity_h__

#include "triangle.h"

#include <fstream>
#include <sstream>

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
        __host__ void depthTest(int WIDTH,int HEIGHT,int &count, u_int32_t* frameBuffer, float* depthBuffer,std::vector<float> facingRatios);




};
__global__ void scaleK(vec4 inputVec, triangle*, int);
__global__ void translationK(vec4 inputVec, triangle*, int);
__global__ void rotationXK(float radians,  triangle* , int );
__global__ void rotationYK(float radians, triangle*, int);
__global__ void cullingK( vec4, triangle*, float*,int);
__global__ void frustumCullingK(float vertFOV, float horiFOV,float nearPlane,float farPlane,triangle* d_tris, float* d_facenorm, int object);
__global__ void hitTestK(int WIDTH,int HEIGHT,triangle* d_tris, float* d_facenorm, u_int32_t* d_frameBuffer,float*  d_depthBuffer,u_int32_t* count,int numOfTris);


#endif