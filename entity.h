#ifndef __entity_h__
#define __entity_h__

#include "triangle.h"


class entity {

    private:
        triangle* tris;
        int triCount;

    public:
        entity();
        entity(triangle*);
        entity(entity&);
        __host__ __device__ triangle& operator[](const int index);

        __host__ __device__ void translateEntity(vec4);
        __host__ __device__ void rotateEntityX(float);
        __host__ __device__ void scaleEntity(vec4);
        __host__ __device__ void rotateEntityY(float);
        __host__ __device__ void rotateEntityZ(float);

        __host__ __device__ void setTriCount(int);
        __host__ __device__ int getTriCount();



};



#endif