#ifndef __triangle_h__
#define __triangle_h__

#include "vec4.h"
#include "transformation.h"

class triangle {
    private:
        vec4 point1;//reference points in clockwise orientation
        vec4 point2;
        vec4 point3;

        vec4 normal;

        u_int colour;

    public:
        __host__ __device__ triangle();
        __host__ __device__ triangle(vec4, vec4, vec4);

        __host__ __device__ vec4 getP1();
        __host__ __device__ vec4 getP2();
        __host__ __device__ vec4 getP3();
        __host__ __device__ void setP1(vec4);
        __host__ __device__ void setP2(vec4);
        __host__ __device__ void setP3(vec4);

        __host__ __device__ void translate(vec4);
        __host__ __device__ void triscale(vec4);
        __host__ __device__ void rotateX(float);
        __host__ __device__ void rotateY(float);
        __host__ __device__ void rotateZ(float);


};


#endif