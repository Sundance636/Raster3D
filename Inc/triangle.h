#ifndef __triangle_h__
#define __triangle_h__

#include "vec4.h"
#include "transformation.h"
#include <vector>


class triangle {
    private:
        vec4 point1;//reference points in clockwise orientation
        vec4 point2;
        vec4 point3;

        vec4 normal;

        //RBG bitshift
        u_int32_t colour;

    public:
        __host__ __device__ triangle();
        __host__ __device__ triangle(vec4, vec4, vec4);

        __host__ __device__ vec4 getP1();
        __host__ __device__ vec4 getP2();
        __host__ __device__ vec4 getP3();
        __host__ __device__ void setP1(vec4);
        __host__ __device__ void setP2(vec4);
        __host__ __device__ void setP3(vec4);
        __host__ __device__ vec4 getSurfaceNormal();
        __host__ __device__ void setColour(u_int32_t A,u_int32_t R,u_int32_t G,u_int32_t B);
        __host__ __device__ u_int32_t getColour();


        __host__ __device__ void translate(vec4);
        __host__ __device__ void rotateX(float);
        __host__ __device__ void rotateY(float);
        __host__ __device__ void rotateZ(float);
        __host__ __device__ void calculateSurfaceNormal();

        __host__ __device__ bool hitTest(float, float,float,float, int, int,u_int32_t* frameBuffer, float* depthBuffer, float facingRatio);
        __host__ __device__ bool pixelInTri(int screenX, int screenY);
        __host__ __device__ void setPixel(int screenX, int screenY, float depth, int WIDTH,int HEIGHT,u_int32_t* frameBuffer, float* depthBuffer, float facingRatio);

};

__host__ __device__ float edgeFunction( const vec4 a, const vec4 b, const vec4 c);


#endif