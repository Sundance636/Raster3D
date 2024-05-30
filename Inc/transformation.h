#ifndef __transformation_h__
#define __transformation_h__

#include "triangle.h"
#include "vec4.h"

//Affine Transformations
__host__ __device__ vec4 translation(vec4,vec4&);

__host__ __device__ vec4 rotationX(float rads, vec4&);
__host__ __device__ vec4 rotationY(float rads, vec4&);
__host__ __device__ vec4 rotationZ(float rads, vec4&);

__host__ __device__ vec4 shearX(float rads, vec4&);
__host__ __device__ vec4 shearY(float rads, vec4&);
__host__ __device__ vec4 shearZ(float rads, vec4&);

__host__ __device__ vec4 scale(vec4, vec4&);
__global__ void testK(vec4* point);


//__host__ __device__ vec4 perspectiveProjection();

//Orthogonal Transformations
__host__ __device__ vec4 orthoProjection();

//Both




#endif