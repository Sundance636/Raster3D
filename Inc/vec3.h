#ifndef __vec3_h__
#define __vec3_h__
#include "vec4.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

class vec3 : vec4 {
    private:
        float e[3];


    public:
        __host__ __device__ vec3();
        __host__ __device__ vec3(float,float,float);
        

        __host__ __device__ float x() const;
        __host__ __device__ float y() const;
        __host__ __device__ float z() const;

        __host__ __device__ vec3 operator=(const vec3 &otherVector);
        __host__ __device__ vec3 operator+(const vec3 &vector);//vector addition
        __host__ __device__ vec3 operator-(const vec3 &vector);//vector subtraction
        //__host__ __device__ vec3 operator*( const vec3 &vector, float scalar);
        //__host__ __device__ vec3 operator*(float);//define scaling vectors

        __host__ __device__ float magnitude();
        __host__ __device__ vec3 normalize();
        
};

//vector overload declarations
__host__ __device__ vec3 operator*(const vec3&, float);
__host__ __device__ vec3 operator*(float, const vec3&);
__host__ __device__ vec3 operator/(const vec3&, float);
__host__ __device__ vec3 operator/(float, const vec3&);
__host__ __device__ vec3 operator+(const vec3&, const vec3&);

//Vector operations
__device__ vec3 unit_vector(vec3);
__device__ float dot_product(vec3,vec3);
__device__ vec3 cross_product(vec3,vec3);




#endif