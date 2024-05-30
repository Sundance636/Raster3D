#ifndef __vec4_h__
#define __vec4_h__

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);


class vec4 {
    private:
        float e[4];


    public:
        __host__ __device__ vec4();
        __host__ __device__ vec4(float,float,float,float);

        __host__ __device__ float x() const;
        __host__ __device__ float y() const;
        __host__ __device__ float z() const;
        __host__ __device__ float w() const;

        __host__ __device__ void setx(float);
        __host__ __device__ void sety(float);
        __host__ __device__ void setz(float);
        __host__ __device__ void setw(float);

        __host__ __device__ vec4 operator=(const vec4 &otherVector);
        __host__ __device__ vec4 operator+(const vec4 &vector);//vector addition
        __host__ __device__ vec4 operator-(const vec4 &vector);//vector subtraction
        __host__ __device__ float& operator[](const int index);
        __host__ std::ostream& operator<<(std::ostream& out);

        //__host__ __device__ vec4 operator*( const vec4 &vector, float scalar);
        //__host__ __device__ vec4 operator*(float);//define scaling vectors

        __host__ __device__ float magnitude();
        __host__ __device__ vec4 normalize();
        
        
};

//vector overload declarations
__host__ __device__ vec4 operator*(const vec4&, float);
__host__ __device__ vec4 operator*(float, const vec4&);
__host__ __device__ vec4 operator/(const vec4&, float);
__host__ __device__ vec4 operator/(float, const vec4&);
__host__ __device__ vec4 operator+(const vec4&, const vec4&);
__host__ __device__ vec4 operator-(const vec4&, const vec4&);
__host__ std::ostream& operator<<(std::ostream& out, const vec4& o);


//Vector operations
__host__ __device__ vec4 unit_vector4(vec4);
__host__ __device__ float dot_product4(vec4,vec4);
__host__ __device__ vec4 cross_product4(vec4,vec4);//3d use case




#endif