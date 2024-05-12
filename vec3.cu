#include "vec3.h"

__host__ __device__ vec3::vec3() {

}

__host__ __device__ vec3::vec3(float e0,float e1,float e2) {
    this->e[0] = e0;
    this->e[1] = e1;
    this->e[2] = e2;
}

__host__ __device__ float vec3::x() const {
    return this->e[0];
}

__host__ __device__ float vec3::y() const {
    return this->e[1];

}

__host__ __device__ float vec3::z() const {
    return this->e[2];

}

__host__ __device__ vec3 vec3::operator=(const vec3 &otherVector) {
    this->e[0] = otherVector.e[0];
    this->e[1] = otherVector.e[1];
    this->e[2] = otherVector.e[2];


    return *this;
}

__host__ __device__ vec3 vec3::operator+(const vec3 &vector) {
    this->e[0] = this->e[0] + vector.e[0];
    this->e[1] = this->e[1] + vector.e[1];
    this->e[2] = this->e[2] + vector.e[2];
    return *this;
}

__host__ __device__ vec3 vec3::operator-(const vec3 &vector) {
    this->e[0] = this->e[0] - vector.e[0];
    this->e[1] = this->e[1] - vector.e[1];
    this->e[2] = this->e[2] - vector.e[2];
    return *this;
}

__host__ __device__ vec3 operator*(const vec3 &v, float t) {
    return vec3(t* v.x(), t*v.y(), t*v.z());
}

__host__ __device__ vec3 operator*(float t, const vec3 &v) {
    return vec3(t* v.x(), t*v.y(), t*v.z());
}

__host__ __device__ vec3 operator/(const vec3 &v, float t) {
    return vec3(v.x()/t, v.y()/t, v.z()/t);
}

__host__ __device__ vec3 operator/(float t, const vec3 &v) {
    return vec3(v.x()/t, v.y()/t, v.z()/t);
}

__host__ __device__ vec3 operator+(const vec3 &Vector1, const vec3 &Vector2) {
    return vec3( Vector1.x() + Vector2.x(), Vector1.y() + Vector2.y(), Vector1.z() + Vector2.z());
}

__host__ __device__ float vec3::magnitude() {
    return sqrtf( (this->e[0] * this->e[0]) + (this->e[1]*this->e[1]) + (this->e[2]*this->e[2]) );
}

__device__ vec3 unit_vector(vec3 vector) {
    return vector / vector.magnitude(); //normalize the vector
}

__device__ float dot_product(vec3 vec1, vec3 vec2) {
    return (vec1.x() * vec2.x()) + (vec1.y() * vec2.y()) + (vec1.z() * vec2.z());
}

