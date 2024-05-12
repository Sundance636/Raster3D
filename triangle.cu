#include "triangle.h"

__host__ __device__ triangle::triangle() {

}
__host__ __device__ triangle::triangle(vec4 p1, vec4 p2, vec4 p3) {
    this->point1 = p1;
    this->point2 = p2;
    this->point3 = p3;
}

__host__ __device__ vec4 triangle::getP1() {
    return this->point1;
}
__host__ __device__ vec4 triangle::getP2() {
    return this->point2;
}
__host__ __device__ vec4 triangle::getP3() {
    return this->point3;
}
__host__ __device__ void triangle::setP1(vec4 p1) {
    this->point1 = p1;
}
__host__ __device__ void triangle::setP2(vec4 p2) {
    this->point2 = p2;
}
__host__ __device__ void triangle::setP3(vec4 p3) {
    this->point3 = p3;
}

__host__ __device__ void triangle::translate(vec4 input) {
    
    translation(input,this->point1);
    translation(input,this->point2);
    translation(input,this->point3);
}

__host__ __device__ void triangle::triscale(vec4 input) {
    scale(input,this->point1);
    scale(input,this->point2);
    scale(input,this->point3);

}

__host__ __device__ void triangle::rotateX(float angle) {
    rotationX(angle, this->point1);
    rotationX(angle, this->point2);
    rotationX(angle, this->point3);

}
__host__ __device__ void triangle::rotateY(float angle) {
    rotationY(angle, this->point1);
    rotationY(angle, this->point2);
    rotationY(angle, this->point3);

}

__host__ __device__ void triangle::rotateZ(float angle) {
    rotationZ(angle, this->point1);
    rotationZ(angle, this->point2);
    rotationZ(angle, this->point3);

}