#include "entity.h"

entity::entity() {

}

entity::entity(triangle* tris) {
    this->tris = tris;
}

entity::entity(entity &copy) {
    this->triCount = copy.getTriCount();
    //triangle newTris[triCount];
    this->tris = new triangle[triCount];

    for(int i = 0; i < this->triCount; i++ ) {
        this->tris[i] = copy[i];
    }
}

__host__ __device__ triangle& entity::operator[](const int index) {
    return this->tris[index];
}

__host__ __device__ void entity::translateEntity(vec4 input) {

    //vectorize later for cuda
    for(int i = 0; i < triCount; i++ ) {
        
        //translate the three points in each tri;
        (this->tris[i]).translate(input);

    }

}

__host__ __device__ void entity::scaleEntity(vec4 scaleFactor) {

    //vectorize later for cuda
    for(int i = 0; i < triCount; i++ ) {
        
        //translate the three points in each tri;
        (this->tris[i]).triscale(scaleFactor);
        //(this->tris[i]).setP1((this->tris[i].getP1() * scaleFactor));
        //(this->tris[i]).setP2((this->tris[i].getP2() * scaleFactor));
        //(this->tris[i]).setP3((this->tris[i].getP3() * scaleFactor));

    }

}

__host__ __device__ void entity::rotateEntityX(float angle) {
    //vectorize later for cuda
    for(int i = 0; i < triCount; i++ ) {
        
        //translate the three points in each tri;
        (this->tris[i]).rotateX(angle);

    }
}

__host__ __device__ void entity::rotateEntityY(float angle) {
    //vectorize later for cuda
    for(int i = 0; i < triCount; i++ ) {
        
        //translate the three points in each tri;
        (this->tris[i]).rotateY(angle);

    }
}

__host__ __device__ void entity::rotateEntityZ(float angle) {
    //vectorize later for cuda
    for(int i = 0; i < triCount; i++ ) {
        
        //translate the three points in each tri;
        (this->tris[i]).rotateZ(angle);

    }
}

__host__ __device__ void entity::setTriCount(int count) {
    this->triCount = count;
}

__host__ __device__ int entity::getTriCount() {
    return this->triCount;
}
