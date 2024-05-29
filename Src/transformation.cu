#include "transformation.h"

__host__ __device__ vec4 translation(vec4 inputVec, vec4 &point) {
    vec4 TranslationMat[] = { vec4(1.0f,0.0f,0.0f,inputVec.x()),//init transl matrix
                            vec4(0.0f,1.0f,0.0f,inputVec.y()),
                            vec4(0.0f,0.0f,1.0f,inputVec.z()),
                            vec4(0.0f,0.0f,0.0f,inputVec.w()) };


    vec4 newVec = vec4(dot_product4(TranslationMat[0], point),
                dot_product4(TranslationMat[1], point),
                dot_product4(TranslationMat[2], point),
                dot_product4(TranslationMat[3], point));
    
    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());



    return newVec;
}

__host__ __device__ vec4 scale(vec4 inputVec, vec4 &point) {
    vec4 ScaleMat[] = { vec4(inputVec.x(),0.0f,0.0f,0.0f),//init transl matrix
                            vec4(0.0f,inputVec.y(),0.0f,0.0f),
                            vec4(0.0f,0.0f,inputVec.z(), 0.0f),
                            vec4(0.0f,0.0f,0.0f,1.0f) };


    vec4 newVec = vec4(dot_product4(ScaleMat[0], point),
                dot_product4(ScaleMat[1], point),
                dot_product4(ScaleMat[2], point),
                dot_product4(ScaleMat[3], point));
    
    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());



    return newVec;
}

__global__ void scaleK(vec4 inputVec, vec4* point, int numOfPoints) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //point .setx(20);
    
    if(idx < numOfPoints) { //three points per triangle


        vec4 ScaleMat[] = { vec4(inputVec.x(),0.0f,0.0f,0.0f),//init transl matrix
                                vec4(0.0f,inputVec.y(),0.0f,0.0f),
                                vec4(0.0f,0.0f,inputVec.z(), 0.0f),
                                vec4(0.0f,0.0f,0.0f,1.0f) };

        vec4 currPoint = point[idx];


        vec4 newVec = vec4(dot_product4(ScaleMat[0], currPoint),
                    dot_product4(ScaleMat[1], currPoint),
                    dot_product4(ScaleMat[2], currPoint),
                    dot_product4(ScaleMat[3], currPoint));
        
        point[idx].setx(newVec.x());
        point[idx].sety(newVec.y());
        point[idx].setz(newVec.z());
        point[idx].setw(newVec.w());

        }
        point[1].setx(20);
}
__global__ void testK(vec4* point) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //point .setx(20);
    
    if(idx < 3) { //three points per triangle


        point[idx] = vec4(250,250,250,250);
        }

}


__host__ __device__ vec4 rotationX(float radians, vec4 &point) {
    vec4 RotationMatX[] = {vec4(1.0f, 0.0f, 0.0f, 0.0f),//init rot matrix
                            vec4(0.0f, cos(radians), -sin(radians), 0.0f),
                            vec4(0.0f,sin(radians), cos(radians),0.0f),
                            vec4(0.0f,0.0f,0.0f, 1.0f)};

    vec4 newVec = vec4(dot_product4(RotationMatX[0], point),
                dot_product4(RotationMatX[1], point),
                dot_product4(RotationMatX[2], point),
                dot_product4(RotationMatX[3], point));
    
    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    return newVec;
}

__host__ __device__ vec4 rotationY(float radians, vec4 &point) {
    vec4 RotationMatY[] = {vec4(cos(radians), 0.0f, sin(radians), 0.0f),//init rot matrix
                            vec4(0.0f, 1.0f, 0.0f, 0.0f),
                            vec4(-sin(radians),0.0f,cos(radians),0.0f),
                            vec4(0.0f,0.0f,0.0f, 1.0f)};

    vec4 newVec = vec4(dot_product4(RotationMatY[0], point),
                dot_product4(RotationMatY[1], point),
                dot_product4(RotationMatY[2], point),
                dot_product4(RotationMatY[3], point));
    
    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    return newVec;
}

__host__ __device__ vec4 rotationZ(float radians, vec4 &point) {
    vec4 RotationMatZ[] = {vec4(cos(radians), -sin(radians), 0.0f, 0.0f),//init rot matrix
                            vec4(sin(radians), cos(radians), 0.0f, 0.0f),
                            vec4(0.0f,0.0f,1.0f,0.0f),
                            vec4(0.0f,0.0f,0.0f, 1.0f)};

    vec4 newVec = vec4(dot_product4(RotationMatZ[0], point),
                dot_product4(RotationMatZ[1], point),
                dot_product4(RotationMatZ[2], point),
                dot_product4(RotationMatZ[3], point));
    
    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    return newVec;
}

__host__ __device__ vec4 shearX(float radians, vec4 &point) {
    vec4 ShearMatX[] = {vec4(1.0f, 0.0f, 0.0f, 0.0f),//init rot matrix
                            vec4(0.0f, cos(radians), -sin(radians), 0.0f),
                            vec4(0.0f,sin(radians), cos(radians),0.0f),
                            vec4(0.0f,0.0f,0.0f, 1.0f)};

    vec4 newVec = vec4(dot_product4(ShearMatX[0], point),
                dot_product4(ShearMatX[1], point),
                dot_product4(ShearMatX[2], point),
                dot_product4(ShearMatX[3], point));
    
    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    return newVec;
}

__host__ __device__ vec4 shearY(float radians, vec4 &point) {
    vec4 RotationMatY[] = {vec4(cos(radians), 0.0f, sin(radians), 0.0f),//init rot matrix
                            vec4(0.0f, 1.0f, 0.0f, 0.0f),
                            vec4(-sin(radians),0.0f,cos(radians),0.0f),
                            vec4(0.0f,0.0f,0.0f, 1.0f)};

    vec4 newVec = vec4(dot_product4(RotationMatY[0], point),
                dot_product4(RotationMatY[1], point),
                dot_product4(RotationMatY[2], point),
                dot_product4(RotationMatY[3], point));
    
    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    return newVec;
}

__host__ __device__ vec4 shearZ(float radians, vec4 &point) {
    vec4 RotationMatZ[] = {vec4(cos(radians), -sin(radians), 0.0f, 0.0f),//init rot matrix
                            vec4(sin(radians), cos(radians), 0.0f, 0.0f),
                            vec4(0.0f,0.0f,1.0f,0.0f),
                            vec4(0.0f,0.0f,0.0f, 1.0f)};

    vec4 newVec = vec4(dot_product4(RotationMatZ[0], point),
                dot_product4(RotationMatZ[1], point),
                dot_product4(RotationMatZ[2], point),
                dot_product4(RotationMatZ[3], point));
    
    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    return newVec;
}