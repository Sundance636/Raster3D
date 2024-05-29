#include "triangle.h"

__host__ __device__ triangle::triangle() {

}
__host__ __device__ triangle::triangle(vec4 p1, vec4 p2, vec4 p3) {
    this->point1 = p1;
    this->point2 = p2;
    this->point3 = p3;

    calculateSurfaceNormal();
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

__host__ __device__ vec4 triangle::getSurfaceNormal() {
    return normal;
}


__host__ __device__ void triangle::translate(vec4 input) {
    
    translation(input,this->point1);
    translation(input,this->point2);
    translation(input,this->point3);
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__host__ void triangle::triscale(vec4 input) {
    //allocate memory/ transfer point to gpu in array/vec

    vec4 h_points[3] = {this->point1, this->point2, this->point3}; 
    vec4* d_points = nullptr;

    //std::cout << d_points << "\n";

    //allocate and transfer points to the GPU
    checkCudaErrors(cudaMallocManaged((void**)&d_points, 3 * sizeof(vec4)));
    d_points[0] = this->point1;
    d_points[1] = this->point2;
    d_points[2] = this->point3;

    //std::cout << "Vec4 size: " << sizeof(vec4) << "\n";

    //cudaMemcpy(d_points,h_points, 3 * sizeof(vec4), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int numBlocks = (3 + blockSize - 1) / blockSize;


    //std::cout << "Block size: " << blockSize << "\n";
    //std::cout << "numBlock: " << numBlocks << "\n";

    //std::cout << "scale input: " << input << "\n";
    //std::cout << d_points << "\n";
    //std::cout << d_points[1] << "\n";

    //make kernel call
    scaleK<<<numBlocks, blockSize>>>(input, d_points, 3);
    //testK<<<numBlocks, blockSize>>>(d_points);
    //std::cout << d_points[1] << "\n";
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    

    //transfer the results back
    //cudaMemcpy(h_points, d_points, 3 * sizeof(vec4), cudaMemcpyDeviceToHost);

    this->point1 = d_points[0];
    this->point2 = d_points[1];
    this->point3 = d_points[2];

    checkCudaErrors(cudaFree(d_points));
    
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
__host__ __device__ void triangle::calculateSurfaceNormal() {
    vec4 U = vec4(point2) - vec4(point1);
    vec4 V = vec4(point3) - vec4(point1);
    normal = cross_product4(U,V);

    normal.setw(0); //direction not point

    normal = unit_vector4(normal);
}
