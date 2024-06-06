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

__host__ __device__ void triangle::setColour(u_int32_t A, u_int32_t R, u_int32_t G, u_int32_t B) {
    this->colour = (A << 24) | ((R << 16 )) | ((G << 8)) | B;
    //this->colour = R << 16;
}

__host__ __device__ u_int32_t triangle::getColour() {
    return colour;
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

__host__ __device__ bool triangle::hitTest(float boxMinX, float boxMaxX, float boxMinY, float boxMaxY, int WIDTH, int HEIGHT,u_int32_t* frameBuffer, float* depthBuffer, float facingRatio) {
    int xMin = std::max(0, std::min(WIDTH - 1, (int)std::floor(boxMinX)));
    int yMin = std::max(0, std::min(HEIGHT - 1, (int)std::floor(boxMinY)));
    int xMax = std::max(0, std::min(WIDTH - 1, (int)std::floor(boxMaxX)));
    int yMax = std::max(0, std::min(HEIGHT - 1, (int)std::floor(boxMaxY)));


    //looping through the bounding box for given triangle
    for (int y = yMin; y <= yMax; ++y) {
        for (int x = xMin; x <= xMax; ++x) {
            float w0 = edgeFunction(this->point2, this->point3, vec4(x, y,0, 0));
            float w1 = edgeFunction(this->point3, this->point1, vec4(x, y,0, 0));
            float w2 = edgeFunction(this->point1, this->point2, vec4(x, y,0, 0));


            if (pixelInTri(x, y)) {


                //normalize depth buffer values
                float depth = (w0 * this->point1.z() + w1 * this->point2.z() + w2 * this->point3.z())/(w0 + w1 + w2);
                setPixel(x,y,depth, WIDTH, HEIGHT, frameBuffer,depthBuffer, facingRatio);
            }
        }
    }

    return true;
}

__host__ __device__ bool triangle::pixelInTri(int screenX, int screenY) {
    float w0 = edgeFunction(this->point2, this->point3, vec4(screenX, screenY,0, 0));
    float w1 = edgeFunction(this->point3, this->point1, vec4(screenX, screenY,0, 0));
    float w2 = edgeFunction(this->point1, this->point2, vec4(screenX, screenY,0, 0));


    return w0 >= 0 && w1 >= 0 && w2 >= 0; 
}

__host__ __device__ float edgeFunction( const vec4 a, const vec4 b, const vec4 c) {
    return (vec4(c).x() - a.x()) * (vec4(b).y() - a.y()) - (vec4(c).y() - a.y()) * (vec4(b).x() - a.x());
}

__host__ __device__ void triangle::setPixel(int screenX, int screenY, float depth, int WIDTH,int HEIGHT, u_int32_t* frameBuffer, float* depthBuffer, float facingRatio ) {
    if (screenX >= 0 && screenX < WIDTH && screenY >= 0 && screenY < HEIGHT) {
            int index = screenY * WIDTH + screenX;
            if (depth < depthBuffer[index]) {
                uint32_t col = this->colour;

                uint8_t A = (col >> 24) & 0xFF;
                uint8_t R = (col >> 16) & 0xFF;
                uint8_t G = (col >> 8) & 0xFF;
                uint8_t B = col & 0xFF;

                R = (uint8_t)(R * facingRatio);
                G = (uint8_t)(G * facingRatio);
                B = (uint8_t)(B * facingRatio);
                A = (uint8_t)(255 * facingRatio);


                uint32_t color = (A << 24) | (R << 16) | (G << 8) | B;
                
                frameBuffer[index] = color;
                depthBuffer[index] = depth;
            }
        }
}