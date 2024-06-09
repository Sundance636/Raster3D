#include "entity.h"

entity::entity() {
    this->triCount = tris.size();

}

entity::entity(std::vector<triangle> tris) {
    this->tris = tris;
    this->triCount = tris.size();
}

entity::entity(entity &copy) {
    this->triCount = copy.getTriCount();
    //triangle newTris[triCount];
    //std::vector<triangle> newVec;
    //this->tris = newVec;

    for(int i = 0; i < this->triCount; i++ ) {
        this->tris.push_back( copy[i]);
    }
}

__host__ __device__ triangle& entity::operator[](const int index) {
    return this->tris[index];
}

__host__ __device__ triangle* entity::getTriangles() {
    return this->tris.data();
}


__host__ __device__ void entity::translateEntity(vec4 input) {
    triangle* trisArray = &(this->tris[0]);//pass vec as an array
    triangle* d_tris;

    checkCudaErrors(cudaMalloc((void**)&d_tris, getTriCount() * sizeof(triangle)));
    checkCudaErrors(cudaMemcpy(d_tris,trisArray, getTriCount() * sizeof(triangle), cudaMemcpyHostToDevice));


    //ENSURE THESE TWO NUMBERS ARE OPTIMAL
    int blockSize = 256;
    int numBlocks = (triCount + blockSize - 1) / blockSize;

    translationK<<<numBlocks, blockSize>>>(input, d_tris, getTriCount());
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    //copy back
    checkCudaErrors(cudaMemcpy(trisArray,d_tris, getTriCount() * sizeof(triangle), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tris));

}

__host__ __device__ void entity::scaleEntity(vec4 scaleFactor) {
    triangle* trisArray = &(this->tris[0]);//pass vec as an array
    triangle* d_tris;

    checkCudaErrors(cudaMalloc((void**)&d_tris, getTriCount() * sizeof(triangle)));
    checkCudaErrors(cudaMemcpy(d_tris,trisArray, getTriCount() * sizeof(triangle), cudaMemcpyHostToDevice));


    //ENSURE THESE TWO NUMBERS ARE OPTIMAL
    int blockSize = 256;
    int numBlocks = (triCount + blockSize - 1) / blockSize;

    scaleK<<<numBlocks, blockSize>>>(scaleFactor, d_tris, getTriCount());
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    //copy back
    checkCudaErrors(cudaMemcpy(trisArray,d_tris, getTriCount() * sizeof(triangle), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tris));

}

//
// MOVE BACK LATER
__global__ void scaleK(vec4 inputVec, triangle* tri, int numOfTris) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(idx < numOfTris) { //three points per triangle


        vec4 ScaleMat[] = { vec4(inputVec.x(),0.0f,0.0f,0.0f),//init transl matrix
                                vec4(0.0f,inputVec.y(),0.0f,0.0f),
                                vec4(0.0f,0.0f,inputVec.z(), 0.0f),
                                vec4(0.0f,0.0f,0.0f,1.0f) };

        vec4 points[3] = {tri[idx].getP1(), tri[idx].getP2(), tri[idx].getP3()};


        vec4 newVec = vec4( dot_product4(ScaleMat[0], points[0]),
                            dot_product4(ScaleMat[1], points[0]),
                            dot_product4(ScaleMat[2], points[0]),
                            dot_product4(ScaleMat[3], points[0]));
                
        tri[idx].setP1(newVec);

        newVec = vec4(  dot_product4(ScaleMat[0], points[1]),
                        dot_product4(ScaleMat[1], points[1]),
                        dot_product4(ScaleMat[2], points[1]),
                        dot_product4(ScaleMat[3], points[1]));
            
        tri[idx].setP2(newVec);


        newVec = vec4(  dot_product4(ScaleMat[0], points[2]),
                        dot_product4(ScaleMat[1], points[2]),
                        dot_product4(ScaleMat[2], points[2]),
                        dot_product4(ScaleMat[3], points[2]));
            
        tri[idx].setP3(newVec);

        }
}

//MOVE BACK AS WELL
__global__ void translationK(vec4 inputVec, triangle* tri, int numOfTris) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(idx < numOfTris) { //each triangle

        vec4 TranslationMat[] = { vec4(1.0f,0.0f,0.0f,inputVec.x()),//init transl matrix
                                vec4(0.0f,1.0f,0.0f,inputVec.y()),
                                vec4(0.0f,0.0f,1.0f,inputVec.z()),
                                vec4(0.0f,0.0f,0.0f,1.0f) };

        vec4 points[3] = {tri[idx].getP1(), tri[idx].getP2(), tri[idx].getP3()};



        vec4 newVec = vec4( dot_product4(TranslationMat[0], points[0]),
                            dot_product4(TranslationMat[1], points[0]),
                            dot_product4(TranslationMat[2], points[0]),
                            dot_product4(TranslationMat[3], points[0]));

        tri[idx].setP1(newVec);

        newVec = vec4(      dot_product4(TranslationMat[0], points[1]),
                            dot_product4(TranslationMat[1], points[1]),
                            dot_product4(TranslationMat[2], points[1]),
                            dot_product4(TranslationMat[3], points[1]));

        tri[idx].setP2(newVec);

        newVec = vec4(      dot_product4(TranslationMat[0], points[2]),
                            dot_product4(TranslationMat[1], points[2]),
                            dot_product4(TranslationMat[2], points[2]),
                            dot_product4(TranslationMat[3], points[2]));

        tri[idx].setP3(newVec);
    }
}


//FINISH REFACTORING ROTATIONS
//Define kernels that reotate all given tris by the input angle
__host__ __device__ void entity::rotateEntityX(float angle) {
    //vectorize later for cuda
    for(int i = 0; i < triCount; i++ ) {
        
        //translate the three points in each tri;
        (this->tris[i]).rotateX(angle);

    }
}

__global__ void rotationXK(float radians,  triangle* tris, int numOfTris) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(idx < numOfTris) { //each triangle

        vec4 RotationMatX[] = {vec4(1.0f, 0.0f, 0.0f, 0.0f),//init rot matrix
                                vec4(0.0f, cos(radians), -sin(radians), 0.0f),
                                vec4(0.0f,sin(radians), cos(radians),0.0f),
                                vec4(0.0f,0.0f,0.0f, 1.0f)};


        vec4 points[3] = {tris[idx].getP1(), tris[idx].getP2(), tris[idx].getP3()};


        vec4 newVec = vec4( dot_product4(RotationMatX[0], points[0]),
                            dot_product4(RotationMatX[1], points[0]),
                            dot_product4(RotationMatX[2], points[0]),
                            dot_product4(RotationMatX[3], points[0]));
        
        tris[idx].setP1(newVec);

        newVec = vec4(      dot_product4(RotationMatX[0], points[1]),
                            dot_product4(RotationMatX[1], points[1]),
                            dot_product4(RotationMatX[2], points[1]),
                            dot_product4(RotationMatX[3], points[1]));

        tris[idx].setP2(newVec);

        newVec = vec4(      dot_product4(RotationMatX[0], points[2]),
                            dot_product4(RotationMatX[1], points[2]),
                            dot_product4(RotationMatX[2], points[2]),
                            dot_product4(RotationMatX[3], points[2]));

        tris[idx].setP3(newVec);
    }

}

__host__ __device__ void entity::rotateEntityY(float angle) {

    triangle* trisArray = &(this->tris[0]);//pass vec as an array
    triangle* d_tris;

    checkCudaErrors(cudaMalloc((void**)&d_tris, getTriCount() * sizeof(triangle)));
    checkCudaErrors(cudaMemcpy(d_tris,trisArray, getTriCount() * sizeof(triangle), cudaMemcpyHostToDevice));


    //ENSURE THESE TWO NUMBERS ARE OPTIMAL
    int blockSize = 256;
    int numBlocks = (triCount + blockSize - 1) / blockSize;

    rotationYK<<<numBlocks, blockSize>>>(angle, d_tris, getTriCount());
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    //copy back
    checkCudaErrors(cudaMemcpy(trisArray,d_tris, getTriCount() * sizeof(triangle), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tris));
}

__global__ void rotationYK(float radians,  triangle* tris, int numOfTris) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(idx < numOfTris) { //each triangle
        vec4 RotationMatY[] = {vec4(cos(radians), 0.0f, sin(radians), 0.0f),//init rot matrix
                                vec4(0.0f, 1.0f, 0.0f, 0.0f),
                                vec4(-sin(radians),0.0f,cos(radians),0.0f),
                                vec4(0.0f,0.0f,0.0f, 1.0f)};

        vec4 points[3] = {tris[idx].getP1(), tris[idx].getP2(), tris[idx].getP3()};


        vec4 newVec = vec4( dot_product4(RotationMatY[0], points[0]),
                            dot_product4(RotationMatY[1], points[0]),
                            dot_product4(RotationMatY[2], points[0]),
                            dot_product4(RotationMatY[3], points[0]));
        
        tris[idx].setP1(newVec);

        newVec = vec4(      dot_product4(RotationMatY[0], points[1]),
                            dot_product4(RotationMatY[1], points[1]),
                            dot_product4(RotationMatY[2], points[1]),
                            dot_product4(RotationMatY[3], points[1]));

        tris[idx].setP2(newVec);

        newVec = vec4(      dot_product4(RotationMatY[0], points[2]),
                            dot_product4(RotationMatY[1], points[2]),
                            dot_product4(RotationMatY[2], points[2]),
                            dot_product4(RotationMatY[3], points[2]));

        tris[idx].setP3(newVec);
    }

}

__global__ void cullingK( vec4 camPosition, triangle* tris, float* facingRatios, int numOfTris) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < numOfTris) { //each triangle
        vec4 eyeLine =  vec4(camPosition) - tris[idx].getP3();//copy constructor cause my difference overload is weird
        eyeLine.sety(camPosition.y() -  -1.0f*tris[idx].getP3().y() );
        eyeLine = unit_vector4(eyeLine);
        eyeLine.setx(-eyeLine.x());
        eyeLine.setz(-eyeLine.z());
        eyeLine.setw(0);

        facingRatios[idx] = dot_product4(tris[idx].getSurfaceNormal(), eyeLine);

        

    } 
}

__global__ void frustumCullingK(float vertFOV, float horiFOV, float nearPlane,float farPlane, triangle* object, float* faceRatios, int numOfTris) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numOfTris) {
            //evaluate bound
            
            //for top
            float topBound1 = tan(vertFOV/2) * object[idx].getP1().z();// + object[i].getP1().z()/ 60.0);//margin of erro
            float topBound2 = tan(vertFOV/2) * object[idx].getP2().z();// + object[i].getP2().z()/ 60.0);//margin of erro
            float topBound3 = tan(vertFOV/2) * object[idx].getP3().z();// + object[i].getP3().z()/ 60.0);//margin of erro

            float bottomBound1 = -tan(vertFOV/2) * object[idx].getP1().z();
            float bottomBound2 = -tan(vertFOV/2) * object[idx].getP2().z();
            float bottomBound3 = -tan(vertFOV/2) * object[idx].getP3().z();

            float rightBound1 = tan(horiFOV/2) * object[idx].getP1().z();
            float rightBound2 = tan(horiFOV/2) * object[idx].getP2().z();
            float rightBound3 = tan(horiFOV/2) * object[idx].getP3().z();

            float leftBound1 = -tan(horiFOV/2) * object[idx].getP1().z();
            float leftBound2 = -tan(horiFOV/2) * object[idx].getP2().z();
            float leftBound3 = -tan(horiFOV/2) * object[idx].getP3().z();


            //change to cull only if all point are out of bounds
            if((object[idx].getP1().y() > topBound1) && (object[idx].getP2().y() > topBound2) &&  object[idx].getP3().y() > topBound3  ) {//if above frustum cull
                //std::cout << "Top Culling: " << object[i].getP1() <<  "\n";
                //std::cout << "Top Plane: " << vec4(this->top) << "\n";
            
                faceRatios[idx] = 1.0f;

            }
        else if( (object[idx].getP1().y() < bottomBound1) && (object[idx].getP2().y() < bottomBound2) &&  object[idx].getP3().y() < bottomBound3  ) {//if below frustum cull
                //std::cout << "Bottom Culling: " << object[i].getP1() <<  "\n";
                //std::cout << "Bottom Plane: " << vec4(this->bottom) << "\n";
            
                faceRatios[idx] = 1.0f;

            }
            else if((object[idx].getP1().x() > rightBound1) && (object[idx].getP2().x() > rightBound2) &&  (object[idx].getP3().x() > rightBound3) ) {//if to the right of frustum
                //std::cout << "Top Culling: " << object[i].getP1() <<  "\n";
                //std::cout << "Top Plane: " << vec4(this->top) << "\n";
            
                faceRatios[idx] = 1.0f;

            }
            else if((object[idx].getP1().x() < leftBound1) && (object[idx].getP2().x() < leftBound2) &&  (object[idx].getP3().x() < leftBound3)) {//if left of frustum cull
                //std::cout << "Bottom Culling: " << object[i].getP1() <<  "\n";
                //std::cout << "Bottom Plane: " << vec4(this->bottom) << "\n";
            
                faceRatios[idx] = 1.0f;

            }
            else if(max( max(object[idx].getP1().z(),
        (object[idx].getP2().z())),
        (object[idx].getP3().z() )) < nearPlane) {//if too close to frustum
                //std::cout << "Near Culling: " << object[i].getP1() <<  "\n";
                //std::cout << "Near Plane: " << vec4(this->near) << "\n";
            
                faceRatios[idx] = 1.0f;

            }

            
            if( min(min(object[idx].getP1().z(),
            (object[idx].getP2().z())),
            (object[idx].getP3() ).z())  > farPlane) {//if too far from frustum
                //std::cout << "Near Culling: " << object[i].getP1() <<  "\n";
                //std::cout << "Near Plane: " << vec4(this->near) << "\n";
            
                //faceRatios[idx] = 1.0f;

            }
            
            


        }

}

__host__ void entity::depthTest(int WIDTH,int HEIGHT,int &count ,u_int32_t* frameBuffer, float* depthBuffer,std::vector<float> faceRatios) {

    triangle* trisArray = this->getTriangles();//pass vec as an array
    float* faceArray = faceRatios.data();

    triangle* d_tris = nullptr;
    float* d_facenorm = nullptr;
    u_int32_t* d_frameBuffer = nullptr;
    float* d_depthBuffer = nullptr;
    u_int32_t* d_count = 0;

    checkCudaErrors(cudaMalloc((void**)&d_tris, this->getTriCount() * sizeof(triangle)));
    checkCudaErrors(cudaMemcpy(d_tris,trisArray, this->getTriCount() * sizeof(triangle), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&d_facenorm, this->getTriCount() * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_facenorm,faceArray, this->getTriCount() * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&d_frameBuffer, (WIDTH * HEIGHT) * sizeof(u_int32_t)));
    checkCudaErrors(cudaMemcpy(d_frameBuffer,frameBuffer, (WIDTH * HEIGHT) * sizeof(u_int32_t), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&d_depthBuffer, WIDTH * HEIGHT * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_depthBuffer,depthBuffer, WIDTH * HEIGHT * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&d_count, sizeof(u_int32_t)));

    //ENSURE THESE TWO NUMBERS ARE OPTIMAL
    int blockSize = 256;
    int numBlocks = (this->getTriCount() + blockSize - 1) / blockSize;
    

    hitTestK<<<numBlocks, blockSize>>>(WIDTH,HEIGHT,d_tris,d_facenorm,d_frameBuffer, d_depthBuffer,d_count,this->getTriCount());
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    
    checkCudaErrors(cudaMemcpy(faceArray, d_facenorm, this->getTriCount() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(frameBuffer,d_frameBuffer, (WIDTH * HEIGHT) * sizeof(u_int32_t), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(depthBuffer,d_depthBuffer, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&count,d_count, sizeof(u_int32_t), cudaMemcpyDeviceToHost));


    checkCudaErrors(cudaFree(d_facenorm));
    checkCudaErrors(cudaFree(d_tris));
    checkCudaErrors(cudaFree(d_frameBuffer));
    checkCudaErrors(cudaFree(d_depthBuffer));
    checkCudaErrors(cudaFree(d_count));


    d_tris = nullptr;
    d_facenorm = nullptr;
    d_depthBuffer = nullptr;
    d_frameBuffer = nullptr;
    d_count = nullptr;
}
__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
        float old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

        return old;
}

__global__ void hitTestK(int WIDTH,int HEIGHT,triangle* d_tris, float* d_facenorm, u_int32_t* d_frameBuffer,float*  d_depthBuffer,u_int32_t* d_count, int numOfTris) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numOfTris && d_facenorm[idx] < 0.0f) {
        //(*d_count)++;

        //atomicInc(d_count,*d_count);

            //get bounding box for current triangle
            float boxMinX = min(min(d_tris[idx].getP1().x(),d_tris[idx].getP2().x()),d_tris[idx].getP3().x());
            float boxMaxX = max(max(d_tris[idx].getP1().x(),d_tris[idx].getP2().x()),d_tris[idx].getP3().x());

            float boxMinY = min(min(d_tris[idx].getP1().y(),d_tris[idx].getP2().y()),d_tris[idx].getP3().y());
            float boxMaxY = max(max(d_tris[idx].getP1().y(),d_tris[idx].getP2().y()),d_tris[idx].getP3().y());
    
        // Clamp bounding box to screen dimensions if too big
        int xMin = max(0, min(WIDTH - 1, (int)floor(boxMinX)));
        int yMin = max(0, min(HEIGHT - 1, (int)floor(boxMinY)));
        int xMax = max(0, min(WIDTH - 1, (int)floor(boxMaxX)));
        int yMax = max(0, min(HEIGHT - 1, (int)floor(boxMaxY)));
        

        // Parallelize pixel processing within bounding box
        for (int y = yMin; y <= yMax; y ++) {
            for (int x = xMin; x <= xMax; x ++) {
                float w0 = edgeFunction(d_tris[idx].getP2(), d_tris[idx].getP3(), vec4(x, y, 0.0f, 0.0f));
                float w1 = edgeFunction(d_tris[idx].getP3(), d_tris[idx].getP1(), vec4(x, y, 0.0f, 0.0f));
                float w2 = edgeFunction(d_tris[idx].getP1(), d_tris[idx].getP2(), vec4(x, y, 0.0f, 0.0f));

                if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
                    // Normalize depth buffer values
                    float depth = (w0 * d_tris[idx].getP1().z() + w1 * d_tris[idx].getP2().z() + w2 * d_tris[idx].getP3().z()) / (w0 + w1 + w2);
                    int index = y * WIDTH + x;

                    // Atomic operation to ensure correct depth buffering
                    //fatomicMin(&d_depthBuffer[index], (int)depth);
                    
                    // If the depth buffer was updated, set the pixel color
                    if (d_depthBuffer[index] > depth) {
                        d_depthBuffer[index] = depth;

                        uint32_t col = d_tris[idx].getColour();
                        uint8_t A = (col >> 24) & 0xFF;
                        uint8_t R = (col >> 16) & 0xFF;
                        uint8_t G = (col >> 8) & 0xFF;
                        uint8_t B = col & 0xFF;

                        R = (uint8_t)(R * -d_facenorm[idx]);
                        G = (uint8_t)(G * -d_facenorm[idx]);
                        B = (uint8_t)(B * -d_facenorm[idx]);
                        A = (uint8_t)(255 * -d_facenorm[idx]);

                        uint32_t color = (A << 24) | (R << 16) | (G << 8) | B;
                        d_frameBuffer[index] = color;
                    }
                }
            }
        }
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

__host__ void entity::loadObj(std::string file) {

    std::ifstream objFile(file);
    
    if(!objFile.is_open()) {
        return;
    }

    std::vector<vec4> vertices;
    std::vector<triangle> triangles;

    while(!objFile.eof()) {

        char line[128];//lets hope there aren't more than 180s chars on a line
        objFile.getline(line, 128);

        std::basic_stringstream<char> stream;
        


        stream << line;

        char  discard;

        if (line[0] == 'v') {

            
            float x;
            float y;
            float z;

            stream >> discard >> x >> y >> z;
            vec4 point(x,y,z,1.0f);
            vertices.push_back(point);
        }

        if(line[0] == 'f') {
            int face[3];
            stream >> discard >> face[0];
                        stream.ignore(128, ' ');

            stream >> face[1];
                        stream.ignore(128, ' ');
            stream >> face[2];
            //stream.ignore(128, ' ');
            triangles.push_back(triangle(vertices[ face[0] - 1],vertices[ face[1] - 1],vertices[ face[2] - 1]));

        }
    }
    

    this->tris = triangles;
    this->triCount = triangles.size();

/*
    for(int i = 0; i < triCount;i++ ) {
        std::cout << (getTriangles()[i].getP1()) << "\n";
        std::cout << (getTriangles()[i].getP2()) << "\n";
        std::cout << (getTriangles()[i].getP3()) << "\n";

    }

*/

    std::cout << "Loaded Model: " << triangles.size() << " Triangles.\n";

    
}
