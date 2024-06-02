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
        vec4 eyeLine =  camPosition - tris[idx].getP3();
        eyeLine.sety(camPosition.y() -  -1.0f*tris[idx].getP3().y() );
        eyeLine = unit_vector4(eyeLine);
        eyeLine.setx(-eyeLine.x());
        eyeLine.setz(-eyeLine.z());
        eyeLine.setw(0);

        facingRatios[idx] = dot_product4(tris[idx].getSurfaceNormal(), eyeLine);

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
