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

    
}
