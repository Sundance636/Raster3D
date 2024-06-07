#include "camera.h"

__host__ __device__ camera::camera() {
    //initialize default camera parameters
         start = 100;//the near plane of the frustum
         end = 600;//far plane

         topPlane = tan(M_PI_4/2.0f) * start;

        //vertical FOV (cross of top bottom)
         vertFOV = M_PI_4; // 90 degrees

        //frustrum dimensions
         
        rightPlane = topPlane * (4.0f/3.0f);//4 by 3 aspect ratio
        leftPlane = -rightPlane;
        left = vec4(-1,0,vertFOV/2,0);//just a direction
        right = vec4(1,0,vertFOV/2,0);//just a direction

        top = vec4(0,1,vertFOV/2,0);
        bottom = vec4(0,1,vertFOV/2,0);

         //topPlane = 200;
         bottomPlane = -topPlane;

        

         //initialize camera to origin for now
         position = vec4 (0,0,0,1);

        //default look is straight forward on Z
        look = vec4(0,0,1,1);
        lookAngle = 0.0f;

        up = vec4(0,0,1,1);
        upAngle = 0.0f;



}

__host__ __device__ vec4 camera::perspectiveProjection(vec4 point) {
    //symmetric viewing volume (probably correct i dunno?)
    vec4 ProjMat[4] = {vec4(1.0f/rightPlane, 0.0f, 0.0f , 0.0f),
                    vec4(0.0f, 1.0f/topPlane,0.0f, 0.0f),
                    vec4(0.0f,0.0f,1.0f, 0.0f),
                    vec4(0.0f,0.0f,1.0f,0.0f)};


    //the position offset should account for cameras position during transformations
    vec4 newVec = vec4(dot_product4(ProjMat[0], point ),
                dot_product4(ProjMat[1], point ),
                dot_product4(ProjMat[2], point ),
                dot_product4(ProjMat[3], point ));



    if(newVec.w() != 0.0f) {
        newVec = newVec/newVec.w();
    }

    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    //newVec.sety(-newVec.y());//y orientation it flipped to screen space
    //newVec.setx(-newVec.x());
    //newVec.setz(-newVec.z());

    return newVec;
}

__host__ entity camera::perspectiveProjectionR(entity &object) {

    triangle* trisArray = object.getTriangles();//pass vec as an array
    triangle* d_tris;

    checkCudaErrors(cudaMalloc((void**)&d_tris, object.getTriCount() * sizeof(triangle)));
    checkCudaErrors(cudaMemcpy(d_tris,trisArray, object.getTriCount() * sizeof(triangle), cudaMemcpyHostToDevice));


    //ENSURE THESE TWO NUMBERS ARE OPTIMAL
    int blockSize = 256;
    int numBlocks = (object.getTriCount() + blockSize - 1) / blockSize;

    projectionK<<<numBlocks, blockSize>>>(this->rightPlane, this->leftPlane, this->topPlane, this->bottomPlane, this->end , this->start, d_tris, object.getTriCount());

    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    //copy back
    checkCudaErrors(cudaMemcpy(trisArray,d_tris, object.getTriCount() * sizeof(triangle), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tris));
    return entity(object);
}

__global__ void projectionK(float rightPlane,float leftPlane, float topPlane, float bottomPlane, float farPlane, float nearPlane, triangle* tris, int numOfTris) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(idx < numOfTris) { //each triangle
        //symmetric viewing volume (probably correct i dunno?)
        vec4 ProjMat[4] = { vec4(2*nearPlane/(rightPlane - leftPlane), 0.0f, (rightPlane + leftPlane)/(rightPlane - leftPlane) , 0.0f),
                            vec4(0.0f, 2*nearPlane/(topPlane - bottomPlane),(topPlane+bottomPlane)/(topPlane-bottomPlane), 0.0f),
                            vec4(0.0f,0.0f,((farPlane +nearPlane )/(farPlane-nearPlane)), -(2*farPlane* nearPlane/(farPlane-nearPlane))),
                            vec4(0.0f,0.0f,1.0f,0.0f)};

        vec4 points[3] = {tris[idx].getP1(), tris[idx].getP2(), tris[idx].getP3()};


        vec4 newVec = vec4( dot_product4(ProjMat[0], points[0]),
                            dot_product4(ProjMat[1], points[0]),
                            dot_product4(ProjMat[2], points[0]),
                            dot_product4(ProjMat[3], points[0]));

        if(newVec.w() != 0.0f) {
            newVec = newVec/newVec.w();
        }
        
        tris[idx].setP1(newVec);

        newVec = vec4(      dot_product4(ProjMat[0], points[1]),
                            dot_product4(ProjMat[1], points[1]),
                            dot_product4(ProjMat[2], points[1]),
                            dot_product4(ProjMat[3], points[1]));

        if(newVec.w() != 0.0f) {
            newVec = newVec/newVec.w();
        }

        tris[idx].setP2(newVec);

        newVec = vec4(      dot_product4(ProjMat[0], points[2]),
                            dot_product4(ProjMat[1], points[2]),
                            dot_product4(ProjMat[2], points[2]),
                            dot_product4(ProjMat[3], points[2]));

        if(newVec.w() != 0.0f) {
            newVec = newVec/newVec.w();
        }

        tris[idx].setP3(newVec);
    }

}

__host__ __device__ float camera::getLookAngle() {
    return this->lookAngle;
}

__host__ __device__ vec4 camera::getLookVec() {
    return this->look;
}

__host__ __device__ float camera::getUpAngle() {
    return this->upAngle;
}

__host__ __device__ vec4 camera::getUpVec() {
    return this->up;
}

__host__ __device__ vec4 camera::movecam(vec4 nudge) {
    position = position + nudge;

    return position;
}

__host__ __device__ vec4 camera::rotateLook(float radians) {
    look.setw(0);



    //look = unit_vector4(look);
    float numerator = dot_product4(look, vec4(0,0,1,0));// straight forward on Z axis
    
    //inital cross product should be zero vec

    if((look.magnitude() * vec4(0,0,1,0).magnitude()) != 0) {
        float sinval = numerator / look.magnitude() * vec4(0,0,1,0).magnitude();

        float theta;// = acos(sinval) + radians;

        if(radians < 0.0) {
            theta = -acos(sinval) + radians;

            if(lookAngle > 0.0) {
                theta = acos(sinval) + radians;//good for rot
            } else {
                theta = -acos(sinval) + radians;
            }

        } else {

            if(lookAngle < 0.0) {
                theta = -acos(sinval) + radians;//good for rot
            } else {
                theta = acos(sinval) + radians;
            }

            
            
        }

        if(theta >= M_PI) {
        lookAngle = -(2 * M_PI - theta);
        //radians = -radians;
        }
        else if(theta <= -M_PI) {
            lookAngle = 2* M_PI + theta;
        }
        else {
            lookAngle = theta;
        }

        
        
        //look = unit_vector4(look);
        //look.setw(1);
        //look = unit_vector4(look);
        
        look = vec4(0,0,1,1);
        rotationY(lookAngle,look);//rotate look vec to recalc angle
        //look = unit_vector4(look);
    }

    

    return vec4(0,0,1,0);

}

__host__ __device__ vec4 camera::rotateUp(float radians) {
    up.setw(0);



    //look = unit_vector4(look);
    float numerator = dot_product4(up, vec4(0,0,1,0));// straight forward on Z axis
    
    //inital cross product should be zero vec

    if((up.magnitude() * vec4(0,0,1,0).magnitude()) != 0) {
        float sinval = numerator / up.magnitude() * vec4(0,0,1,0).magnitude();

        float theta;// = acos(sinval) + radians;

        if(radians < 0.0) {
            theta = -acos(sinval) + radians;

            if(upAngle > 0.0) {
                theta = acos(sinval) + radians;//good for rot
            } else {
                theta = -acos(sinval) + radians;
            }

        } else {

            if(upAngle < 0.0) {
                theta = -acos(sinval) + radians;//good for rot
            } else {
                theta = acos(sinval) + radians;
            }

            
            
        }

        if(theta >= M_PI) {
        upAngle = -(2 * M_PI - theta);
        //radians = -radians;
        }
        else if(theta <= -M_PI) {
            upAngle = 2* M_PI + theta;
        }
        else {
            upAngle = theta;
        }

        
        
        //look = unit_vector4(look);
        //look.setw(1);
        //look = unit_vector4(look);
        
        up = vec4(0,0,1,1);
        rotationX(upAngle,up);//rotate look vec to recalc angle
    }

    

    return vec4(0,0,1,0);

}

__host__ __device__ vec4 camera::direction() {
    vec4 reference = vec4(0,0,1,1);
    rotationY(lookAngle,reference);
    rotationX(upAngle,reference);

    

    reference.setw(0);
    reference = unit_vector4(reference);

    //reference = reference + position;

    return reference;
}

__host__ __device__ vec4 camera::getPosition() {
    return this->position;
}


// rewite to take entity and do a batch transformation on all the tris to cam coodinates
//

__host__ __device__ vec4 camera::viewTransform(vec4 point) {

    //pseudo view matrix operations
    point.setw(1);
    vec4 orientation = vec4(position);

    orientation.setz(orientation.z() * -1.0f);
    orientation.setx(orientation.x() * -1.0f);

    translation(orientation, point);
    //translate point p, about the position of cam

    point.setw(1);
    //rotate point p
    rotationY(-lookAngle,point);
    rotationX(-upAngle,point);
    


    //translate back
    //point.setw(1);
    //translation(position, point);


    //translate point relative to the cams position
    //point.setw(1);
    //vec4 orientation = vec4(position);
    //orientation.sety(-position.y());
    //translation(-1.0f*orientation, point);

    return point;
}

__host__ entity camera::viewTransformR(entity& object) {

    triangle* trisArray = object.getTriangles();//pass vec as an array
    //std::cout << "tri ref: " << trisArray[0].getP1() << "\n";
    //trisArray[0].setP1(vec4(69,69,69,69));

    triangle* d_tris = nullptr;

    checkCudaErrors(cudaMalloc((void**)&d_tris, object.getTriCount() * sizeof(triangle)));
    checkCudaErrors(cudaMemcpy(d_tris,trisArray, object.getTriCount() * sizeof(triangle), cudaMemcpyHostToDevice));

    //ENSURE THESE TWO NUMBERS ARE OPTIMAL
    int blockSize = 256;
    int numBlocks = (object.getTriCount() + blockSize - 1) / blockSize;


//pseudo view matrix operations
    
    vec4 orientation = vec4(position);
    orientation.setz(orientation.z() * -1.0f);
    orientation.setx(orientation.x() * -1.0f);

    //translate all the tris
    translationK<<<numBlocks, blockSize>>>(orientation,d_tris, object.getTriCount());
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    //then rotate them
    rotationYK<<<numBlocks, blockSize>>>(-lookAngle,d_tris,object.getTriCount());
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    
    rotationXK<<<numBlocks, blockSize>>>(-upAngle,d_tris,object.getTriCount());
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    //copy back

    checkCudaErrors(cudaMemcpy(trisArray,d_tris, object.getTriCount() * sizeof(triangle), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_tris));
    d_tris = nullptr;

    return entity(object);
}

__host__ void camera::faceCulling(std::vector<float>&faceRatios, entity &object) {
    triangle* trisArray = object.getTriangles();//pass vec as an array
    float* faceArray = faceRatios.data();

    triangle* d_tris = nullptr;
    float* d_facenorm = nullptr;

    checkCudaErrors(cudaMalloc((void**)&d_tris, object.getTriCount() * sizeof(triangle)));
    checkCudaErrors(cudaMemcpy(d_tris,trisArray, object.getTriCount() * sizeof(triangle), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&d_facenorm, object.getTriCount() * sizeof(float)));


    //ENSURE THESE TWO NUMBERS ARE OPTIMAL
    int blockSize = 256;
    int numBlocks = (object.getTriCount() + blockSize - 1) / blockSize;
    

    cullingK<<<numBlocks, blockSize>>>(this->getPosition(),d_tris,d_facenorm,object.getTriCount()); 
    checkCudaErrors (cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());    

    checkCudaErrors(cudaMemcpy(faceArray, d_facenorm, object.getTriCount() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_facenorm));
    checkCudaErrors(cudaFree(d_tris));

    d_tris = nullptr;
    d_facenorm = nullptr;
}

__host__ void camera::frustumCulling(std::vector<float>&faceRatios, entity& object) {
    for(int i = 0; i < object.getTriCount(); i++) {
        /*
        if(std::min(object[i].getP1().y(),std::min(object[i].getP2().y(), object[i].getP3().y())) > 1.0 ) {
            faceRatios[i] = 1.0;
        } else if(std::min(object[i].getP1().x(),std::min(object[i].getP2().x(), object[i].getP3().x())) > 1.0 ) {
            faceRatios[i] = 1.0;
        } else if(std::max(object[i].getP1().z(),std::max(object[i].getP2().z(), object[i].getP3().z())) < 0.0 ) {
            faceRatios[i] = 1.0;

        } else if(std::min(object[i].getP1().z(),std::min(object[i].getP2().z(), object[i].getP3().z())) > 1.50) {
            faceRatios[i] = 1.0;
        }*/

        if(!(triInFrustum(object[i]))) {
            faceRatios[i] = 1.0;

        }
    }
}

//camera function for checking if all the triangles of a given entity are even in the frustum
//in  this case for clipping the entire tri

__host__ bool camera::triInFrustum(triangle tri) {
   
   //if(tri.getP1() - ) 

}