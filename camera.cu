#include "camera.h"

__host__ __device__ camera::camera() {
    //initialize default camera parameters
         start = -10;//the near plane of the frustum
         end = -400;//far plane

         topPlane = tan(M_PI_4/2.0f) * start;


        //frustrum dimensions
         
        rightPlane = topPlane * (4.0f/3.0f);//4 by 3 aspect ratio
        leftPlane = -rightPlane;
         //topPlane = 200;
         bottomPlane = -topPlane;

        //vertical FOV (cross of top bottom)
         vertFOV = M_PI_4; //pi/2


}

__host__ __device__ vec4 camera::perspectiveProjection(vec4 point) {
    //symmetric viewing volume
vec4 ProjMat[4] = {vec4(start/rightPlane, 0.0f, 0.0f , 0.0f),
                vec4(0.0f, start/topPlane,0.0f, 0.0f),
                vec4(0.0f,0.0f,-1.0f, -2.0f * start),
                vec4(0.0f,0.0f,-1.0f,0.0f)};

    vec4 newVec = vec4(dot_product4(ProjMat[0], point),
                dot_product4(ProjMat[1], point),
                dot_product4(ProjMat[2], point),
                dot_product4(ProjMat[3], point));



    if(newVec.w() != 0.0f) {
        newVec = newVec/newVec.w();
    }

    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    return newVec;
}
