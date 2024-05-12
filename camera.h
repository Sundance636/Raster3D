#ifndef __camera_h__
#define __camera_h__

#include "vec4.h"
#include "entity.h"

class camera {

    private:
        //frustum attributes
        
        //starting and end planes (0 > start < end)

        float start;
        float end;

        //frustrum dimensions
        vec4 left; //direction of plane
        float leftPlane;
        vec4 right;//direction of plane
        float rightPlane;

        vec4 top;
        float topPlane;
        vec4 bottom;
        float bottomPlane;

        //horizontal FOV (cross product of left and right)
        float horiFOV;

        //vertical FOV (cross of top bottom)
        float vertFOV;

        entity* objects;

        //vec4 ProjMat[4] = {vec4((2.0f * start)/ (rightPlane - leftPlane), 0.0f, -1.0*(rightPlane - leftPlane)/(rightPlane - leftPlane) , 0.0f),
        //        vec4(0.0f, 2.0f * start / (topPlane - bottomPlane), -1.0 * (topPlane + bottomPlane)/(topPlane - bottomPlane), 0.0f),
        //        vec4(0.0f,0.0f,(end + start)/(end - start), -(2.0 * end * start)/(end - start)),
        //        vec4(0.0f,0.0f,1.0f,0.0f)};



    public:
        __host__ __device__ camera();
        __host__ __device__ vec4 perspectiveProjection(vec4);



};


#endif