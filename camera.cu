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

         //initialize camera to origin for now
         position = vec4 (0,0,0,1);

        //default look is straight forward on Z
        look = vec4(0,0,1,1);
        lookAngle = 0.0f;

        up = vec4(0,0,1,1);
        upAngle = 0.0f;



}

__host__ __device__ vec4 camera::perspectiveProjection(vec4 point) {
    //symmetric viewing volume
vec4 ProjMat[4] = {vec4(start/rightPlane, 0.0f, 0.0f , 0.0f),
                vec4(0.0f, start/topPlane,0.0f, 0.0f),
                vec4(0.0f,0.0f,-1.0f, -2.0f * start),
                vec4(0.0f,0.0f,-1.0f,0.0f)};

    //translate point p, about the position of cam
    translation(vec4(-position.x(),-position.y(),-position.z(),-position.w()), point);

    //rotate point p
    rotationY(lookAngle,point);
    rotationX(upAngle,point);

    //translate back
    translation(position, point);
    //translation(position, point);

    //the position offset should account for cameras position during transformations
    vec4 newVec = vec4(dot_product4(ProjMat[0], point + position ),
                dot_product4(ProjMat[1], point + position ),
                dot_product4(ProjMat[2], point + position ),
                dot_product4(ProjMat[3], point + position ));



    if(newVec.w() != 0.0f) {
        newVec = newVec/newVec.w();
    }

    point.setx(newVec.x());
    point.sety(newVec.y());
    point.setz(newVec.z());
    point.setw(newVec.w());

    return newVec;
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
