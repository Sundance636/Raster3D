#include "vec4.h"


__host__ __device__ vec4::vec4() {

}

__host__ __device__ vec4::vec4(float e0,float e1,float e2, float e3) {
    this->e[0] = e0;
    this->e[1] = e1;
    this->e[2] = e2;
    this->e[3] = e3;
}

__host__ __device__ float vec4::x() const {
    return this->e[0];
}

__host__ __device__ float vec4::y() const {
    return this->e[1];

}

__host__ __device__ float vec4::z() const {
    return this->e[2];

}

__host__ __device__ float vec4::w() const {
    return this->e[3];

}

__host__ __device__ void vec4::setx(float newVal) {
    this->e[0] = newVal;

}

__host__ __device__ void vec4::sety(float newVal) {
    this->e[1] = newVal;
}
__host__ __device__ void vec4::setz(float newVal) {
    this->e[2] = newVal;
}
__host__ __device__ void vec4::setw(float newVal) {
    this->e[3] = newVal;
}

__host__ __device__ vec4 vec4::operator=(const vec4 &otherVector) {
    this->e[0] = otherVector.e[0];
    this->e[1] = otherVector.e[1];
    this->e[2] = otherVector.e[2];
    this->e[3] = otherVector.e[3];


    return *this;
}

__host__ __device__ vec4 vec4::operator+(const vec4 &vector) {
    this->e[0] = this->e[0] + vector.e[0];
    this->e[1] = this->e[1] + vector.e[1];
    this->e[2] = this->e[2] + vector.e[2];
    this->e[3] = this->e[3] + vector.e[3];
    return *this;
}

__host__ __device__ vec4 vec4::operator-(const vec4 &vector) {
    this->e[0] = this->e[0] - vector.e[0];
    this->e[1] = this->e[1] - vector.e[1];
    this->e[2] = this->e[2] - vector.e[2];
    this->e[3] = this->e[3] + vector.e[3];
    return *this;
}

__host__ __device__ float& vec4::operator[](const int index) {
    return (this->e[index]);
}

__host__ std::ostream& vec4::operator<<(std::ostream& out) {
    return out << "[" << this->e[0] << ", " << this->e[1] << ", " << this->e[2] << ", " << this->e[3] << "]";
}

__host__ std::ostream& operator<<(std::ostream& out, const vec4& o) {
    return out << "[" << o.x() << ", " << o.y() << ", " <<  o.z() << ", " <<  o.w() << "]";
}


__host__ __device__ vec4 operator*(const vec4 &v, float t) {
    return vec4(t* v.x(), t*v.y(), t*v.z(), t*v.w());
}

__host__ __device__ vec4 operator*(float t, const vec4 &v) {
    return vec4(t* v.x(), t*v.y(), t*v.z(), t*v.w());
}

__host__ __device__ vec4 operator/(const vec4 &v, float t) {
    return vec4(v.x()/t, v.y()/t, v.z()/t, v.w()/t);
}

__host__ __device__ vec4 operator/(float t, const vec4 &v) {
    return vec4(v.x()/t, v.y()/t, v.z()/t, v.w()/t);
}

__host__ __device__ vec4 operator+(const vec4 &Vector1, const vec4 &Vector2) {
    return vec4( Vector1.x() + Vector2.x(), Vector1.y() + Vector2.y(), Vector1.z() + Vector2.z(), Vector1.w() + Vector2.w());
}

__host__ __device__ float vec4::magnitude() {
    return sqrtf( (this->e[0] * this->e[0]) + (this->e[1]*this->e[1]) + (this->e[2]*this->e[2]) + (this->e[3]*this->e[3]) );
}

__device__ vec4 unit_vector(vec4 vector) {
    return vector / vector.magnitude(); //normalize the vector
}

__host__ __device__ float dot_product4(vec4 vec1, vec4 vec2) {
    return (vec1.x() * vec2.x()) + (vec1.y() * vec2.y()) + (vec1.z() * vec2.z()) + (vec1.w() * vec2.w());
}

