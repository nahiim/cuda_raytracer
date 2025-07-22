
#ifndef MATH_H
#define MATH_H


__device__ const float PI = 3.14159265359;

struct Vec3
{
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float a, float b, float c) : x(a), y(b), z(c) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    __host__ __device__ Vec3 operator*(const Vec3& v) const { return Vec3(x * v.x, y * v.y, z * v.z); }
    __host__ __device__ Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }

    __host__ __device__ float dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }

    __host__ __device__ Vec3 normalize() const
    {
        float len = sqrtf(x * x + y * y + z * z);
        return (*this) / len;
    }

    __host__ __device__ Vec3 cross(const Vec3& a) const
    {
        return Vec3(
            y * a.z - z * a.y,
            z * a.x - x * a.z,
            x * a.y - y * a.x
        );
    }
};
__host__ __device__
inline Vec3 operator*(float s, const Vec3& v)
{
    return Vec3(v.x * s, v.y * s, v.z * s);
}
__host__ __device__
inline float dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}



#endif
