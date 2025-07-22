
#ifndef KERNELS_H
#define KERNELS_H


__host__ __device__
inline uint8_t floatToByte(float c)
{
    c = fminf(fmaxf(c, 0.0f), 1.0f); 
    return static_cast<uint8_t>(c * 255.0f + 0.5f); 
}


__device__
Vec3 fresnelSchlick(float cosTheta, const Vec3& F0)
{
    return F0 + (Vec3(1.0f, 1.0f, 1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__
float distributionGGX(const Vec3& N, const Vec3& H, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = fmaxf(dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    return a2 / (PI * denom * denom + 1e-6f);
}

__device__
float geometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

__device__
float geometrySmith(const Vec3& N, const Vec3& V, const Vec3& L, float roughness)
{
    float NdotV = fmaxf(dot(N, V), 0.0f);
    float NdotL = fmaxf(dot(N, L), 0.0f);
    float ggx1 = geometrySchlickGGX(NdotV, roughness);
    float ggx2 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

__device__
Vec3 cookTorranceBRDF(const Vec3& N, const Vec3& V, const Vec3& L, const Vec3& albedo, float roughness, float metallic, const Vec3& F0)
{
    Vec3 H = (V + L).normalize();

    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    Vec3 F = fresnelSchlick(fmaxf(dot(H, V), 0.0f), F0);

    Vec3 numerator = F * G * NDF;
    float denom = 4.0f * fmaxf(dot(N, V), 0.0f) * fmaxf(dot(N, L), 0.0f) + 1e-6f;
    Vec3 specular = numerator / denom;

    Vec3 kS = F;
    Vec3 kD = Vec3(1.0f, 1.0f, 1.0f) - kS;
    kD = kD * (1.0f - metallic);

    float NdotL = fmaxf(dot(N, L), 0.0f);
    return (kD * albedo / PI + specular) * NdotL;
}


__device__
bool hit(const Vec3& origin, const Vec3& direction, Sphere sphere, float tMin, float tMax, float& t, Vec3& normal, Vec3& hitPoint)
{
    Vec3 oc = origin - sphere.center;
    float a = direction.dot(direction);
    float b = oc.dot(direction);
    float c = oc.dot(oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - a * c;

    if (discriminant > 0) {
        float root = sqrtf(discriminant);
        float temp = (-b - root) / a;
        if (temp < tMax && temp > tMin)
        {
            t = temp;
            hitPoint = origin + direction * t; //AT
            normal = (hitPoint - sphere.center).normalize();
            return true;
        }
    }

    return false;
}

__device__
Vec3 shade(const Vec3& origin, const Vec3& direction, const Sphere* spheres, int sphereCount, Vec3 lightDir)
{
    Vec3 radiance = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 throughput = Vec3(1.0f, 1.0f, 1.0f);

    float closestT = 1e20f;
    Vec3 hitColor = Vec3(0, 0, 0);
    for (int i = 0; i < sphereCount; i++) 
    {
        float t;
        Vec3 normal, hitPoint;
        if (hit(origin, direction, spheres[i], 0.001f, closestT, t, normal, hitPoint))
        {
            closestT = t;
            Vec3 viewDir = -1*direction;
            Vec3 light = lightDir.normalize();
            Vec3 F0 = Vec3(0.04f, 0.04f, 0.04f) * (1.0f - spheres[i].metallic) + spheres[i].albedo * spheres[i].metallic;
//            radiance = radiance + (throughput);
            radiance = radiance + cookTorranceBRDF(normal, viewDir, light, spheres[i].albedo, spheres[i].roughness, spheres[i].metallic, F0);
        }
    }

    Vec3 skyColor = Vec3(0.6f, 0.7f, 0.9f);
    return radiance;
}



__global__
void render(uint8_t* framebuffer, int width, int height, Vec3 origin, Vec3 front, Sphere* spheres)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = (y * width + x) * 4;
    float u = ((float(x) + 0.5) / float(width)) * 2.0 - 1.0;
    float v = ((float(y) + 0.5) / float(height)) * 2.0 - 1.0;

    Vec3 right = front.cross(Vec3(0.0f, 1.0f, 0.0f)).normalize();
    Vec3 up = right.cross(front).normalize();

    Vec3 direction = (front + u * right + v * up).normalize();

    Vec3 color = shade(origin, direction, spheres, 5, Vec3(0.0f, 0.0f, 1.0f).normalize());

    framebuffer[index + 0] = floatToByte(color.x);
    framebuffer[index + 1] = floatToByte(color.y);
    framebuffer[index + 2] = floatToByte(color.z);
    framebuffer[index + 3] = 255;
}

#endif