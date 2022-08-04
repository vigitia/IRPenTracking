#ifndef PARTICLE_H
#define PARTICLE_H

#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

const int PARTICLE_ANGLE_MIN = 20;
const int PARTICLE_ANGLE_MAX = 340;
const int PARTICLE_SIZE_MIN = 1;
const int PARTICLE_SIZE_MAX = 3;
const int PARTICLE_VELOCITY_MIN = 100;
const int PARTICLE_VELOCITY_MAX = 400;
const float PARTICLE_LIFETIME_MIN = 0.1;
const float PARTICLE_LIFETIME_MAX = 0.5;

class Particle
{
    protected:
        float x, y;
        int size;
        float vx, vy;
        float a;
        float lifetime, age;
        uint32_t color;
    public:
        Particle(float x, float y, Uint32 color);
        bool update(float timedelta);
        void render(SDL_Renderer *renderer) const;
        void setAngle(float a);
        void setLifetime(float lifetime);
        void setVelocity(float v);
};

#endif
