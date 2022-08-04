#include "particle.h"
#include "constants.h"

Particle::Particle(float x, float y, Uint32 color)
{
    this->x = x;
    this->y = y;

    this->color = color;

    size = PARTICLE_SIZE_MIN + rand() % (PARTICLE_SIZE_MAX - PARTICLE_SIZE_MIN);
    float v = PARTICLE_VELOCITY_MIN + rand() % (PARTICLE_VELOCITY_MAX - PARTICLE_VELOCITY_MIN);
    a = PARTICLE_ANGLE_MIN + rand() % (PARTICLE_ANGLE_MAX - PARTICLE_ANGLE_MIN);
    a = (a * M_PI / 180);

    vx = sin(a) * v;
    vy = cos(a) * v;
    lifetime = PARTICLE_LIFETIME_MIN + (float)(rand()) / (RAND_MAX / (PARTICLE_LIFETIME_MAX - PARTICLE_LIFETIME_MIN));
    age = 0;
}

bool Particle::update(float timedelta)
{
    x += vx * timedelta;
    y += vy * timedelta;

    vy += GRAVITY * timedelta;

    age += timedelta;

    // still alive?
    if(y > HEIGHT || age > lifetime)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void Particle::render(SDL_Renderer *renderer) const
{
    filledCircleColor(renderer, x, y, size, color);
}

void Particle::setAngle(float a)
{
    this->a = a;
}

void Particle::setLifetime(float lifetime)
{
    this->lifetime = lifetime;
}

void Particle::setVelocity(float v)
{
    vx = sin(a) * v;
    vy = cos(a) * v;
}
