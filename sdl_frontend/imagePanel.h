//#ifndef PALETTE_H
//#define PALETTE_H

#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>

#include "main.h"

#define PALETTE_TEXTURE_PATH "assets/small_palette.png"
#define PALETTE_WIDTH 1000
#define PALETTE_HEIGHT 200


class ImagePanel
{
    protected:
        int id;
        Point position;
        int width;
        int height;

        SDL_Surface *paletteSurface;
        SDL_Texture *paletteTexture;

    public:
        ImagePanel();
        void render(SDL_Renderer* renderer);
        void setPosition(Point Position);
        void setID(int id);
        void setTexture(char* texturePath);
        void setDimensions(int width, int height);
        
        
};



inline ImagePanel palette;
