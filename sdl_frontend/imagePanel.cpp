
#include "imagePanel.h"

#include "constants.h"
#include "main.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_image.h>

ImagePanel::ImagePanel()
{

}

void ImagePanel::setID(int id)
{
    this->id = id;
}

void ImagePanel::setTexture(char* texture_path)
{
    this->paletteSurface = loadSurface(texture_path);
    this->paletteTexture = SDL_CreateTextureFromSurface( renderer, this->paletteSurface);
}

void ImagePanel::render(SDL_Renderer* renderer)
{
    SDL_Rect paletteRect = { this->position.x, this->position.y, PALETTE_WIDTH, PALETTE_HEIGHT };
    SDL_RenderCopy(renderer, this->paletteTexture, NULL, &paletteRect);
}

void ImagePanel::setPosition(Point position)
{
    this->position = position;
}

void ImagePanel::setDimensions(int width, int height)
{
    this->width = width;
    this->height = height;
}
