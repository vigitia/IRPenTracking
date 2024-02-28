
#include "main.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_image.h>

ImagePanel::ImagePanel()
{
    this->visible = true;
}

void ImagePanel::setID(int id)
{
    this->id = id;
}

int ImagePanel::getID()
{
    return this->id;
}

void ImagePanel::loadTexture(char* texture_path)
{
    this->imageSurface = loadSurface(texture_path);
    this->imageTexture = SDL_CreateTextureFromSurface( renderer, this->imageSurface);
}

void ImagePanel::loadTexture()
{
    if (this->defaultImagePath == nullptr){
        cout << "Error loading default texture in ImagePanel " << this->getID() << ": No default texture assigned." << endl; 
    }
    else{
        this->loadTexture(this->defaultImagePath);
    }
}

void ImagePanel::setTexture(SDL_Texture* texture)
{
    this->imageTexture = texture;
}

void ImagePanel::setDefaultImagePath(char * imagePath)
{
    this->defaultImagePath = imagePath;
}

void ImagePanel::render(SDL_Renderer* renderer)
{
    if (this->visible)
    {
        SDL_Rect imageRect = { this->position.x, this->position.y, static_cast<int>(this->width), static_cast<int>(this->height) };
        int error = SDL_RenderCopy(renderer, this->imageTexture, NULL, &imageRect);
        //if (error)
        //{
        //    cout << SDL_GetError() << endl;
        //}
    }
}

void ImagePanel::setPosition(Point position)
{
    this->position = position;
}

Point ImagePanel::getPosition()
{
    return this->position;
}

void ImagePanel::setDimensions(int width, int height)
{
    this->width = width;
    this->height = height;
}

int ImagePanel::getWidth()
{
    return this->width;
}

int ImagePanel::getHeight()
{
    return this->height;
}

void ImagePanel::setVisibility(bool visible)
{
    this->visible = visible;
}

bool ImagePanel::getVisibility()
{
    return this->visible;
}
