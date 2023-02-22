#include "main.h"
#include "render.h"
#include "study.h"
#include "document.h"

#include <vector>
#include <map>
#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL_image.h>
#include <stdlib.h>


void render(SDL_Renderer* renderer)
{
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);    //

    //cout << "document alive render: " << document.alive << endl;

    if(document.alive) renderHighlights(renderer);
    if(currentMode == phrase) renderPhrase(renderer);
    if(currentMode == image) renderImage(renderer);
    if(SHOW_LINES) renderLines(renderer);
    if(currentMode == cross && isSaving == false) renderCrosses(renderer);
    if(SHOW_HOVER_INDICATOR && currentMode != cross) renderHoverIndicator(renderer);
    if(SHOW_PARTICLES) renderParticles(renderer);
    if(showBrokenPipeIndicator) renderBrokenPipeIndicator(renderer);

    if(!isSaving) SDL_RenderPresent(renderer);
}

void renderLine(SDL_Renderer *rend, vector<Point> *line, SDL_Color color)
{
    if(line->size() > 1)
    {
        SDL_Point point_array[line->size()];
        for(int i = 0; i < line->size(); i++)
        {
            point_array[i] = pointToSDL(line->at(i));
        }
        SDL_SetRenderDrawColor(rend, color.r, color.g, color.b, 255);
        SDL_RenderDrawLines(rend, point_array, line->size());
    }
}

void renderParticles(SDL_Renderer* renderer)
{
    // fixme

    // if draw
    int x = currentX;
    int y = currentY;
    for (int i = 0; i < 10; i++)
    {
        Particle trailParticle = Particle(x, y, 0xFF0088FF);
        trailParticle.setAngle((rand() % 359) / 180 * M_PI);
        trailParticle.setLifetime(5 + (rand() % 5) / 5.0f);
        trailParticle.setVelocity(1 + rand() % 3);
        particles.push_back(trailParticle);
    }

    for (vector<Particle>::iterator it = particles.begin(); it != particles.end();)
    {
        bool alive = it->update(0.2);
        if(!alive)
            it = particles.erase(it);
        else 
            ++it;
    }

    for(const auto particle : particles)
    {
        particle.render(renderer);
    }
}

void renderCrosses(SDL_Renderer* renderer)
{
    SDL_Rect crossesRect = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };
    SDL_RenderCopy(renderer, crossesTexture, NULL, &crossesRect);
}

void renderImage(SDL_Renderer* renderer)
{
    SDL_Rect imageRect = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };
    SDL_RenderCopy(renderer, imageTexture, NULL, &imageRect);
}

void renderPhrase(SDL_Renderer* renderer)
{
    int w = textSurface->w;
    int h = textSurface->h;

    SDL_Rect phraseRect = { WINDOW_WIDTH / 2 - w / 2, 200, w, h };

    SDL_RenderCopy( renderer, textTexture, NULL, &phraseRect );

    int textBoxWidth = TEXT_BOX_WIDTH;
    int textBoxHeight;

    switch(currentTextSize)
    {
        case 2:
            textBoxHeight = TEXT_BOX_HEIGHT_SMALL;
            break;
        case 1:
            textBoxHeight = TEXT_BOX_HEIGHT_MEDIUM;
            break;
        case 0:
            textBoxHeight = TEXT_BOX_HEIGHT_LARGE;
            break;
    }

    SDL_Rect textBoxRect = { WINDOW_WIDTH / 2 - textBoxWidth / 2,
        WINDOW_HEIGHT / 2 - textBoxHeight / 2 + TEXTBOX_OFFSET,
        textBoxWidth, textBoxHeight };

    SDL_SetRenderDrawColor(renderer, 50, 50, 50, 255);
    SDL_RenderFillRect(renderer, &textBoxRect);
}

void renderHighlights(SDL_Renderer* renderer)
{
    for (auto const& entry : rects)
    {
        struct Poly poly = entry.second;

        filledPolygonColor(renderer, poly.x, poly.y, 4, highlightColor);
    }
}

void renderLines(SDL_Renderer* renderer)
{
    for (auto const& line : lines)
    {
        vector<Point> coords = line.coords;
        renderLine(renderer, &coords, line.color);
    }

    if (document.alive)
    {
	    for (auto const& line : documentLines)
	    {
		vector<Point> coords = line.coords;
		renderLine(renderer, &coords, {255, 255, 0});
	    }
    }

    for (auto const& entry : pens)
    {
        struct Pen pen = entry.second;

        if(pen.alive)
        {
            if(pen.currentLine.coords.size() > 1)
            {
		// TODO: use renderLine() ?
                SDL_Point point_array[pen.currentLine.coords.size()];
                for(int i = 0; i < pen.currentLine.coords.size(); i++)
                {
                    point_array[i] = pointToSDL(pen.currentLine.coords.at(i));
                }
                SDL_SetRenderDrawColor(renderer, pen.currentLine.color.r, pen.currentLine.color.g, pen.currentLine.color.b, 255);
                SDL_RenderDrawLines(renderer, point_array, pen.currentLine.coords.size());
            }
        }
    }
}

void renderHoverIndicator(SDL_Renderer* renderer)
{
    // fixme
    //filledCircleColor(renderer, currentX, currentY, 3, HOVER_INDICATOR_COLOR);

    for (auto const& entry : pens)
    {
        struct Pen pen = entry.second;

        if(pen.state == 0 && pen.alive == 1)
        {
            filledCircleColor(renderer, pen.position.x, pen.position.y, 3, HOVER_INDICATOR_COLOR);
        }
    }
}

void renderLatencyTest(SDL_Renderer* renderer)
{
    // fixme
    /*
    if(currentState == STATE_DRAW)
    {
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    }
    else
    {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    }
    */
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
}

void renderBrokenPipeIndicator(SDL_Renderer* renderer)
{
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_Rect brokenPipeIndicator = { 0, 0, 20, 20 };

    SDL_RenderFillRect(renderer, &brokenPipeIndicator);
}

