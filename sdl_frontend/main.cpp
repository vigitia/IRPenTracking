#include "main.h"
#include "uds.h"
#include "fifo.h"
#include "document.h"
#include "study.h"
#include "particle.h"
#include "render.h"
#include "path_game.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h> 
#include <errno.h>

#include <typeinfo> //for debug purposes only. TODO: Remove



#include <sys/stat.h>
#include <signal.h>
#include <iostream>
#include <SDL2/SDL.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL_image.h>
#include <vector>
#include <map>
#include <ctime>
#include <iostream>
#include <fstream>

using namespace std;

void clearScreen()
{
    mutex_lines.lock();
    lines.clear();
    documentLines.clear();
    currentLine.coords.clear();
    mutex_lines.unlock();
}

void onExit(int signum)
{
    cout << "exiting..." << endl;

    // end inter process communication
    if(COMMUNICATION_MODE == MODE_FIFO)
    {
        pthread_cancel(fifo_thread);
        unlink(fifo_path);
    }
    else if(COMMUNICATION_MODE == MODE_UDS)
    {
        pthread_cancel(uds_thread);
        close(client_socket);
        close(server_socket);
    }

    exit(EXIT_SUCCESS);
}

void onBrokenPipe(int signum)
{
    showBrokenPipeIndicator = true;
}

// https://lazyfoo.net/tutorials/SDL/06_extension_libraries_and_loading_other_image_formats/index2.php
SDL_Surface* loadSurface(string path)
{
    //The final optimized image
    SDL_Surface* optimizedSurface = NULL;

    //Load image at specified path
    SDL_Surface* loadedSurface = IMG_Load( path.c_str() );
    if( loadedSurface == NULL )
    {
        printf( "Unable to load image %s! SDL_image Error: %s\n", path.c_str(), IMG_GetError() );
    }

    return loadedSurface;
}

void saveImage()
{ 
    char filename[120];
    sprintf(filename, "%s%d_%s.png", SCREENSHOT_PATH, participantId, currentDateTime().c_str());

    const Uint32 format = SDL_PIXELFORMAT_ARGB8888;

    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, WINDOW_WIDTH, WINDOW_HEIGHT, 32, format);

    isSaving = true;

    render(renderer);
    usleep(20000);
    SDL_RenderReadPixels(renderer, NULL, format, surface->pixels, surface->pitch);
    IMG_SavePNG(surface, filename);

    isSaving = false;
}

void saveProfilePicture()
{
    const Uint32 format = SDL_PIXELFORMAT_ARGB8888;
    cout << pathGame.profileRect.w << " " << pathGame.profileRect.h << endl;
    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, pathGame.profileRect.w, pathGame.profileRect.h, 32, format);

    //renderProfilePicture(renderer);
    usleep(20000);

    char filename[400];
    sprintf(filename, "%s/%d.png", HIGHSCORE_PATH, pathGame.participant_id);

    SDL_RenderReadPixels(renderer, &pathGame.profileRect, format, surface->pixels, surface->pitch);
    IMG_SavePNG(surface, filename);

    pathGame.isSavingProfilePicture = false;
}

void preloadTextures(SDL_Renderer* renderer)
{
    SDL_Surface* preloadedPaletteSurface = loadSurface("assets/big_palette_expanded.png");
    preloadedPaletteTexture = SDL_CreateTextureFromSurface( renderer, preloadedPaletteSurface);

    SDL_Surface* preloadedPaletteIndicatorSurface = loadSurface("assets/palette_indicator.png");
    preloadedPaletteIndicatorTexture = SDL_CreateTextureFromSurface( renderer, preloadedPaletteIndicatorSurface);

}

void initPalette()
{
    cout << "Initializing Palette..." << endl;
    vector<vector<vector<int>>> colors = {{
        {-2,-2,-2}, // == Clear everything
        {-1, -1, ERASE_RADIUS_BIG}, // == Eraser, bigger Radius
        {-1, -1, ERASE_RADIUS_SMALL}, // == Eraser, smaller Radius
        {255,  51, 255}, // == an RGB color value (as everything below)
        {255,  51,  51},
        {255, 149,   0},
        {255, 255,  17},
        { 51, 255,  51},
        { 51, 238, 238},
        { 76,  76, 255},
        {128, 128, 128},
        {255, 255, 255}
    }};

    //std::shared_ptr<Widget> smartPointerToWidget(new Palette(colors, 1, 12, PALETTE_HEIGHT));
    std::shared_ptr<Palette> smartPointerToPalette  = std::make_shared<Palette>(colors, 1, 12, PALETTE_HEIGHT);
    smartPointerToPalette->setPosition({(WINDOW_WIDTH-12*PALETTE_HEIGHT) / 2.0f, 0.0f});
    smartPointerToPalette->select(11,0);
    smartPointerToPalette->setDefaultImagePath(PALETTE_TEXTURE_PATH, PALETTE_SELECTION_INDICATOR_PATH);
    std::shared_ptr<Widget> smartPointerToWidget = dynamic_pointer_cast<Widget>(smartPointerToPalette);
    
    uiElements.push_back(smartPointerToWidget);

}

// It might be more practical to put this code into a separate file in the future.
//

    void processPenDownEvent(int id, int x, int y, int state){
        bool isPenOnWidget = false;
        cout << "Processing Pen Event at " << x <<", " << y << endl;
        cout << "There are currently " << uiElements.size() << endl;
        Point position = {x,y};
        for (vector<shared_ptr<Widget>>::iterator widgit = uiElements.begin(); widgit != uiElements.end(); ++widgit)
        {
            cout << "Pointer is " << (*widgit) << endl;
            cout << (*widgit)->isPointOnWidget(position) << " for widget "<< (*widgit)->getID() << endl;
            cout << "at "<< (*widgit)->getPosition().x << "|" << (*widgit)->getPosition().y << endl;
            if ((*widgit)->isPointOnWidget(position)){ //TODO: What if two widgets overlap? which one receives the click
                (*widgit)->onClick(position);
                isPenOnWidget = true;
            }
        }
        if (!isPenOnWidget){
            switch(currentTool){
                case pencil: drawLine(id, x, y, state); break;
                case eraser: erase(id, x, y, state, eraserRadius); break;
            }
        }
    }

    void processPenUpEvent(int id){
        switch(currentTool){
            case pencil: finishLine(id); break;
            case eraser: finishLine(id); break;
        }
    }

    void drawLine(int id, int x, int y, int state){
        uint r = currentColor.r;
        uint g = currentColor.g;
        uint b = currentColor.b;

        mutex_pens.lock();
        if(pens.find(id) == pens.end())
        {
            struct Pen currentPen;
            currentPen.currentLine.id = id;
            currentPen.alive = 1;
            pens[id] = currentPen;
        }

        pens[id].position.x = x;
        pens[id].position.y = y;
        pens[id].state = (bool) state;

        //cout << "id: " << id << " x: " << x << "; y: " << y << " state: " << state << endl;
        if (state == STATE_DRAW && id == pens[id].currentLine.id)
        {
            pens[id].currentLine.color = {r, g, b};
        }
        //pens[id].currentLine.id = currentId;

        bool inDocument = document.isPointInDocument(x, y);

        mutex_lines.lock();
        if (document.alive && inDocument && !pens[id].wasInDocument)
        {
            lines.push_back(pens[id].currentLine);
            pens[id].currentLine.coords.clear();
            if(state == STATE_DRAW) pens[id].currentLine.coords.push_back({x, y});
            pens[id].wasInDocument = true;
        }
        else if (document.alive && !inDocument && pens[id].wasInDocument)
        {
            documentLines.push_back(pens[id].currentLine);
            pens[id].currentLine.coords.clear();
            if(state == STATE_DRAW) pens[id].currentLine.coords.push_back({x, y});
            pens[id].wasInDocument = false;
        }
        else
        {
            if(state == STATE_DRAW)
            {
                pens[id].currentLine.coords.push_back({x, y});
            }
        }

        pathGame.update(x, y, state);

        mutex_lines.unlock();
        mutex_pens.unlock();
    }

    void erase(int id, float x, float y, int state, int radius){
        mutex_pens.lock();
        if(pens.find(id) == pens.end())
        {
            struct Pen currentPen;
            currentPen.currentLine.id = id;
            currentPen.alive = 1;
            pens[id] = currentPen;
        }

        pens[id].position.x = x;
        pens[id].position.y = y;
        pens[id].state = 1;

        eraserIndicatorRadius = radius;
        mutex_lines.lock();
        //fflush(stdout);
        //NEW BEHAVIOR: Lines are selected with a circle
        //Possible Optimization: Compute Bounding boxes of lines and check collision with eraser before looping over every point

        if (state == STATE_DRAW)
        {
            showEraserIndicator = true;
            //re-implementation of dubious code that is hopefully more readable (and actually works)

            // keeps track of new line segment that are created when an existing line is split
            vector<Line> newLines;
            //loop over every line saved.
            for (vector<Line>::iterator lineit = lines.begin(); lineit != lines.end();)
            {

                //when iterating through the vector of points, we will alternate between sections of points that will remain and sections of points that will be erased.
                //this bool keeps track of which type of section we are currently in.
                bool inCollidingSpan = false;

                vector<Point> points = lineit->coords;

                //during the iteration, we take action at the points where we change from an erased section to an unerased section.
                //if we change from an unerased section to an erased section, we chop everything before off and save it as a new line.
                //if we change from an erased section to an unerased section, we chop everything before off and discard it.
                for(vector<Point>::iterator pointit = lineit->coords.begin(); pointit != lineit->coords.end();++pointit)
                {
                    if(getDistance(pointit->x, pointit->y, x, y) <= radius) // is the current point colliding with the eraser?
                    {

                        if(!inCollidingSpan) //and are we changing from a non-erased section to an erased section?
                        {
                            inCollidingSpan = true;
                            if (pointit != lineit->coords.begin())
                            {
                                Line newLineSegment; //put all points that came before into a new line
                                newLineSegment.color = lineit->color;
                                newLineSegment.alive = lineit->alive;
                                newLineSegment.id = lineit->id; //NOTE: THIS WILL BREAK any future additions that rely on the ids of lines being unique
                                //TODO: find a better way of assigning new line ids
                                copy(lineit->coords.begin(), pointit, back_inserter(newLineSegment.coords)); //copy points from existing line to new line
                                pointit = lineit->coords.erase(lineit->coords.begin(), pointit); //and then erase these points from the original line
                                if (newLineSegment.coords.size() >= MIN_NON_ERASE_LINE_POINTS)
                                {
                                    newLines.push_back(newLineSegment);
                                }
                            }

                        }
                    }
                    else
                    {
                        if(inCollidingSpan)//are we changing from an erased section to an unerased section?
                        {
                            inCollidingSpan = false;
                            pointit = lineit->coords.erase(lineit->coords.begin(), pointit); //erase all points in the erased section from the line
                        }
                    }
                }

                //if this algorithm is finished and there are no points remaining in the line, we can discard it
                if (lineit->coords.size() < MIN_NON_ERASE_LINE_POINTS || inCollidingSpan)
                {
                    lineit = lines.erase(lineit);
                }
                else
                {
                    //const Line restLine = *lineit;
                    //lines.erase(lineit);
                    //lines.insert(lineit, restLine);
                    ++lineit; //increment iterator
                }
            }


            // save all new line segments
            for(vector<Line>::iterator lineit = newLines.begin(); lineit != newLines.end(); ++lineit)
            {
                lines.push_back(*lineit);
            }

            //TODO: implement the same for document lines
            for (vector<Line>::iterator lineit = documentLines.begin(); lineit != documentLines.end(); )
            {
                vector<Point> colliding_points = collideLineWithCircle(lineit->coords, x, y, radius);
                if (colliding_points.size() > 0)
                {
                    lineit=lines.erase(lineit);
                }
                else
                    ++lineit;
            }
        }

        
        
        mutex_lines.unlock();
        
        mutex_pens.unlock();
        
    }

    void finishLine(int id){
        mutex_pens.lock();
        mutex_lines.lock();
        bool inDocument = document.isPointInDocument(pens[id].position.x, pens[id].position.y);

        if (inDocument)
        {
            documentLines.push_back(pens[id].currentLine);
        }
        else
        {
            lines.push_back(pens[id].currentLine);
        }

        pens[id].currentLine.coords.clear();
        pens[id].alive = 0;

        pens.erase(id);
        mutex_lines.unlock();
        mutex_pens.unlock();
        
    }
    void finishErase(int id){
        showEraserIndicator = false;
        pens[id].alive = 0;
        pens[id].state = 0;

        pens.erase(id);
        mutex_pens.unlock();
    }

    void clearCanvas();


//
//

int main(int argc, char* argv[]) 
{
    signal(SIGINT, onExit);
    signal(SIGPIPE, onBrokenPipe);

    srand(time(NULL));


    if(argc > 1)
    {
        if(COMMUNICATION_MODE == MODE_FIFO)
        {
            fifo_path = argv[1];
            if(!init_fifo()) return 1;
        }
        else
        {
            uds_path = argv[1];
            if(!init_uds()) return 1;
        }
    }
    if(argc > 2)
    {
        participantId = atoi(argv[2]);
        pathGame.participant_id = participantId;
    }

    mkdir(SCREENSHOT_PATH, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    char participantPath[100];
    sprintf(participantPath, "%s%02d/", SCREENSHOT_PATH, participantId);
    mkdir(participantPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    SCREENSHOT_PATH = participantPath;
    cout << SCREENSHOT_PATH << endl;

    SDL_Init(SDL_INIT_VIDEO);

    IMG_Init(IMG_INIT_PNG);

    if ( TTF_Init() < 0 ) {
        cout << "Error initializing SDL_ttf: " << TTF_GetError() << endl;
    }

    font = TTF_OpenFont("font.ttf", FONT_SIZE);

    textSurface = TTF_RenderText_Solid( font, "Hello World!", textColor );

    crossesSurface = loadSurface(CROSSES_PATH);
    imageSurface = loadSurface(IMAGE_PATH);

    SDL_SetHint(SDL_HINT_VIDEO_MINIMIZE_ON_FOCUS_LOSS, "0");

    SDL_Window* window = SDL_CreateWindow(__FILE__, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_FULLSCREEN);
    renderer = SDL_CreateRenderer(window, -1, 0);

    crossesTexture = SDL_CreateTextureFromSurface( renderer, crossesSurface );
    imageTexture = SDL_CreateTextureFromSurface( renderer, imageSurface );

    initPalette();
    //preloadTextures(renderer);
    for (vector<std::shared_ptr<Widget>>::iterator widgit = uiElements.begin(); widgit != uiElements.end(); ++widgit)
    {
        cout << "HAS WIDGET" << endl;
        (*widgit)->loadTexture();
    }

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);


    bool quit = false;
    SDL_Event event;

    loadPhrases();

    while(!quit)
    {
        SDL_PollEvent(&event);

        switch (event.type)
        {
            case SDL_KEYDOWN:
                switch(event.key.keysym.sym)
                {
                    case SDLK_q:
                        break;
                    case SDLK_ESCAPE:
                        saveImage();
                        quit = true;
                        break;
                    case SDLK_w:
                        saveImage();
                        clearScreen();
                        currentMode = draw;
                        break;
                    case SDLK_e:
                        saveImage();
                        currentMode = phrase;
                        nextPhrase();
                        currentTextSize = 0;
                        clearScreen();
                        break;
                    case SDLK_r:
                        saveImage();
                        clearScreen();
                        currentMode = cross;
                        break;
                    case SDLK_u:
                        currentMode = image;
                        break;
                    case SDLK_t:
                        currentMode = latency;
                        break;
                    case SDLK_z:
                        SHOW_PARTICLES = !SHOW_PARTICLES;
                        break;
                    case SDLK_s:
                        saveImage();
                        break;
                    case SDLK_PAGEUP:
                        saveImage();
                        break;
                    case SDLK_a:
                        showBrokenPipeIndicator = false;
                        break;
                    case SDLK_d:
                        saveImage();
                        clearScreen();
                        pathGame.reset();
                        currentMode = path;
                        break;
                    case SDLK_SPACE:
                        saveImage();
                        clearScreen();
                        if(currentMode == phrase)
                        {
                            nextPhrase();
                        }
                        break;
                }
                break;
            case SDL_QUIT:
                quit = true;
                break;
                // TODO input handling code goes here
        }

        if(currentMode == latency) renderLatencyTest(renderer);
        else render(renderer);

        usleep(1000);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    TTF_Quit();

    return 0;
}
