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

    preloadTextures(renderer);

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
