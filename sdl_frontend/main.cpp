#include "main.h"
#include "uds.h"
#include "document.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h> 
#include <errno.h>
//#include <sys/types.h>

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

#include "particle.h"


using namespace std;

vector<string> phrases;
vector<int> usedPhrases;
string currentPhrase;
int currentTextSize = 0;

bool SHOW_PARTICLES = false;
vector<Particle> particles;

Modes currentMode = draw;

SDL_Renderer* renderer;

SDL_Point pointToSDL(Point p)
{
	SDL_Point result = {(int) p.x, (int) p.y};
	return result;
}



//document = Document();

SDL_Texture* textTexture;
TTF_Font* font;
SDL_Color textColor = { 255, 255, 255 };

uint32_t highlightColor = 0x9900FFFF;

SDL_Surface* textSurface;
SDL_Surface* crossesSurface;
SDL_Texture* crossesTexture;
SDL_Surface* imageSurface;
SDL_Texture* imageTexture;

bool isSaving = false;

bool showBrokenPipeIndicator = false;


void clearScreen()
{
    //for (auto const& entry : lines)
    //{
    //    entry.second.clear();
    //}
    lines.clear();
    documentLines.clear();
    currentLine.coords.clear();
}


void *handle_fifo(void *args)
{
    char buffer[80];

    while(1)
    {
        // open the FIFO - this call blocks the thread until someone writes to the FIFO
        fifo_fd = open(fifo_path, O_RDONLY);


        if(read(fifo_fd, buffer, 80) <= 0) continue; // read the FIFO's content into a buffer and skip setting the variables if an error occurs

        parseMessage(buffer);

        close(fifo_fd);
        usleep(500);
    }
}

int init_fifo()
{
    unlink(fifo_path); // unlink the FIFO if it already exists
    umask(0); // needed for permissions, I have no idea what this exactly does
    if(mkfifo(fifo_path, 0666) == -1) return 0; // create the FIFO

    // create a thread reading the FIFO and adjusting the delay times
    pthread_create(&fifo_thread, NULL, handle_fifo, NULL); 

    return 1;
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

// https://stackoverflow.com/questions/997946/how-to-get-current-time-and-date-in-c
const string currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);

    return buf;
}

void loadPhrases()
{
    string line;
    ifstream phraseFile(PHRASES_PATH);

    if(phraseFile.is_open())
    {
        while(getline(phraseFile, line))
        {
            line.pop_back();
            phrases.push_back(line);
        }
        phraseFile.close();
    }
}

void nextPhrase()
{
    bool newPhraseFound = false;
    int phraseIndex = 0;

    while(!newPhraseFound)
    {
        newPhraseFound = true;
        phraseIndex = rand() % phrases.size();

        for (auto const& index : usedPhrases)
        {
            if(index == phraseIndex)
            {
                newPhraseFound = false;
                break;
            }
        }
    }

    usedPhrases.push_back(phraseIndex);
    currentPhrase = phrases.at(phraseIndex);
    //cout << phraseIndex << " " << currentPhrase << endl;

    currentTextSize = (currentTextSize + 1) % 3;

    textSurface = TTF_RenderText_Solid( font, currentPhrase.c_str(), textColor );
    textTexture = SDL_CreateTextureFromSurface( renderer, textSurface );
}

void renderParticles(SDL_Renderer* renderer)
{
    if(currentState == STATE_DRAW)
    {
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

    if(currentLine.coords.size() > 1)
    {
        SDL_Point point_array[currentLine.coords.size()];
        for(int i = 0; i < currentLine.coords.size(); i++)
        {
            point_array[i] = pointToSDL(currentLine.coords.at(i));
        }
        SDL_SetRenderDrawColor(renderer, currentLine.color.r, currentLine.color.g, currentLine.color.b, 255);
        SDL_RenderDrawLines(renderer, point_array, currentLine.coords.size());
    }
}

void renderHoverIndicator(SDL_Renderer* renderer)
{
    filledCircleColor(renderer, currentX, currentY, 3, HOVER_INDICATOR_COLOR);
}

void renderLatencyTest(SDL_Renderer* renderer)
{
    if(currentState == STATE_DRAW)
    {
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    }
    else
    {
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
    }
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);
}

void renderBrokenPipeIndicator(SDL_Renderer* renderer)
{
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_Rect brokenPipeIndicator = { 0, 0, 20, 20 };

    SDL_RenderFillRect(renderer, &brokenPipeIndicator);
}

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
    if(SHOW_HOVER_INDICATOR && currentState == STATE_HOVER && currentMode != cross) renderHoverIndicator(renderer);
    if(SHOW_PARTICLES) renderParticles(renderer);
    if(showBrokenPipeIndicator) renderBrokenPipeIndicator(renderer);

    if(!isSaving) SDL_RenderPresent(renderer);
}

// https://lazyfoo.net/tutorials/SDL/06_extension_libraries_and_loading_other_image_formats/index2.php
SDL_Surface* loadSurface(string path)
{
    //The final optimized image
    SDL_Surface* optimizedSurface = NULL;

    //Load image at specified path
    SDL_Surface* loadedSurface = IMG_Load( path.c_str() );
    //if( loadedSurface == NULL )
    //{
    //    printf( "Unable to load image %s! SDL_image Error: %s\n", path.c_str(), IMG_GetError() );
    //}
    //else
    //{
    //    //Convert surface to screen format
    //    optimizedSurface = SDL_ConvertSurface( loadedSurface, gScreenSurface->format, 0 );
    //    if( optimizedSurface == NULL )
    //    {
    //        printf( "Unable to optimize image %s! SDL Error: %s\n", path.c_str(), SDL_GetError() );
    //    }

    //    //Get rid of old loaded surface
    //    SDL_FreeSurface( loadedSurface );
    //}
    //
    //return optimizedSurface;

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
    //SDL_SaveBMP(surface, filename);
    IMG_SavePNG(surface, filename);

    isSaving = false;
}

int main(int argc, char* argv[]) 
{
    signal(SIGINT, onExit);
    signal(SIGPIPE, onBrokenPipe);

    srand(time(NULL));

    if(argc > 1)
    {
        fifo_path = argv[1];
        if(COMMUNICATION_MODE == MODE_FIFO)
        {
            if(!init_fifo()) return 1;
        }
        else
        {
            if(!init_uds()) return 1;
        }
    }
    if(argc > 2)
    {
        participantId = atoi(argv[2]);
    }

    mkdir(SCREENSHOT_PATH, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    char participantPath[100];
    sprintf(participantPath, "%s%02d/", SCREENSHOT_PATH, participantId);
    mkdir(participantPath, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    SCREENSHOT_PATH = participantPath;
    cout << SCREENSHOT_PATH << endl;

    //SDL_Init(SDL_INIT_EVERYTHING); // maybe we have to reduce this?
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
                    case SDLK_a:
                        showBrokenPipeIndicator = false;
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
