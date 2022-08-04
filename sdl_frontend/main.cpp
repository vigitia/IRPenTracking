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
#include <vector>
#include <map>
#include <ctime>

#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080

#define HOVER_INDICATOR_COLOR 0xFF00FFFF
#define SHOW_HOVER_INDICATOR 1

#define STATE_HOVER 0
#define STATE_DRAW 1

const char* SCREENSHOT_PATH = "screenshots/";
const char* PHRASES_PATH = "../phrase_set/phrases.txt";

using namespace std;

vector<string> phrases;
string currentPhrase;

enum Modes {
    draw,
    phrase
};

Modes currentMode = draw;

int fifo_fd = -1;    // path to FIFO for remotely controlled delay times
char* fifo_path;
pthread_t fifo_thread; 
    
SDL_Renderer* renderer;

map<int, vector<SDL_Point>> lines;
vector<SDL_Point> currentLine;

int currentId = 0;

int participantId = 0;

int currentX, currentY = 0;
int currentState = 0;

void *handle_fifo(void *args)
{
    char buffer[80];

    while(1)
    {
        // open the FIFO - this call blocks the thread until someone writes to the FIFO
        fifo_fd = open(fifo_path, O_RDONLY);

        int id, x, y, state;

        if(read(fifo_fd, buffer, 80) <= 0) continue; // read the FIFO's content into a buffer and skip setting the variables if an error occurs

        // parse new values from the FIFO
        // only set the delay times if all four values could be read correctly
        if(sscanf(buffer, "%d %d %d %d ", &id, &x, &y, &state) == 3)
        {
            cout << "id: " << id << " x: " << x << "; y: " << y << " state: " << state << endl;
            currentX = x;
            currentY = y;
            currentState = state;
            if(id != currentId)
            {
                lines[currentId] = currentLine;
                currentId = id;
                currentLine.clear();
            }
            else
            {
                if(state == STATE_DRAW)
                {
                    currentLine.push_back({x, y});
                }
            }
        }
        else
        {
            cout << "could not read input " << buffer << endl;
        }


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
    // end inter process communication
    pthread_cancel(fifo_thread);
    unlink(fifo_path);

    exit(EXIT_SUCCESS);
}

void renderLine(SDL_Renderer *rend, vector<SDL_Point> *line)
{
    if(line->size() > 1)
    {
        SDL_Point point_array[line->size()];
        for(int i = 0; i < line->size(); i++)
        {
            point_array[i] = line->at(i);
        }
        SDL_SetRenderDrawColor(rend, 255, 255, 255, 255);
        SDL_RenderDrawLines(rend, point_array, line->size());
    }
}

void clearScreen()
{
    //for (auto const& entry : lines)
    //{
    //    entry.second.clear();
    //}
    lines.clear();
    currentLine.clear();
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

void saveImage()
{ 
    char filename[120];
    sprintf(filename, "%s%d_%s.bmp", SCREENSHOT_PATH, participantId, currentDateTime().c_str());

    const Uint32 format = SDL_PIXELFORMAT_ARGB8888;
 
    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, WINDOW_WIDTH, WINDOW_HEIGHT, 32, format);
    SDL_RenderReadPixels(renderer, NULL, format, surface->pixels, surface->pitch);
    SDL_SaveBMP(surface, filename);
    SDL_FreeSurface(surface);
}

void nextPhrase()
{

}

int main(int argc, char* argv[]) 
{
    signal(SIGINT, onExit);

    if(argc > 1)
    {
        fifo_path = argv[1];
        if(!init_fifo()) return 1;
    }
    if(argc > 2)
    {
        participantId = atoi(argv[2]);
    }

    //SDL_Init(SDL_INIT_EVERYTHING); // maybe we have to reduce this?
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow(__FILE__, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_FULLSCREEN);
    renderer = SDL_CreateRenderer(window, -1, 0);

    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);

    bool quit = false;
    SDL_Event event;


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
                        currentMode = draw;
                        break;
                    case SDLK_e:
                        currentMode = phrase;
                        break;
                    case SDLK_s:
                        saveImage();
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

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);    //

        for (auto const& entry : lines)
        {
            vector<SDL_Point> line = entry.second;
            renderLine(renderer, &line);
        }

        if(currentLine.size() > 1)
        {
            SDL_Point point_array[currentLine.size()];
            for(int i = 0; i < currentLine.size(); i++)
            {
                point_array[i] = currentLine.at(i);
            }
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_RenderDrawLines(renderer, point_array, currentLine.size());
        }

        if(SHOW_HOVER_INDICATOR && currentState == STATE_DRAW)
        {
            filledCircleColor(renderer, currentX, currentY, 3, HOVER_INDICATOR_COLOR);
        }

        SDL_RenderPresent(renderer);  // their sequence appears to not matter

        usleep(1000);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
