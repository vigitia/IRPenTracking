#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h> 
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include <iostream>
#include <SDL2/SDL.h>
#include <vector>

using namespace std;

int fifo_fd = -1;    // path to FIFO for remotely controlled delay times
char* fifo_path;
pthread_t fifo_thread; 
    
SDL_Renderer* renderer;

vector<SDL_Point> points;

void *handle_fifo(void *args)
{
    char buffer[80];

    while(1)
    {
        // open the FIFO - this call blocks the thread until someone writes to the FIFO
        fifo_fd = open(fifo_path, O_RDONLY);

        int x, y;

        if(read(fifo_fd, buffer, 80) <= 0) continue; // read the FIFO's content into a buffer and skip setting the variables if an error occurs

        // parse new values from the FIFO
        // only set the delay times if all four values could be read correctly
        if(sscanf(buffer, "%d %d", &x, &y) == 2)
        {
            cout << "x: " << x << "; y: " << y << endl;
            points.push_back({x, y});
        }
        else
        {
            cout << "could not read input " << buffer << endl;
        }


        close(fifo_fd);
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


int main(int argc, char* argv[]) 
{
    signal(SIGINT, onExit);

    if(argc > 1)
    {
        fifo_path = argv[1];
        if(!init_fifo()) return 1;
    }

    SDL_Init(SDL_INIT_EVERYTHING); // maybe we have to reduce this?

    SDL_Window* window = SDL_CreateWindow(__FILE__, 0, 0, 3840, 2160, SDL_WINDOW_FULLSCREEN);
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
            case SDL_QUIT:
                quit = true;
                break;
            // TODO input handling code goes here
        }

        if(points.size() > 1)
        {

            SDL_Point point_array[points.size()];

            for(int i = 0; i < points.size(); i++)
            {
                point_array[i] = points.at(i);
                //cout << point_array[i].x << " " << point_array[i].y << " " << points.size() << endl;
            }


            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_RenderDrawLines(renderer, point_array, points.size());
        }
        //SDL_Point point_array[3] = {
        //    {100, 200},
        //    {150, 200},
        //    {150, 400}
        //};

        //SDL_RenderDrawLines(renderer, point_array, points.size());
        //SDL_RenderDrawLine(renderer, 100, 100, 200, 200);

        SDL_RenderPresent(renderer);  // their sequence appears to not matter
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);    //
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
