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


using namespace std;

int fifo_fd = -1;    // path to FIFO for remotely controlled delay times
char* fifo_path;
pthread_t fifo_thread; 

void *handle_fifo(void *args)
{
    char buffer[80];

    while(1)
    {
        // open the FIFO - this call blocks the thread until someone writes to the FIFO
        fifo_fd = open(fifo_path, O_RDONLY);

        if(read(fifo_fd, buffer, 80) <= 0) continue; // read the FIFO's content into a buffer and skip setting the variables if an error occurs

        /*
        // parse new values from the FIFO
        // only set the delay times if all four values could be read correctly
        if(sscanf(buffer, "%d %d %d %d", &buffer_min_delay_click, &buffer_max_delay_click, &buffer_min_delay_move, &buffer_max_delay_move) == 4)
        {
            // set delay times
            min_delay_click = buffer_min_delay_click;
            max_delay_click = buffer_max_delay_click;
            min_delay_move = buffer_min_delay_move;
            max_delay_move = buffer_max_delay_move;

            // make sure max >= min
            if(max_delay_click < min_delay_click) max_delay_click = min_delay_click;
            if(max_delay_move < min_delay_move) max_delay_move = min_delay_move;

            if(DEBUG) printf("set new values: %d %d %d %d\n", min_delay_click, max_delay_click, min_delay_move, max_delay_move);
        }
        else
        {
            if(DEBUG) printf("could not set new delays - bad data structure\n");
        }
        */

        cout << buffer << endl;

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

    while(1)
    {

    }

    return 0;
}
