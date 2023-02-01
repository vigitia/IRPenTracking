#ifndef __FIFO_H__
#define __FIFO_H__

inline int fifo_fd = -1;    // path to FIFO for remotely controlled delay times
inline char* fifo_path;
inline pthread_t fifo_thread; 

int init_fifo();
void *handle_fifo(void *args);

#endif
