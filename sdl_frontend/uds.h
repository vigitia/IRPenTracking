#ifndef __UDS_H__
#define __UDS_H__

#define BUF 1024
#define UDS_FILE "/tmp/sock.uds"

inline int fifo_fd = -1;    // path to FIFO for remotely controlled delay times
inline char* fifo_path;

inline int server_socket, client_socket;

inline pthread_t fifo_thread; 
inline pthread_t uds_thread; 

int init_uds();
void *handle_uds(void *args);

#endif
