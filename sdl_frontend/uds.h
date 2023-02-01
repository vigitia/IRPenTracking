#ifndef __UDS_H__
#define __UDS_H__

#define BUF 1024
#define UDS_FILE "/tmp/sock.uds"

inline int uds_fd = -1;    // path to uds for remotely controlled delay times
inline char* uds_path;
inline pthread_t uds_thread; 

inline int server_socket, client_socket;

int init_uds();
void *handle_uds(void *args);

#endif
