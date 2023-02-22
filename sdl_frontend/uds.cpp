#include "main.h"
#include "uds.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h> 
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>


int init_uds()
{
    socklen_t addrlen;
    ssize_t size;
    struct sockaddr_un address;
    const int y = 1;
    if((server_socket=socket (AF_LOCAL, SOCK_STREAM, 0)) > 0)
        printf ("created socket\n");
    unlink(uds_path);
    address.sun_family = AF_LOCAL;
    strcpy(address.sun_path, uds_path);
    if (bind ( server_socket,
                (struct sockaddr *) &address,
                sizeof (address)) != 0) {
        printf( "port is not free!\n");
    }
    listen (server_socket, 5);
    addrlen = sizeof (struct sockaddr_in);
    while (1) {
        client_socket = accept ( server_socket,
                (struct sockaddr *) &address,
                &addrlen );
        if (client_socket > 0)
        {
            printf ("client connected\n");
            break;
        }
    }

    pthread_create(&uds_thread, NULL, handle_uds, NULL); 

    return 1;
}

void *handle_uds(void *args)
{
    while(1)
    {
        // receive header message containing the length of the message
        char header[4];
        int header_size = recv(client_socket, header, 4, MSG_WAITALL);

        // create buffer with appropriate size
        int buffer_length = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
        char buffer[buffer_length + 1];

        // receive actual message
        int size = recv(client_socket, buffer, buffer_length, MSG_WAITALL);

        // add terminating character to avoid junk in the end
        buffer[buffer_length] = '\0';

        // hand over to message parser
        int result = parseMessage(buffer);

        // for debugging
        //if(result != 1)
        //{
        //    cout << buffer << endl;
        //}
    }
}
