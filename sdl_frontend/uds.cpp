#include "main.h"
#include "uds.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h> 
#include <sys/stat.h>
#include <unistd.h>


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
    const int buffer_length = 20;
    char buffer[buffer_length];
    string residual = "";

    int id, x, y, state;
    int size;

    while(1)
    {
        size = recv(client_socket, buffer, buffer_length-1, MSG_WAITALL);
        //size = read(client_socket, buffer, buffer_length-1);
	
	// print raw messages limited to buffer size
	//cout << buffer << endl;

	//cout << size << " -- " << buffer << endl;
        if(size > 0)
        {
            vector<string> substrings = split(residual + buffer, "|");
            int i = 0;

	    residual = "";

            for (auto message : substrings)
            {
                i++;
                int result = parseMessage((char *) message.c_str());

		//if (result == 0)
		//{
		//	cout << "failed " << i << " " << substrings.size() << " " << message << endl;
		//}

                if (result == 0 && i == substrings.size())
                {
                    residual = message;
                }

		if (result == 1)
		{
			cout << message << endl;
		}
            }
        }
        //send(client_socket, "ok", 2, 0);
        //usleep(500);
    }
}
