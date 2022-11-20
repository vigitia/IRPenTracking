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
#include <sys/stat.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "particle.h"

#define MODE_FIFO 0
#define MODE_UDS 1

#define COMMUNICATION_MODE MODE_UDS

#define MODE_1080 0
#define MODE_4K 1

#define MODE MODE_4K

#if MODE == MODE_1080
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080
#define CROSSES_PATH "evaluation/crosses_dots_1080.png"
#else
#define WINDOW_WIDTH 3840
#define WINDOW_HEIGHT 2160
#define CROSSES_PATH "evaluation/crosses_dots_4k.png"
#endif

#define IMAGE_PATH "image.png"

#define HOVER_INDICATOR_COLOR 0xFF00FFFF
#define SHOW_HOVER_INDICATOR 1

#define SHOW_LINES 1

#define STATE_HOVER 0
#define STATE_DRAW 1

#define FONT_SIZE 42

const int TEXT_BOX_WIDTH = (int)(WINDOW_WIDTH * 0.7);
const int TEXT_BOX_HEIGHT_SMALL = (int)(WINDOW_HEIGHT * 0.05);
const int TEXT_BOX_HEIGHT_MEDIUM = (int)(WINDOW_HEIGHT * 0.075);
const int TEXT_BOX_HEIGHT_LARGE = (int)(WINDOW_HEIGHT * 0.1);
const int TEXTBOX_OFFSET = (int)(WINDOW_HEIGHT * 0.1);

char* SCREENSHOT_PATH = "screenshots/";
const char* PHRASES_PATH = "evaluation/phrases.txt";

using namespace std;

vector<string> phrases;
vector<int> usedPhrases;
string currentPhrase;
int currentTextSize = 0;

bool SHOW_PARTICLES = false;
vector<Particle> particles;

enum Modes {
    draw,
    phrase,
    cross,
    save,
    latency,
    particle,
    image
};

Modes currentMode = draw;

int fifo_fd = -1;    // path to FIFO for remotely controlled delay times
char* fifo_path;
pthread_t fifo_thread; 

pthread_t uds_thread; 

SDL_Renderer* renderer;

struct Line {
    int id;
    SDL_Color color;
    vector<SDL_Point*> coords;
    bool alive;
};

vector<Line> lines;
vector<Line> documentLines;
Line currentLine;

struct Poly {
    int id;
    short int x[4];
    short int y[4];
    bool alive;
};

// https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library
bool is_on_right_side(int x, int y, SDL_Point xy0, SDL_Point xy1)
{
    int x0 = xy0.x;
    int y0 = xy0.y;
    int x1 = xy1.x;
    int y1 = xy1.y;
    float a = float(y1 - y0);
    float b = float(x0 - x1);
    float c = - a * x0 - b * y0;
    return a * x + b * y + c >= 0;
}

class Document {
    public:
        Document(SDL_Point top_left, SDL_Point top_right, SDL_Point bottom_left, SDL_Point bottom_right);
        Document();
        SDL_Point top_left, top_right, bottom_left, bottom_right;
        bool isPointInDocument(int x, int y);
        bool alive = false;
        void setPoints(SDL_Point top_left, SDL_Point top_right, SDL_Point bottom_left, SDL_Point bottom_right);

};

Document::Document(SDL_Point top_left, SDL_Point top_right, SDL_Point bottom_left, SDL_Point bottom_right)
{
    this->top_left = top_left;
    this->top_right = top_right;
    this->bottom_left = bottom_left;
    this->bottom_right = bottom_right;
    alive = true;
}

Document::Document()
{
    alive = false;
}

void Document::setPoints(SDL_Point top_left, SDL_Point top_right, SDL_Point bottom_left, SDL_Point bottom_right)
{
    this->top_left = top_left;
    this->top_right = top_right;
    this->bottom_left = bottom_left;
    this->bottom_right = bottom_right;
}

// https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library
bool Document::isPointInDocument(int x, int y)
{
    if (!alive) return false;

    bool is_above = is_on_right_side(x, y, top_left, top_right);
    bool is_below = !is_on_right_side(x, y, bottom_left, bottom_right);
    bool is_left = !is_on_right_side(x, y, top_left, bottom_left);
    bool is_right = is_on_right_side(x, y, top_right, bottom_right);
     
    return !(is_above || is_below || is_left || is_right) || (is_above && is_below && is_left && is_right);
}

Document document = Document();

bool wasInDocument = false;

map<int, struct Poly> rects;

int currentId = 0;

int participantId = 0;

int currentX, currentY = 0;
int currentState = 0;

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

#define BUF 1024
#define UDS_FILE "/tmp/sock.uds"

int server_socket, client_socket;

void clearScreen()
{
    //for (auto const& entry : lines)
    //{
    //    entry.second.clear();
    //}
    lines.clear();
    currentLine.coords.clear();
}

// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}


vector<SDL_Point> parseAppendLine(char* buffer)
{
    vector<SDL_Point> points;

    vector<string> substrings = split(buffer, ";");
    int i = 0;

    int x, y;
    for (auto message : substrings)
    {
        i++;
        if (sscanf(message.c_str(), "%d,%d", &x, &y) == 2)
        {
            //cout << x << ", " << y << endl;
            points.push_back({x, y});
        }
    }
    //cout << "num coords: " << i << endl;

    return points;
}

SDL_Point multiplyPointMatrix(SDL_Point *point, float matrix[3][3])
{
    int x = point->x;
    int y = point->y;

    float result_x = matrix[0][0] * x + matrix[1][0] * y + matrix[2][0];
    float result_y = matrix[0][1] * x + matrix[1][1] * y + matrix[2][1];
    float result_z = matrix[0][2] * x + matrix[1][2] * y + matrix[2][2];

    result_x /= result_z;
    result_y /= result_z;

    SDL_Point result = {(int)result_x, (int)result_y};

    return result;
}

int parseMessage(char* buffer)
{
    //printf(buffer);
    //printf("\n--------\n");
    //fflush(stdout);

    if(buffer[0] == 'l')
    {
        int id, x, y, state;
        unsigned int r;
        unsigned int g;
        unsigned int b;
        // parse new values from the FIFO
        // only set the delay times if all four values could be read correctly
        if(sscanf(buffer, "l %d %u %u %u %d %d %d ", &id, &r, &g, &b, &x, &y, &state) == 7)
        {
            //cout << "id: " << id << " x: " << x << "; y: " << y << " state: " << state << endl;
            currentX = x;
            currentY = y;
            currentState = state;
            currentLine.color = {r, g, b};
            currentLine.id = currentId;

            bool inDocument = document.isPointInDocument(x, y);

            if (inDocument && !wasInDocument)
            {
                lines.push_back(currentLine);
                currentLine.coords.clear();
		SDL_Point pnt = {x, y};
                currentLine.coords.push_back(&pnt);
                wasInDocument = true;
            }
            else if (!inDocument && wasInDocument)
            {
                documentLines.push_back(currentLine);
                currentLine.coords.clear();
		SDL_Point pnt = {x, y};
                currentLine.coords.push_back(&pnt);
                wasInDocument = false;
            }
            else
            {
                if(id != currentId)
                {
                    if (inDocument)
                    {
                        documentLines.push_back(currentLine);
                    }
                    else
                    {
                        lines.push_back(currentLine);
                    }

                    currentId = id;
                    currentLine.coords.clear();
                }
                else
                {
                    if(state == STATE_DRAW)
                    {
		SDL_Point pnt = {x, y};
                currentLine.coords.push_back(&pnt);
                    }
                } 
            }

            return 1; 
        }
        //else
        //{
        //    cout << "could not read input " << buffer << endl;
        //	return 0;
        //}
    }
    else if(buffer[0] == 'e')
    {
        int id, state;
        int x1, x2, x3, x4;
        int y1, y2, y3, y4;
        if(sscanf(buffer, "e %d %d %d %d %d %d %d %d %d \n ", &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &state) == 9)
        {
            document.alive = (bool) state;

            document.top_left = {x1, y1};
            document.top_right = {x2, y2};
            document.bottom_right = {x3, y3};
            document.bottom_left = {x4, y4};
        }
    }
    else if(buffer[0] == 'm')
    {
        float matrix[3][3];

        if(sscanf(buffer, "m %f %f %f %f %f %f %f %f %f \n ", &matrix[0][0], &matrix[1][0], &matrix[2][0], &matrix[0][1], &matrix[1][1], &matrix[2][1], &matrix[0][2], &matrix[1][2], &matrix[2][2]) == 9)
        {
		int check = 0;
            for (auto line : documentLines)
            {

		    for (int i = 0; i < line.coords.size(); i++)
		    {
			SDL_Point *point = line.coords.at(i);
			if (check == 0) cout << "before " << point->x << endl;
			SDL_Point pnt = multiplyPointMatrix(point, matrix);
			point->x = pnt.x;
			point->y = pnt.y;
			if (check == 0) cout << "after " << point->x << endl;
			check = 1;
		    }
                //for (auto point : line.coords)
                //{
		//	if (check == 0) cout << "before " << point.x << endl;
                //    SDL_Point pnt = multiplyPointMatrix(point, matrix);
		//    point.x = pnt.x;
		//    point.y = pnt.y;
		//	if (check == 0) cout << "after " << point.x << endl;
		//	check = 1;
                //}
            }
        }
    }
    else if(buffer[0] == 'r')
    {
        int id, state;
        int x1, x2, x3, x4;
        int y1, y2, y3, y4;
        // parse new values from the FIFO
        // only set the delay times if all four values could be read correctly
        if(sscanf(buffer, "r %d %d %d %d %d %d %d %d %d %d \n ", &id, &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &state) == 10)
        {
            //cout << buffer << endl;
            struct Poly poly;

            poly.id = id;
            poly.alive = state;
            poly.x[0] = x1;
            poly.x[1] = x2;
            poly.x[2] = x3;
            poly.x[3] = x4;
            poly.y[0] = y1;
            poly.y[1] = y2;
            poly.y[2] = y3;
            poly.y[3] = y4;

            // new rect
            if(rects.find(id) == rects.end())
            {
                rects[id] = poly;
            }
            else
            {
                if(state == 0)
                {
                    rects.erase(id);
                }
                else
                {
                    rects[id].x[0] = x1;
                    rects[id].x[1] = x2;
                    rects[id].x[2] = x3;
                    rects[id].x[3] = x4;
                    rects[id].y[0] = y1;
                    rects[id].y[1] = y2;
                    rects[id].y[2] = y3;
                    rects[id].y[3] = y4;
                }
            }

            return 1;
        }
    }
    else if(buffer[0] == 'c')
    {
        rects.clear();
        return 1;
    }
    else if(buffer[0] == 'x')
    {
        clearScreen();
        return 1;
    }
    else if(buffer[0] == 'd')
    {
        int id;
        if(sscanf(buffer, "d %d ", &id) == 1)
        {
            for (vector<Line>::iterator it = lines.begin(); it != lines.end(); )
            {

                if(it->id == id) 
                    it = lines.erase(it);
                else 
                    ++it;
            }

            for (vector<Line>::iterator it = documentLines.begin(); it != documentLines.end(); )
            {

                if(it->id == id) 
                    it = documentLines.erase(it);
                else 
                    ++it;
            }

            //for (auto line : lines)
            //{
            //    if (line.id == id)
            //    {
            //        line.alive = false;
            //    }
            //}
            //remove_if(lines.begin(), lines.end(), removeLine);
        }
        return 1;
    }
    return 0;
}

void *handle_uds(void *args)
{
    const int buffer_length = 400;
    char buffer[buffer_length];
    string residual = "";

    while(1)
    {
        int id, x, y, state;

        int size;

        size = recv(client_socket, buffer, buffer_length-1, MSG_WAITALL);
        //size = read(client_socket, buffer, buffer_length-1);

        if(size > 0)
        {
            vector<string> substrings = split(residual + buffer, "|");
            int i = 0;

            for (auto message : substrings)
            {
                i++;
                int result = parseMessage((char *) message.c_str());

                if (result == 0 && i == substrings.size())
                {
                    residual = message;
                }
            }
        }
        //send(client_socket, "ok", 2, 0);
        usleep(500);
    }
}

int init_uds()
{
    socklen_t addrlen;
    ssize_t size;
    struct sockaddr_un address;
    const int y = 1;
    if((server_socket=socket (AF_LOCAL, SOCK_STREAM, 0)) > 0)
        printf ("created socket\n");
    unlink(fifo_path);
    address.sun_family = AF_LOCAL;
    strcpy(address.sun_path, fifo_path);
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

void renderLine(SDL_Renderer *rend, vector<SDL_Point*> *line, SDL_Color color)
{
    if(line->size() > 1)
    {
        SDL_Point point_array[line->size()];
        for(int i = 0; i < line->size(); i++)
        {
            point_array[i] = *(line->at(i));
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
        vector<SDL_Point*> coords = line.coords;
        //renderLine(renderer, &coords, line.color);
        renderLine(renderer, &coords, {255, 255, 0});
    }

    for (auto const& line : documentLines)
    {
        vector<SDL_Point*> coords = line.coords;
        renderLine(renderer, &coords, {255, 255, 255});
    }

    if(currentLine.coords.size() > 1)
    {
        SDL_Point point_array[currentLine.coords.size()];
        for(int i = 0; i < currentLine.coords.size(); i++)
        {
            point_array[i] = *(currentLine.coords.at(i));
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

    renderHighlights(renderer);
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
