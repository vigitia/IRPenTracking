#include <SDL2/SDL.h>
#include <vector>

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

enum Modes {
    draw,
    phrase,
    cross,
    save,
    latency,
    particle,
    image
};

struct Point {
	float x;
	float y;
};

struct Line {
    int id;
    SDL_Color color;
    vector<Point> coords;
    bool alive;
};

struct Poly {
    int id;
    short int x[4];
    short int y[4];
    bool alive;
};
