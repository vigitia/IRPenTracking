#ifndef __MAIN_H__
#define __MAIN_H__

#include <SDL2/SDL.h>
#include <vector>
#include <map>
#include <string>
#include <SDL2/SDL.h>
#include <stdio.h>
#include <iostream>
#include <mutex>

#define MODE_FIFO 0
#define MODE_UDS 1

#define COMMUNICATION_MODE MODE_UDS

#define MODE_1080 0
#define MODE_4K 1

#define MODE MODE_4K

#if MODE == MODE_1080
#define WINDOW_WIDTH 1920
#define WINDOW_HEIGHT 1080
#else
#define WINDOW_WIDTH 3840
#define WINDOW_HEIGHT 2160
#endif

#define IMAGE_PATH "image.png"

#define HOVER_INDICATOR_COLOR 0xFF00FFFF
#define SHOW_HOVER_INDICATOR 1

#define SHOW_LINES 1

#define STATE_HOVER 0
#define STATE_DRAW 1


#define CODE_LINE 'l'
#define CODE_LINE_FINISH 'f'
#define CODE_DOCUMENT 'k'
#define CODE_STATE 's'
#define CODE_MATRIX 'm'
#define CODE_RECT 'r'
#define CODE_CLEAR 'c'
#define CODE_CLEAR_SCREEN 'x'
#define CODE_DELETE 'd'

inline char* SCREENSHOT_PATH = "screenshots/";
inline const char* PHRASES_PATH = "evaluation/phrases.txt";

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

struct Pen {
    Point position;
    Line currentLine;
    bool wasInDocument;
    bool state;
    bool alive;
};

struct Poly {
    int id;
    short int x[4];
    short int y[4];
    bool alive;
};

inline SDL_Renderer* renderer;

inline Modes currentMode = draw;

inline vector<Line> lines;
inline vector<Line> documentLines;
inline Line currentLine;
inline bool wasInDocument = false;
inline map<int, struct Poly> rects;
inline map<int, struct Pen> pens;
inline int currentId = 0;
inline int participantId = 0;
inline int currentX, currentY = 0;
inline int currentState = 0;

inline mutex mutex_pens;
inline mutex mutex_lines;

inline uint32_t highlightColor = 0x9900FFFF;

void clearScreen();
SDL_Surface* loadSurface(string path);
void saveImage();

bool is_on_right_side(int x, int y, Point xy0, Point xy1);
Point multiplyPointMatrix(Point point, float matrix[3][3]);
const string currentDateTime();
long long millis();
long long micros();
SDL_Point pointToSDL(Point p);

int parseMessage(char* buffer);
int parseMessageLine(char* buffer);
int parseMessageFinishLine(char* buffer);
int parseMessageDocument(char* buffer);
int parseMessageState(char* buffer);
int parseMessageMatrix(char* buffer);
int parseMessageRect(char* buffer);
int parseMessageDelete(char* buffer);

vector<string> split (string s, string delimiter);

#endif
