#ifndef __MAIN_H__
#define __MAIN_H__

#include <SDL2/SDL.h>
#include <vector>
#include <map>
#include <string>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
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
#define PATH_PATH "assets/path_1080.png"
#else
#define WINDOW_WIDTH 3840
#define WINDOW_HEIGHT 2160
#define PATH_PATH "assets/path_4k.png"
#endif

#define IMAGE_PATH "image.png"
#define HIGHSCORE_PATH "highscore"

#define PATH_GAME_LOG_PATH "path_game_log"

#define HOVER_INDICATOR_COLOR 0xFF00FFFF
#define SHOW_HOVER_INDICATOR 1

#define ERASE_INDICATOR_COLOR 0xAA0000FF
#define SHOW_ERASE_INDICATOR 1
#define MIN_NON_ERASE_LINE_POINTS 2

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
#define CODE_ERASE_FINISH 'v'
#define CODE_IMAGE 'i'
#define CODE_UI_ELEMENT 'u'
#define CODE_TOGGLE_HIDE_UI 'h'


#define PALETTE_TEXTURE_PATH "assets/big_palette.png"
#define PALETTE_WIDTH 1800
#define PALETTE_HEIGHT 180

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
    image,
    path
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

struct HighscoreEntry {
    int pid;
    float time;
    float accuracy;
};

class ImagePanel
{
    protected:
        int id;
        Point position;
        int width;
        int height;
        bool visible;

        SDL_Surface* paletteSurface;
        SDL_Texture* paletteTexture;

    public:
        ImagePanel();
        void render(SDL_Renderer* renderer);
        void setPosition(Point Position);
        void setID(int id);
        int getID();
        void loadTexture(char* texturePath);
        void setTexture(SDL_Texture* texture);
        void setDimensions(int width, int height);
        Point getPosition();
        int getWidth();
        int getHeight();
        void setVisibility(bool visible);
        bool getVisibility();
        
        
};

inline SDL_Renderer* renderer;

inline Modes currentMode = draw;

inline TTF_Font* font;
inline SDL_Color textColor = { 255, 255, 255 };

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

inline bool showEraserIndicator;
inline float eraserIndicatorRadius;
inline vector <Point> eraserTips; //quick hack. Probably better to use pens for detecting eraser position and rendering eraser marker. also unused (unless I've overlooked something).

inline bool SHOW_UI = true;

inline mutex mutex_pens;
inline mutex mutex_lines;

inline uint32_t highlightColor = 0x9900FFFF;

//(possible) short-term workaround: preloaded UI textures
inline SDL_Texture* preloadedPaletteTexture;
inline SDL_Texture* preloadedPaletteIndicatorTexture;

void clearScreen();
SDL_Surface* loadSurface(string path);
void saveImage();
void saveProfilePicture();

bool is_on_right_side(int x, int y, Point xy0, Point xy1);
Point multiplyPointMatrix(Point point, float matrix[3][3]);
const string currentDateTime();
long long millis();
long long micros();
SDL_Point pointToSDL(Point p);
float getDistance(float x1, float y1, float x2, float y2);
vector <Point> collideLineWithCircle(vector<Point> line_points, float cx, float cy, float r);
void logData(const string& fileName, const string& data);
bool compareHighscoreEntries(const HighscoreEntry& a, const HighscoreEntry& b);
void toggleHideUI();

void preloadTextures(SDL_Renderer* renderer);

int parseMessage(char* buffer);
int parseMessageLine(char* buffer);
int parseMessageFinishLine(char* buffer);
int parseMessageDocument(char* buffer);
int parseMessageState(char* buffer);
int parseMessageMatrix(char* buffer);
int parseMessageRect(char* buffer);
int parseMessageDelete(char* buffer);
int parseMessageFinishErase(char* buffer);
int parseMessageImage(char* buffer);
int parseMessageUIElement(char* buffer);

vector<string> split (string s, string delimiter);

inline vector<ImagePanel> images;
inline vector<ImagePanel> uiElements;

#endif
