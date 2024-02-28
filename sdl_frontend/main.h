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
#include <memory>

#define MODE_FIFO 0
#define MODE_UDS 1

#define COMMUNICATION_MODE MODE_UDS

#define MODE_1080 0
#define MODE_4K 1

#define MODE MODE_1080

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


#define PALETTE_TEXTURE_PATH "assets/big_palette_expanded.png"
#define PALETTE_SELECTION_INDICATOR_PATH "assets/palette_indicator.png"


#if MODE == MODE_1080
#define ERASE_RADIUS_BIG 38
#define ERASE_RADIUS_SMALL 5
#define PALETTE_HEIGHT 75

#else
#define ERASE_RADIUS_BIG 75
#define ERASE_RADIUS_SMALL 10
#define PALETTE_HEIGHT 150
#endif

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
        char* defaultImagePath;

        SDL_Surface* imageSurface;
        SDL_Texture* imageTexture;

    public:
        ImagePanel();
        virtual void setDefaultImagePath(char * imagePath);
        virtual void render(SDL_Renderer* renderer);
        void setPosition(Point Position);
        void setID(int id);
        int getID();
        virtual void loadTexture(char* texturePath);
        virtual void loadTexture();
        void setTexture(SDL_Texture* texture);
        void setDimensions(int width, int height);
        Point getPosition();
        int getWidth();
        int getHeight();
        void setVisibility(bool visible);
        bool getVisibility();
        
        
};

class Widget : public ImagePanel
{
    public:
        Widget();
        bool isPointOnWidget(Point position);
        Point getRelativeCoordinates(Point position);
        virtual void onClick(Point position); //please override these functions to implement the desired behavior on certain mouse events!
        virtual void onHover(Point position){};

};

class Palette : public Widget
{
    protected:
        ImagePanel selectionIndicator;
        vector<vector<vector<int>>> fields;//think of *fields* as a 2d-array of RGB-Values instead of a 3d-array of ints. rows before columns.
        int field_len_y;
        int field_len_x;
        int fieldSize;
    public: 
        Palette(vector<vector<vector<int>>> fields, int field_len_y, int field_len_x, int field_size); 
        void select(int field_x, int field_y);
        void loadTexture(char* texture_path, char * indicator_texture_path);
        void loadTexture();
        void setDefaultImagePath(char* imagePath, char* selectorImagePath);
        void onClick(Point position);
        void render(SDL_Renderer* renderer);
};

enum Tool{ pencil, eraser, clear};

inline SDL_Renderer* renderer;

inline Modes currentMode = draw;
inline Tool currentTool = pencil;
inline SDL_Color currentColor = {255, 255, 255};

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
inline float eraserRadius;
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

void initPalette();

void processPenDownEvent(int id, int x, int y, int state);
void processPenUpEvent(int id);

void drawLine(int id, int x, int y, int state);
void erase(int id, float x, float y, int state, int radius);

void finishLine(int id);
void finishErase(int id);

void clearCanvas();

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
inline vector<std::shared_ptr<Widget>> uiElements;

#endif
