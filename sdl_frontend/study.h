#ifndef __STUDY_H__
#define __STUDY_H__

#include "main.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>

#if MODE == MODE_1080
#define CROSSES_PATH "evaluation/crosses_dots_1080.png"
#else
#define CROSSES_PATH "evaluation/crosses_dots_4k.png"
#endif

#define FONT_SIZE 42

const int TEXT_BOX_WIDTH = (int)(WINDOW_WIDTH * 0.7);
const int TEXT_BOX_HEIGHT_SMALL = (int)(WINDOW_HEIGHT * 0.05);
const int TEXT_BOX_HEIGHT_MEDIUM = (int)(WINDOW_HEIGHT * 0.075);
const int TEXT_BOX_HEIGHT_LARGE = (int)(WINDOW_HEIGHT * 0.1);
const int TEXTBOX_OFFSET = (int)(WINDOW_HEIGHT * 0.1);

inline vector<string> phrases;
inline vector<int> usedPhrases;

inline string currentPhrase;
inline int currentTextSize = 0;

inline SDL_Texture* textTexture;

inline SDL_Surface* textSurface;
inline SDL_Surface* crossesSurface;
inline SDL_Texture* crossesTexture;

void loadPhrases();
void nextPhrase();

#endif
