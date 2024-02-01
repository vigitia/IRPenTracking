#ifndef __RENDER_H__
#define __RENDER_H__

#include <SDL2/SDL.h>
#include "particle.h"

inline SDL_Surface* imageSurface;
inline SDL_Texture* imageTexture;
inline bool SHOW_PARTICLES = false;
inline bool SHOW_PALETTE = true;
inline vector<Particle> particles;
inline bool showBrokenPipeIndicator = false;
inline bool isSaving = false;

void render(SDL_Renderer* renderer);
void renderLine(SDL_Renderer *rend, vector<Point> *line, SDL_Color color);
void renderParticles(SDL_Renderer* renderer);
void renderCrosses(SDL_Renderer* renderer);
void renderFullscreenImage(SDL_Renderer* renderer);
void renderPhrase(SDL_Renderer* renderer);
void renderHighlights(SDL_Renderer* renderer);
void renderLines(SDL_Renderer* renderer);
void renderHoverIndicator(SDL_Renderer* renderer);
void renderEraserIndicator(SDL_Renderer* renderer);
void renderLatencyTest(SDL_Renderer* renderer);
void renderBrokenPipeIndicator(SDL_Renderer* renderer);
void renderParticipantID(SDL_Renderer* renderer);
void renderAllImages(SDL_Renderer* renderer);
void renderAllUIElements(SDL_Renderer* renderer);

#endif
