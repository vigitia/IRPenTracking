#include "main.h"
#include "path_game.h"
#include <sstream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_image.h>
#include <unistd.h>

PathGame::PathGame()
{
    if (MODE == MODE_1080) resolution_factor = 1;
    else resolution_factor = 2;

    profileTextRect = { 100 * resolution_factor, 770 * resolution_factor, 100 * resolution_factor, 25 * resolution_factor };
    profileRect = { 100 * resolution_factor, 800 * resolution_factor, 150 * resolution_factor, 150 * resolution_factor };

    timer_rect = { 50, 100, 100, 25 };
    timerTexture = SDL_CreateTextureFromSurface( renderer, timerSurface );

    pid_rect = { 50, 50, 100, 25 };
    pidTexture = SDL_CreateTextureFromSurface( renderer, pidSurface );

    pathRect = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };
}

void PathGame::reset()
{
    // TODO: this should be in the constructor, but it does not work there
    // I think this is because something is not initialized when constructor is called.
    pathSurface = loadSurface(PATH_PATH);
    pathTexture = SDL_CreateTextureFromSurface( renderer, pathSurface );

    profileTextSurface = TTF_RenderText_Solid( font, "Profilbild:", textColor );
    profileTextTexture = SDL_CreateTextureFromSurface( renderer, profileTextSurface );

    participant_id++;
    events.clear();
    state = waiting;
}

void PathGame::start()
{
    state = playing;
    start_time = millis();
}

void PathGame::finish()
{
    end_time = getTimer();
    state = finished;

    stringstream ss;
    ss << "timestamp,x,y,state" << endl;

    for(const auto event : events)
    {
        //cout << event.timestamp << ", " << event.x << ", " << event.y << ", " << event.state << endl;
        ss << event.timestamp << "," << event.x << "," << event.y << "," << event.state << endl;
    }

    string data = ss.str();

    char filename[400];
    sprintf(filename, "%s/path_%lld_%d.csv", PATH_GAME_LOG_PATH, millis(), participant_id);

    logData(filename, data);

    isSavingProfilePicture = true;
}

bool PathGame::isPenInStartRegion(int x, int y)
{
    return getDistance(x, y, start_x, start_y) <= start_region_radius;
}

bool PathGame::isPenInFinishRegion(int x, int y)
{
    return getDistance(x, y, finish_x, finish_y) <= finish_region_radius;
}

void PathGame::update(int x, int y, int pen_state)
{
    if (state == playing)
    {
        events.push_back({millis(), x, y, pen_state});
    }

    if (state == waiting && isPenInStartRegion(x, y) && pen_state == STATE_DRAW)
    {
        start();
    }

    if (state == playing && isPenInFinishRegion(x, y))
    {
        finish();
    }
}

float PathGame::getTimer()
{
    if (state == waiting) return 0;
    else if (state == playing) return ((float)(millis() - start_time)) / 1000;
    else if (state == finished) return end_time;
}

void PathGame::renderTimer(SDL_Renderer* renderer)
{
    SDL_RenderCopy( renderer, timerTexture, NULL, &timer_rect );

    float timeRemaining = getTimer();

    char timer_string[20];
    sprintf(timer_string, "%.3f", timeRemaining);

    timerSurface = TTF_RenderText_Solid( font, timer_string, textColor );
    timerTexture = SDL_CreateTextureFromSurface( renderer, timerSurface );
}

void PathGame::renderParticipantID(SDL_Renderer* renderer)
{
    SDL_RenderCopy( renderer, pidTexture, NULL, &pid_rect );

    char pid_string[20];
    sprintf(pid_string, "PID: %03d", participant_id);

    pidSurface = TTF_RenderText_Solid( font, pid_string, textColor );
    pidTexture = SDL_CreateTextureFromSurface( renderer, pidSurface );
}

void PathGame::renderProfilePicture(SDL_Renderer* renderer)
{
    SDL_RenderCopy( renderer, profileTextTexture, NULL, &profileTextRect );
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderDrawRect(renderer, &profileRect);
}

void PathGame::render(SDL_Renderer* renderer)
{
    SDL_RenderCopy(renderer, pathTexture, NULL, &pathRect);

    renderParticipantID(renderer);
    renderTimer(renderer);
    renderProfilePicture(renderer);

    filledCircleColor(renderer, start_x, start_y, start_region_radius, 0xFF0000FF);
    filledCircleColor(renderer, finish_x, finish_y, finish_region_radius, 0xFF0000FF);
}

/*
void PathGame::saveProfilePicture(SDL_Renderer *renderer)
{
    const Uint32 format = SDL_PIXELFORMAT_ARGB8888;
    cout << profileRect.w << " " << profileRect.h << endl;
    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, profileRect.w, profileRect.h, 32, format);

    //renderProfilePicture(renderer);
    usleep(20000);

    char filename[400];
    sprintf(filename, "%s/picture_%lld_%d.png", PATH_GAME_LOG_PATH, millis(), participant_id);

    SDL_RenderReadPixels(renderer, &profileRect, format, surface->pixels, surface->pitch);
    IMG_SavePNG(surface, filename);

    isSavingProfilePicture = false;
}
*/
