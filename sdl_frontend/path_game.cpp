#include "main.h"
#include "path_game.h"

PathGame::PathGame()
{
    reset();
}

void PathGame::reset()
{
    state = wait;
}

void PathGame::start()
{
    state = play;
    start_time = millis();
}

void PathGame::stop()
{
    state = finish;
    end_time = getTimer();
}

void PathGame::update(int x, int y, int state)
{
    cout << x << " " << y << " " << state << endl;
}

float PathGame::getTimer()
{
    return ((float)(millis() - start_time)) / 1000;
}
