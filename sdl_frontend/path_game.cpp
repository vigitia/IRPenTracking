#include "main.h"
#include "path_game.h"

PathGame::PathGame()
{
    reset();
}

void PathGame::reset()
{
    start_time = millis();
}

float PathGame::getTimer()
{
    return ((float)(millis() - start_time)) / 1000;
}
