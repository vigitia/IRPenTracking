#include "main.h"
#include "path_game.h"
#include <sstream>

PathGame::PathGame()
{
    reset();
}

void PathGame::reset()
{
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

    logData("testfile.csv", data);
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
