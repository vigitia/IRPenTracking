#ifndef PATH_GAME_H
#define PATH_GAME_H

class PathGame
{
    public:
        long long start_time;
        long long current_time;

        PathGame();

        void reset();
        float getTimer();
};

inline PathGame pathGame;

#endif
