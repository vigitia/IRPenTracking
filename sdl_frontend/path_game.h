#ifndef PATH_GAME_H
#define PATH_GAME_H

enum PathGameState {
    wait,
    play,
    finish
};

class PathGame
{
    public:
        long long start_time;
        long long current_time;
        float end_time;

        PathGameState state;

        PathGame();

        void reset();
        void start();
        void stop();
        void update(int x, int y, int state);
        float getTimer();
};

inline PathGame pathGame;

#endif
