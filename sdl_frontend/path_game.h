#ifndef PATH_GAME_H
#define PATH_GAME_H

#include <vector>

struct PenEvent {
    long long timestamp;
    int x;
    int y;
    int state;
};

enum PathGameState {
    waiting,
    playing,
    finished
};

class PathGame
{
    private:
        bool isPenInStartRegion(int x, int y);
        bool isPenInFinishRegion(int x, int y);
        vector<PenEvent> events;

    public:
        long long start_time;
        long long current_time;
        float end_time;

        // a bit hacky
        // 4K mode is 1 --> FullHD pixel values are multiplied with 2 if in 4K mode
        int start_x = 116 * (MODE + 1); // 233.482
        int start_y = 627 * (MODE + 1); // 1254.285
        int finish_x = 1780 * (MODE + 1); // 3559.482
        int finish_y = 627 * (MODE + 1); // 1254.285
        int start_region_radius = 8 * (MODE + 1);
        int finish_region_radius = 2 * (MODE + 1);

        PathGameState state;

        PathGame();

        void reset();
        void start();
        void finish();
        void update(int x, int y, int state);
        float getTimer();
};

inline PathGame pathGame;

#endif
