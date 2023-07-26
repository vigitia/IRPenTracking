#ifndef PATH_GAME_H
#define PATH_GAME_H

#include <vector>

struct PenEvent {
    long long timestamp;
    int x;
    int y;
    int state;
    int onLine;
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
        int resolution_factor;

        SDL_Rect profileTextRect;
        SDL_Surface *profileTextSurface;
        SDL_Texture *profileTextTexture;

        SDL_Rect timer_rect;
        SDL_Surface *timerSurface;
        SDL_Texture *timerTexture;

        SDL_Rect pid_rect;
        SDL_Surface *pidSurface;
        SDL_Texture *pidTexture;

        SDL_Rect acc_rect;
        SDL_Surface *accSurface;
        SDL_Texture *accTexture;

        SDL_Rect pathRect;
        SDL_Surface *pathSurface;
        SDL_Texture *pathTexture;

        int num_points_correct = 0;
        int num_points_wrong = 0;

        bool checkPixel(int x, int y);

    public:
        long long start_time;
        long long current_time;
        float end_time;
        int participant_id = 0;

        // a bit hacky
        // 4K mode is 1 --> FullHD pixel values are multiplied with 2 if in 4K mode
        int start_x = 116 * (MODE + 1); // 233.482
        int start_y = 627 * (MODE + 1); // 1254.285
        int finish_x = 1780 * (MODE + 1); // 3559.482
        int finish_y = 627 * (MODE + 1); // 1254.285
        int start_region_radius = 8 * (MODE + 1);
        int finish_region_radius = 8 * (MODE + 1);

        PathGameState state = waiting;

        bool isSavingProfilePicture = false;
        SDL_Rect profileRect;

        PathGame();

        void reset();
        void start();
        void finish();
        void update(int x, int y, int state);
        float getTimer();
        float getAccuracy();
        void renderTimer(SDL_Renderer* renderer);
        void renderParticipantID(SDL_Renderer* renderer);
        void render(SDL_Renderer* renderer);
        void renderProfilePicture(SDL_Renderer* renderer);
        void renderAccuracy(SDL_Renderer* renderer);
        void saveHighscore();
        void renderHighscore(SDL_Renderer* renderer);
        //void saveProfilePicture(SDL_Renderer* renderer);
};

inline PathGame pathGame;

#endif
