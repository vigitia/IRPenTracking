#include "main.h"
#include <ctime>
#include <cmath>
#include <sys/time.h>
#include <SDL2/SDL.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "path_game.h"

// https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library
bool is_on_right_side(int x, int y, Point xy0, Point xy1)
{
    int x0 = xy0.x;
    int y0 = xy0.y;
    int x1 = xy1.x;
    int y1 = xy1.y;
    float a = float(y1 - y0);
    float b = float(x0 - x1);
    float c = - a * x0 - b * y0;
    return a * x + b * y + c >= 0;
}

Point multiplyPointMatrix(Point point, float matrix[3][3])
{
    float x = point.x;
    float y = point.y;

    float result_x = matrix[0][0] * x + matrix[1][0] * y + matrix[2][0];
    float result_y = matrix[0][1] * x + matrix[1][1] * y + matrix[2][1];
    float result_z = matrix[0][2] * x + matrix[1][2] * y + matrix[2][2];

    result_x /= result_z;
    result_y /= result_z;

    return {result_x, result_y};
}

// https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

// https://stackoverflow.com/questions/997946/how-to-get-current-time-and-date-in-c
const string currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);

    return buf;
}

long long millis()
{
    struct timeval te; 
    gettimeofday(&te, NULL);
    long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;
    return milliseconds;
}

long long micros()
{
    struct timeval te; 
    gettimeofday(&te, NULL);
    long long microseconds = te.tv_sec + te.tv_usec;
    return microseconds;
}

SDL_Point pointToSDL(Point p)
{
	SDL_Point result = {(int) p.x, (int) p.y};
	return result;
}

float getDistance(float x1, float y1, float x2, float y2)
{
    float a = abs(x1 - x2);
    float b = abs(y1 - y2);
    return sqrt(a * a + b * b);
}

// still used, but marked for deletion
// cx, cy = coords of circle center; r = radius
vector<Point> collideLineWithCircle(vector<Point> line_points, float cx, float cy, float r)
{
    vector <Point> colliding_points;
    for (vector<Point>::iterator it = line_points.begin(); it != line_points.end(); )
    {
        if (getDistance(it->x, it->y, cx, cy) <= r)
        {
            //printf("Line at Point %f, %f collides with eraser at %f, %f with radius %f \n", it->x, it->y, cx, cy, r);
            //fflush(stdout);
            colliding_points.push_back(*it);
            
        }
        ++it;
    }  
    return colliding_points;
}

void logData(const string& fileName, const string& data) 
{
    ofstream outputFile(fileName);

    if (outputFile.is_open()) 
    {
        outputFile << data;

        outputFile.close();
    } 
    else 
    {
        cerr << "logData - Error opening file: " << fileName << endl;
    }
}

bool compareHighscoreEntries(const HighscoreEntry& a, const HighscoreEntry& b)
{
    return a.time < b.time;
}

void toggleHideUI(){
    SHOW_UI = !SHOW_UI;
}