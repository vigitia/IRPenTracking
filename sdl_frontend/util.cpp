#include "main.h"
#include <ctime>
#include <SDL2/SDL.h>

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

SDL_Point pointToSDL(Point p)
{
	SDL_Point result = {(int) p.x, (int) p.y};
	return result;
}
