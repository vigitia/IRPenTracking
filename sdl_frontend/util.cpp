#include "main.h"

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
