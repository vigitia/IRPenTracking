#include "document.h"

Document::Document(Point top_left, Point top_right, Point bottom_left, Point bottom_right)
{
    this->top_left = top_left;
    this->top_right = top_right;
    this->bottom_left = bottom_left;
    this->bottom_right = bottom_right;
    alive = true;
}

Document::Document()
{
    alive = false;
}

void Document::setPoints(Point top_left, Point top_right, Point bottom_left, Point bottom_right)
{
    this->top_left = top_left;
    this->top_right = top_right;
    this->bottom_left = bottom_left;
    this->bottom_right = bottom_right;
}

// https://stackoverflow.com/questions/63527698/determine-if-points-are-within-a-rotated-rectangle-standard-python-2-7-library
bool Document::isPointInDocument(int x, int y)
{
    if (!alive) return false;

    bool is_above = is_on_right_side(x, y, top_left, top_right);
    bool is_below = !is_on_right_side(x, y, bottom_left, bottom_right);
    bool is_left = !is_on_right_side(x, y, top_left, bottom_left);
    bool is_right = is_on_right_side(x, y, top_right, bottom_right);
     
    return !(is_above || is_below || is_left || is_right) || (is_above && is_below && is_left && is_right);
}
