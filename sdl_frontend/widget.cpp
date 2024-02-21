#include "main.h"

Widget::Widget(){
    this->setVisibility(true);
}

bool Widget::isPointOnWidget(Point position){
    float x = position.x;
    float y = position.y;
    float my_x = this->position.x;
    float my_y = this->position.y;
    float w = this->width;
    float h = this->height;
    return (x >= my_x && x <= my_x + w && y >= my_y && y <= my_y + h);
}

// turns coordinates from the absolute coordinate system (origin in the upper left corner of the desk)
// to coordinates in the widgets own coordinate system (origin in the widget's own upper left corner)
Point Widget::getRelativeCoordinates(Point position){
    Point relPos;
    relPos.x = position.x - this->position.x;
    relPos.y = position.y - this->position.y;
    return relPos;
}

void onClick(Point position){

}

void onHover(Point position){
    
}
