#ifndef __DOCUMENT_H__
#define __DOCUMENT_H__

#include "main.h"

class Document {
    public:
        Document(Point top_left, Point top_right, Point bottom_left, Point bottom_right);
        Document();
        Point top_left, top_right, bottom_left, bottom_right;
        bool isPointInDocument(int x, int y);
        bool alive = false;
        void setPoints(Point top_left, Point top_right, Point bottom_left, Point bottom_right);
};

inline Document document = Document();

#endif
