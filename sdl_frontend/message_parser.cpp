#include "main.h"
#include "path_game.h"
#include "document.h"
#include <algorithm>

int parseMessage(char* buffer)
{
    //printf(buffer);
    //printf("\n");
    //fflush(stdout);
    //
    switch (buffer[0])
    {
        case CODE_LINE:
            return parseMessageLine(buffer);
            break;
        case CODE_LINE_FINISH:
            return parseMessageFinishLine(buffer);
            break;
        case CODE_DOCUMENT:
            return parseMessageDocument(buffer);
            break;
        case CODE_STATE:
            return parseMessageState(buffer);
            break;
        case CODE_MATRIX:
            return parseMessageMatrix(buffer);
            break;
        case CODE_RECT:
            return parseMessageRect(buffer);
            break;
        case CODE_CLEAR:
            rects.clear();
            return 1;
            break;
        case CODE_CLEAR_SCREEN:
            clearScreen();
            return 1;
            break;
        case CODE_DELETE:
            return parseMessageDelete(buffer);
            break;
        case CODE_ERASE_FINISH:
            return parseMessageFinishErase(buffer);
            break;
    }

    return 0;
}

long long last_micros = 0;

int parseMessageLine(char* buffer)
{
    int id, x, y, state;
    unsigned int r;
    unsigned int g;
    unsigned int b;
    // parse new values from IPC
    // only continue if all values could be read correctly
    if(sscanf(buffer, "l %d %u %u %u %d %d %d ", &id, &r, &g, &b, &x, &y, &state) == 7)
    {
        //Latency (?)

        //long long cur_micros = micros();
        //cout << cur_micros - last_micros << endl;
        //last_micros = cur_micros;

        mutex_pens.lock();
        if(pens.find(id) == pens.end())
        {
            struct Pen currentPen;
            currentPen.currentLine.id = id;
            currentPen.alive = 1;
            pens[id] = currentPen;
        }

        pens[id].position.x = x;
        pens[id].position.y = y;
        pens[id].state = (bool) state;

        //cout << "id: " << id << " x: " << x << "; y: " << y << " state: " << state << endl;
        if (state == STATE_DRAW && id == pens[id].currentLine.id)
        {
            pens[id].currentLine.color = {r, g, b};
        }
        //pens[id].currentLine.id = currentId;

        bool inDocument = document.isPointInDocument(x, y);

        mutex_lines.lock();
        if (document.alive && inDocument && !pens[id].wasInDocument)
        {
            lines.push_back(pens[id].currentLine);
            pens[id].currentLine.coords.clear();
            if(state == STATE_DRAW) pens[id].currentLine.coords.push_back({x, y});
            pens[id].wasInDocument = true;
        }
        else if (document.alive && !inDocument && pens[id].wasInDocument)
        {
            documentLines.push_back(pens[id].currentLine);
            pens[id].currentLine.coords.clear();
            if(state == STATE_DRAW) pens[id].currentLine.coords.push_back({x, y});
            pens[id].wasInDocument = false;
        }
        else
        {
            if(state == STATE_DRAW)
            {
                pens[id].currentLine.coords.push_back({x, y});
            }
        }

        pathGame.update(x, y, state);

        mutex_lines.unlock();
        mutex_pens.unlock();
        return 1; 
    }

    return 0;
}

int parseMessageFinishLine(char* buffer)
{
    int id;
    int pos;

    if(sscanf(buffer, "f %d ", &id) == 1)
    {
        mutex_pens.lock();
        mutex_lines.lock();
        bool inDocument = document.isPointInDocument(pens[id].position.x, pens[id].position.y);

        if (inDocument)
        {
            documentLines.push_back(pens[id].currentLine);
        }
        else
        {
            lines.push_back(pens[id].currentLine);
        }

        pens[id].currentLine.coords.clear();
        pens[id].alive = 0;

        pens.erase(id);
        mutex_lines.unlock();
        mutex_pens.unlock();
        return 1;
    }

    return 0;
}

int parseMessageDocument(char* buffer)
{
    int id, state;
    int x1, x2, x3, x4;
    int y1, y2, y3, y4;
    if(sscanf(buffer, "k %d %d %d %d %d %d %d %d %d \n ", &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &state) == 9)
    {
        //document.alive = (bool) state;
        //cout << buffer << endl;

        document.top_left = {x1, y1};
        document.top_right = {x2, y2};
        document.bottom_right = {x3, y3};
        document.bottom_left = {x4, y4};
        return 1;
    }

    return 0;
}

int parseMessageState(char* buffer)
{
    int state;
    if(sscanf(buffer, "s %d \n ", &state) == 1)
    {
        document.alive = (bool) state;
        return 1;
    }

    return 0;
}

int parseMessageMatrix(char* buffer)
{
    float matrix[3][3];
    //cout << buffer << endl;

    if(sscanf(buffer, "m %f %f %f %f %f %f %f %f %f \n ", &matrix[0][0], &matrix[1][0], &matrix[2][0], &matrix[0][1], &matrix[1][1], &matrix[2][1], &matrix[0][2], &matrix[1][2], &matrix[2][2]) == 9)
    {
        //if (document.alive)
        //{
        for (int j = 0; j < documentLines.size(); j++)
        {
            for (int i = 0; i < documentLines.at(j).coords.size(); i++)
            {
                Point point = documentLines.at(j).coords.at(i);
                Point pnt = multiplyPointMatrix(point, matrix);
                documentLines.at(j).coords.at(i) = pnt;
            }
        }
        //}
        return 1;
    }

    return 0;
}

int parseMessageRect(char* buffer)
{

    int id, state;
    int x1, x2, x3, x4;
    int y1, y2, y3, y4;
    if(sscanf(buffer, "r %d %d %d %d %d %d %d %d %d %d \n ", &id, &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &state) == 10)
    {
        //cout << buffer << endl;
        struct Poly poly;

        poly.id = id;
        poly.alive = state;
        poly.x[0] = x1;
        poly.x[1] = x2;
        poly.x[2] = x3;
        poly.x[3] = x4;
        poly.y[0] = y1;
        poly.y[1] = y2;
        poly.y[2] = y3;
        poly.y[3] = y4;

        // new rect
        if(rects.find(id) == rects.end())
        {
            rects[id] = poly;
        }
        else
        {
            if(state == 0)
            {
                rects.erase(id);
            }
            else
            {
                rects[id].x[0] = x1;
                rects[id].x[1] = x2;
                rects[id].x[2] = x3;
                rects[id].x[3] = x4;
                rects[id].y[0] = y1;
                rects[id].y[1] = y2;
                rects[id].y[2] = y3;
                rects[id].y[3] = y4;
            }
        }

        return 1;
    }

    return 0;
}

int parseMessageDelete(char* buffer)
{
    int id;
    float x;
    float y;
    float radius;
    //printf("Got %s (in C++)", buffer);
    if(sscanf(buffer, "d %d %f %f %f", &id, &x, &y, &radius) == 4)
    {
        showEraserIndicator = true;
        mutex_pens.lock();
        if(pens.find(id) == pens.end())
        {
            struct Pen currentPen;
            currentPen.currentLine.id = id;
            currentPen.alive = 1;
            pens[id] = currentPen;
        }

        pens[id].position.x = x;
        pens[id].position.y = y;
        pens[id].state = 1;

        eraserIndicatorRadius = radius;
        mutex_lines.lock();
        //cout << "Eraser at "<< x << "," << y <<" (in C++)" << endl;
        //fflush(stdout);
        //NEW BEHAVIOR: Lines are selected with a circle
        //Possible Optimization: Compute Bounding boxes of lines and check collision with eraser before looping over every point
        

        //re-implementation of dubious code that is hopefully more readable (and actually works)
        vector<Line> newLines;
        for (vector<Line>::iterator lineit = lines.begin(); lineit != lines.end();++lineit)
        { 
            int idx = 0;
            int lastNotCollidingIndex = 0;

            //when iterating through the vector of points, we will alternate between sections of points that will remain and sections of points that will be erased.
            //this bool keeps track of which type of section we are currently in.
            bool inCollidingSpan = false; //

            vector<Point> points = lineit->coords;
            for(vector<Point>::iterator pointit = lineit->coords.begin(); pointit != lineit->coords.end();++pointit)
            {
                if(getDistance(pointit->x, pointit->y, x, y) <= radius) // is the current point colliding with the eraser?
                {
                    
                    if(!inCollidingSpan) //and are we changing from a non-erased section to an erased section?
                    {
                        inCollidingSpan = true;
                        if (pointit != lineit->coords.begin())
                        {
                            Line newLineSegment; //put all points that came before into a new line
                            newLineSegment.color = lineit->color;
                            newLineSegment.alive = lineit->alive;
                            newLineSegment.id = lineit->id; //Breaks any future additions that rely on the ids of lines being unique
                            copy(lineit->coords.begin(), pointit, back_inserter(newLineSegment.coords));
                            //pointit = remove_if(points.begin(), pointit, [](Point val){return true;}); //inline function definition is a hacky way of telling remove() to remove every element in this range. There may be a better way.
                            pointit = lineit->coords.erase(lineit->coords.begin(), pointit);
                            newLines.push_back(newLineSegment);
                            
                        }
                        
                    }
                }
                else
                {
                    if(inCollidingSpan)//are we changing from an erased section to an unerased section?
                    {
                        inCollidingSpan = false;
                        //pointit = remove_if(points.begin(), pointit, [](Point val){return true;});
                        pointit = lineit->coords.erase(lineit->coords.begin(), pointit);
                    }
                }
            }
        }


        //dubious code.
        //vector<Line> newLines;
        //
        //for (vector<Line>::iterator lineit = lines.begin(); lineit != lines.end(); )
        //{
        //    vector<Point> points = lineit->coords;
        //    vector<Point> collidingPoints = collideLineWithCircle(points, x, y, radius);
        //    cout << "len collidingPoints l 300 " << collidingPoints.size() << endl;
        //    if (collidingPoints.size() > 0)
        //    {
        //        
        //        vector<vector<Point>> remainingLineSegments {{}};
        //        int numLineSegments = 0;
        //        bool currentlyNonColliding = true;
        //        for (vector<Point>::iterator pointit = points.begin(); pointit != points.end(); ++pointit)
        //        {
        //            bool is_colliding = false;
        //            for(vector<Point>::iterator pointjt = points.begin(); pointjt != points.end(); ++pointjt)
        //            {
        //                if(pointjt->x == pointit->x && pointjt->y == pointjt->y)
        //                {
        //                    is_colliding == true;
        //                    break;
        //                }
        //            }
        //            
        //            if (is_colliding)
        //            {
        //                if (currentlyNonColliding)
        //                {
        //                    currentlyNonColliding = false;
        //                }
        //            }
        //            else
        //            {
        //                if (!currentlyNonColliding)
        //                {
        //                    currentlyNonColliding = true;
        //                    ++numLineSegments;
        //                    vector<Point> newLineSegment{};
        //                    remainingLineSegments.push_back(newLineSegment);

        //                }
        //                remainingLineSegments[numLineSegments].push_back(*pointit);
        //            }
        //        }
        //        
        //        cout << "remainingLineSegments.size() l 339 "<< remainingLineSegments.size() << endl;
        //        int idx = 0;
        //        for(vector<vector<Point>>::iterator segit = remainingLineSegments.begin(); segit != remainingLineSegments.end(); ++segit)
        //        {
        //            vector<Point> points = *segit;
        //            cout << "points.size l344 "<< points.size() << endl;
        //            if (points.size() > 1)
        //            {
        //                Line newLine;
        //                newLine.coords = points;
        //                newLine.id = lineit->id * -1000 + idx;
        //                newLine.color = lineit->color;
        //                newLines.push_back(newLine);
        //            }

        //            ++idx;
        //        }
        //        lineit=lines.erase(lineit);
        //    }
        //    else
        //        ++lineit;
        //}
        for(vector<Line>::iterator lineit = newLines.begin(); lineit != newLines.end(); ++lineit)
        {
            lines.push_back(*lineit);
        }


        for (vector<Line>::iterator lineit = documentLines.begin(); lineit != documentLines.end(); )
        {
            vector<Point> colliding_points = collideLineWithCircle(lineit->coords, x, y, radius);
            if (colliding_points.size() > 0)
            {
                lineit=lines.erase(lineit);
            }
            else
                ++lineit;
        }

        
        
        mutex_lines.unlock();
        
        mutex_pens.unlock();

        //OLD BEHAVIOR: Lines are selected by their ID
    //if(sscanf(buffer, "d %d ", &id) == 1)
    //{
        //for (vector<Line>::iterator it = lines.begin(); it != lines.end(); )
        //{

        //    if(it->id == id) 
        //        it = lines.erase(it);
        //    else 
        //        ++it;
        //}

        //for (vector<Line>::iterator it = documentLines.begin(); it != documentLines.end(); )
        //{

        //    if(it->id == id) 
        //        it = documentLines.erase(it);
        //    else 
        //        ++it;
        //}

        //for (auto line : lines)
        //{
        //    if (line.id == id)
        //    {
        //        line.alive = false;
        //    }
        //}
        //remove_if(lines.begin(), lines.end(), removeLine);
    }
    return 1;
}


int parseMessageFinishErase(char* buffer)
{
    int id;
    
    if(sscanf(buffer, "v %d ", &id) == 1)
    {
        showEraserIndicator = false;
        pens[id].alive = 0;
        pens[id].state = 0;

        pens.erase(id);
        mutex_pens.unlock();
    }
}
