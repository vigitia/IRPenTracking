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
        case CODE_IMAGE:
            return parseMessageImage(buffer);
            break;
        case CODE_UI_ELEMENT:
            return parseMessageUIElement(buffer);
            break;
        case CODE_TOGGLE_HIDE_UI:
            toggleHideUI();
            return 1;
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
    int state;
    //printf("Got %s (in C++)", buffer);
    if(sscanf(buffer, "d %d %f %f %f %d", &id, &x, &y, &radius, &state) == 5)
    {
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
        //fflush(stdout);
        //NEW BEHAVIOR: Lines are selected with a circle
        //Possible Optimization: Compute Bounding boxes of lines and check collision with eraser before looping over every point

        if (state == STATE_DRAW)
        {
            showEraserIndicator = true;
            //re-implementation of dubious code that is hopefully more readable (and actually works)
            vector<Line> newLines;
            for (vector<Line>::iterator lineit = lines.begin(); lineit != lines.end();)
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
                                pointit = lineit->coords.erase(lineit->coords.begin(), pointit);
                                if (newLineSegment.coords.size() >= MIN_NON_ERASE_LINE_POINTS)
                                {
                                    newLines.push_back(newLineSegment);
                                }
                            }

                        }
                    }
                    else
                    {
                        if(inCollidingSpan)//are we changing from an erased section to an unerased section?
                        {
                            inCollidingSpan = false;
                            pointit = lineit->coords.erase(lineit->coords.begin(), pointit);
                        }
                    }
                }

                if (lineit->coords.size() < MIN_NON_ERASE_LINE_POINTS)
                {
                    lineit = lines.erase(lineit);
                }
                else
                {
                    ++lineit;
                }
            }

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
        
        return 1;
    }
    return 0;
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


int parseMessageImage(char* buffer)
{
    int id;
    int visibility;
    float x = -1.0;
    float y = -1.0;
    int width = -1;
    int height = -1;
    char filepath [200];
    int num_args = sscanf(buffer, "i %d %d %f %f %d %d %s", &id, &visibility, &x, &y, &width, &height, &filepath);
    if (num_args >= 4) 
    {

        bool is_known = false;
        ImagePanel img;
        for (vector<ImagePanel>::iterator imgit = images.begin(); imgit != images.end(); ++imgit)
        {
            if (imgit->getID() == id)
            {
                is_known = true;
                img = *imgit;
                break;
            }
        }


        if (!is_known && num_args < 7)
        {
            cout << "WARNING: You are trying to create a new image, but you only passed " << num_args << "arguments." << endl;
            return 0;
        }

        img.setVisibility((visibility > 0));
        if (x != -1.0 && y != -1.0)
        {
            Point newPosition;
            newPosition.x = x;
            newPosition.y = y;
            img.setPosition(newPosition);
        }
        if (width != -1 && height != -1)
        {
            img.setDimensions(width, height);
        }
        
        if (num_args == 7)
        {
            img.setTexture(filepath);
        }
        
        if (!is_known)
        {
            images.push_back(img);
        }

        return 1;

    }
    else return 0;
}


int parseMessageUIElement(char* buffer)
{
    int id;
    int visibility;
    float x = -1.0;
    float y = -1.0;
    int width = -1;
    int height = -1;
    char filepath [200];
    int num_args = sscanf(buffer, "u %d %d %f %f %d %d %s", &id, &visibility, &x, &y, &width, &height, &filepath);

    if (num_args >= 4) 
    {

        bool is_known = false;

        ImagePanel* img;
        vector<ImagePanel>::const_iterator imgIndex;
        for (vector<ImagePanel>::iterator imgit = uiElements.begin(); imgit != uiElements.end(); ++imgit)
        {
            if (imgit->getID() == id)
            {
                is_known = true;
                img = &(*imgit);
                imgIndex = imgit;
                break;
            }
        }


        if (!is_known)
        {
            ImagePanel newimg;
            img = &newimg;
        }




        if (!is_known && num_args < 7)
        {
            cout << "WARNING: You are trying to create a new UI Element, but you only passed " << num_args << "arguments." << endl;
            return 0;
        }

        img->setVisibility((visibility > 0));
        if (x != -1.0 && y != -1.0)
        {
            Point newPosition;
            newPosition.x = x;
            newPosition.y = y;
            img->setPosition(newPosition);
        }
        if (width != -1 && height != -1)
        {
            img->setDimensions(width, height);
        }

        if (num_args == 7)
        {
            img->setTexture(filepath);
        }
        
        if (!is_known)
        {
            img->setID(id);
            uiElements.push_back(*img);
        }
        else
        {
            const ImagePanel immutableImage = *img;
            uiElements.erase(imgIndex);
            uiElements.insert(imgIndex, immutableImage);
        }

        return 1;

    }
    else return 0;
}

