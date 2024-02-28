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

        processPenDownEvent(id,x,y,state);
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
        processPenUpEvent(id);
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

//currently unused.
//Might be useful again if erasing and drawing are done with different tangibles, so I left it in.
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
        erase(id, x, y, state, radius);
        return 1;
    }
    return 0;
}

//signals that the eraser has been lifted from the surface. The erasing process has been paused.
//currently unused.
//Might be useful again if erasing and drawing are done with different tangibles, so I left it in.
int parseMessageFinishErase(char* buffer)
{
    int id;
    if(sscanf(buffer, "v %d ", &id) == 1)
    {
        finishErase(id);
    }
}

//display or change an image (that is not part of the UI). For comments, look at parseMessageUIElement() below. Both function very similarly.
//currently unused by the backend.
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
            img.loadTexture(filepath);
        }
        
        if (!is_known)
        {
            images.push_back(img);
        }

        return 1;

    }
    else return 0;
}

