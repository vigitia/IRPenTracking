#include "main.h"
#include "document.h"

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
    }

    return 0;
}

int parseMessageLine(char* buffer)
{
    int id, x, y, state;
    unsigned int r;
    unsigned int g;
    unsigned int b;
    // parse new values from the FIFO
    // only set the delay times if all four values could be read correctly
    if(sscanf(buffer, "l %d %u %u %u %d %d %d ", &id, &r, &g, &b, &x, &y, &state) == 7)
    {
        if(pens.find(id) == pens.end())
        {
            struct Pen currentPen;
            //currentPen.id = id;
            currentPen.state = (bool) state;
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

        return 1; 
    }

    return 0;
}

int parseMessageFinishLine(char* buffer)
{

    int id;
    // parse new values from the FIFO
    // only set the delay times if all four values could be read correctly
    if(sscanf(buffer, "f %d ", &id) == 1)
    {
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
    // parse new values from the FIFO
    // only set the delay times if all four values could be read correctly
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
    if(sscanf(buffer, "d %d ", &id) == 1)
    {
        for (vector<Line>::iterator it = lines.begin(); it != lines.end(); )
        {

            if(it->id == id) 
                it = lines.erase(it);
            else 
                ++it;
        }

        for (vector<Line>::iterator it = documentLines.begin(); it != documentLines.end(); )
        {

            if(it->id == id) 
                it = documentLines.erase(it);
            else 
                ++it;
        }

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

