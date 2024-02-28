#include "main.h"

//@params: fields: a 2d vectors of vectors of length 3. 3-vectors contain RGB color values. Negative values are reserved for special codes: For example, a value of (-1, -1, -1) doesn't select a color, but the ERASE-tool instead.
//   ( r,  g,  b) = red, green and blue values
//   (-1, -1,  r) = switch to erase tool (r is the radius of the eraser)
//   (-2, -2, -2) = clear everything
//   (-3, -3, -3), (-4, -4, -4) etc... reserved for other actions (when we need them implemented)
    
Palette::Palette(vector<vector<vector<int>>> fields, int field_len_y, int field_len_x, int field_size){
    this->selectionIndicator = ImagePanel();
    this->fields = fields;
    this->field_len_x = field_len_x;
    this->field_len_y = field_len_y;
    this->fieldSize = field_size;
    this->setDimensions(field_len_x * field_size, field_len_y * field_size);
    this->selectionIndicator.setDimensions(field_size, field_size);
    
}

//overloading this function to set the texture of the indicator seems like the right call to me.
//there might be a cleaner solution here.
void Palette::loadTexture(char* texture_path, char * indicator_texture_path){
    this->selectionIndicator.loadTexture(indicator_texture_path);
    this->ImagePanel::loadTexture(texture_path);
}

void Palette::loadTexture()
{
    this->selectionIndicator.loadTexture();
    this->ImagePanel::loadTexture();
}

void Palette::setDefaultImagePath(char* imagePath, char* selectorImagePath)
{
    this->defaultImagePath = imagePath;
    this->selectionIndicator.setDefaultImagePath(selectorImagePath);
}

void Palette::select(int field_x, int field_y){
    bool doSetIndicatorPosition = true;
    vector<int> color_code = this->fields[field_y][field_x];
    if (color_code[0] >= 0 && color_code[1] >= 0 && color_code[2] >= 0){//no negative values => it's actually a color
        currentTool = pencil;
        currentColor = {color_code[0], color_code[1], color_code[2]};
    }
    else if(color_code[0] == -1){
        currentTool = eraser;
        eraserRadius = color_code[2];
    }
    else if(color_code[0] == -2){
        currentTool = clear;
        clearScreen();
        cout << "Cleared Screen" << endl;
        this->select(field_len_x-1, 0);
        doSetIndicatorPosition = false;
    }
    else{
        cout << "Unknown color code: {" << color_code[0] << " " << color_code[1] << " " << color_code[2] << "}" << endl;
    }
    if (doSetIndicatorPosition){
        Point newIndicatorPosition = {field_x * this->fieldSize + this->position.x, field_y * this->fieldSize + this->position.y};
        this->selectionIndicator.setPosition(newIndicatorPosition);
    }
}

void Palette::onClick(Point position){
    if (this->getVisibility()){
        Point relPos = this->getRelativeCoordinates(position);
        int idx_x = floor(relPos.x / this->fieldSize);
        int idx_y = floor(relPos.y / this->fieldSize);
        this->select(idx_x, idx_y);
    }
}

void Palette::render(SDL_Renderer* renderer){
    this->ImagePanel::render(renderer);
    this->selectionIndicator.render(renderer);
}

