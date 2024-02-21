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
    this->fieldSize = fieldSize;
    this->setDimensions(field_len_x * fieldSize, field_len_y * fieldSize);

}

//overloading this function to set the texture of the indicator seems like the right call to me.
//there might be a cleaner solution here.
void Palette::loadTexture(char* texture_path, char * indicator_texture_path){
    this->selectionIndicator.loadTexture(indicator_texture_path);
    this->ImagePanel::loadTexture(texture_path);
}

void Palette::select(int field_x, int field_y){
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
    }
    else{
        cout << "Unknown color code: {" << color_code[0] << " " << color_code[1] << " " << color_code[2] << "}"
    }

    Point newIndicatorPosition = {field_x * this->fieldSize + this->position.x, field_y * this->fieldSize + this->position.x}
    this->selectionIndicator.setPosition(newIndicatorPosition)
}

void Palette::onClick(Point position){
    if (this->getVisibility()){
        Point relPos = this->getRelativeCoordinates(position);
        float idx_x = relPos.x / this->fieldSize;
        float idx_y = relPos.y / this->fieldSize;
        this->select(idx_x, idx_y)
    }
}

