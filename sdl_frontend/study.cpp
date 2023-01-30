#include "main.h"
#include "study.h"
#include <fstream>

void loadPhrases()
{
    string line;
    ifstream phraseFile(PHRASES_PATH);

    if(phraseFile.is_open())
    {
        while(getline(phraseFile, line))
        {
            line.pop_back();
            phrases.push_back(line);
        }
        phraseFile.close();
    }
}

void nextPhrase()
{
    bool newPhraseFound = false;
    int phraseIndex = 0;

    while(!newPhraseFound)
    {
        newPhraseFound = true;
        phraseIndex = rand() % phrases.size();

        for (auto const& index : usedPhrases)
        {
            if(index == phraseIndex)
            {
                newPhraseFound = false;
                break;
            }
        }
    }

    usedPhrases.push_back(phraseIndex);
    currentPhrase = phrases.at(phraseIndex);
    //cout << phraseIndex << " " << currentPhrase << endl;

    currentTextSize = (currentTextSize + 1) % 3;

    textSurface = TTF_RenderText_Solid( font, currentPhrase.c_str(), textColor );
    textTexture = SDL_CreateTextureFromSurface( renderer, textSurface );
}
