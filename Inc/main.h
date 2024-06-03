#ifndef __main_h__
#define __main_h__

#include <iostream>
#include <algorithm>
#include <SDL2/SDL.h>
#include <stdlib.h>

#include "kernels.h"
#include "camera.h"

#define bool int
#define false 0u
#define true 1u

void mainLoop(SDL_Renderer*);
bool Input(entity&, camera&);
void Draw(SDL_Renderer*, entity, camera, std::vector<std::vector<float>>);
void flatShading(SDL_Renderer*, triangle);
void fillBottom(SDL_Renderer*, vec4*);
void fillTop(SDL_Renderer*, vec4*);
#endif