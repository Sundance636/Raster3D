#ifndef __main_h__
#define __main_h__

#include <iostream>
#include <SDL2/SDL.h>
#include <GL/gl.h>

#include "kernels.h"
#include <stdlib.h>
#include "camera.h"

#define bool int
#define false 0u
#define true 1u

void mainLoop(SDL_Renderer*);
bool Input(entity&);
void Draw(SDL_Renderer*, entity);

#endif