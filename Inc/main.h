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
void Draw(SDL_Renderer*, SDL_Texture*, entity, camera, u_int32_t*, float*, u_int32_t);

#endif