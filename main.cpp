#include "main.h"

int main() {

    const int WIDTH = 640;
    const int HEIGHT = 480;

    

    //Initialize SDL stuff
    SDL_Window *applicationWindow;
    SDL_Renderer* renderer;

    if((SDL_Init(SDL_INIT_VIDEO|SDL_INIT_AUDIO)==-1)) { 
        printf("Could not initialize SDL: %s.\n", SDL_GetError());
        exit(-1);
    }

    applicationWindow = SDL_CreateWindow("Raster Engine" , SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED, WIDTH,HEIGHT, SDL_WINDOW_SHOWN);

    if ( applicationWindow == NULL ) {
        fprintf(stderr, "Couldn't set 640x480x8 video mode: %s\n",
                        SDL_GetError());
        exit(1);
    }

    renderer = SDL_CreateRenderer(applicationWindow , 0, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    if(renderer == NULL) {
        fprintf(stderr, "Couldn't create context: %s\n",
                        SDL_GetError());
        exit(2);
    }

    // allocate space for Frame Buffer

    

    // Render our buffer
    //renderBuffer(d_fb, tx, ty, d_world);
    //transferMem(h_fb, d_fb);//transfer mem from device to host
   /* */

    mainLoop(renderer);

    //deallocates
    //freeGPU(d_fb,d_list,d_world);
    //free(h_fb);
    

    //cleaning and quit routine
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(applicationWindow);
    SDL_Quit();

    return 0;
}

void mainLoop(SDL_Renderer *renderer) {

    bool gQuit = false;

    //triangle* tris = new triangle(vec4(100,100,100,1),vec4(140,180,140,1),vec4(180,140,140,1));

    triangle cube[12] = {triangle(vec4(100,100,100,1),vec4(100,200,100,1),vec4(200,200,100,1)),
                        triangle(vec4(100,100,100,1),vec4(200,100,100,1),vec4(200,200,100,1)),
                        triangle(vec4(100,100,200,1),vec4(200,100,200,1),vec4(200,200,200,1)),
                        triangle(vec4(100,100,200,1),vec4(100,200,200,1),vec4(200,200,200,1)),
                        triangle(vec4(100,100,100,1),vec4(100,200,100,1),vec4(100,200,200,1)),
                        triangle(vec4(100,100,100,1),vec4(100,100,200,1),vec4(100,200,200,1)),
                        triangle(vec4(200,100,100,1),vec4(200,200,100,1),vec4(200,200,200,1)),
                        triangle(vec4(200,100,100,1),vec4(200,100,200,1),vec4(200,200,200,1)),
                        triangle(vec4(100,100,100,1),vec4(100,100,200,1),vec4(200,100,100,1)),
                        triangle(vec4(200,100,100,1),vec4(200,100,200,1),vec4(100,100,200,1)),
                        triangle(vec4(200,200,100,1),vec4(200,200,200,1),vec4(100,200,200,1)),
                        triangle(vec4(200,200,100,1),vec4(100,200,100,1),vec4(100,200,200,1))};
    entity testTriangle = entity(cube);
    //tris[0];//
    testTriangle.setTriCount(12);

    

    
    while(!gQuit) {

        gQuit = Input(testTriangle);
        Draw(renderer,testTriangle);

    }

    //delete tris;


}

/*
void filltri(entity testtri) {
    float bottomval = std::min(std::min(testtri[0].getP1().y(),testtri[0].getP2().y()), testtri[0].getP3().y());
    float limit = std::max(std::max(testtri[0].getP1().y(),testtri[0].getP2().y()), testtri[0].getP3().y());

    float leftval = std::min(std::min(testtri[0].getP1().x(),testtri[0].getP2().x()), testtri[0].getP3().x());
    float limitU = std::max(std::max(testtri[0].getP1().x(),testtri[0].getP2().x()), testtri[0].getP3().x());

    for(int v = bottomval; v < limit; v++ ) {
        for(int u = leftval; u < limitU; u++) {
            if();
        }
    }

}*/

void Draw(SDL_Renderer *renderer,entity testTri) {
    //glDrawPixels(640,480,GL_RGB,GL_FLOAT,fb);
    SDL_RenderClear(renderer);
    SDL_SetRenderDrawColor(renderer, 255, 242, 242, 255);//white line

    entity projection = entity(testTri);

    projection.translateEntity(vec4(0.0f,-100.0f,-600.0f,0.0f));

    camera cam = camera();

    for(int i = 0; i < testTri.getTriCount(); i++) {

        //std::cout << "Point 1: " << cam.perspectiveProjection(testTri[i].getP1()) << "\n";

        projection[i].setP1(cam.perspectiveProjection(projection[i].getP1()));
        projection[i].setP2(cam.perspectiveProjection(projection[i].getP2()));
        projection[i].setP3(cam.perspectiveProjection(projection[i].getP3()));

        std::cout << "NDC Point 1: " << projection[0].getP1() << "\n";
        std::cout << "NDC Point 2: " << projection[0].getP2() << "\n";
        std::cout << "NDC Point 3: " << projection[0].getP3() << "\n";



    }

    int WIDTH = 640;
    int HEIGHT = 480;


    //part of coordinate conversion
    projection.translateEntity(vec4(1.0f,1.0f,0,0));
    projection.scaleEntity(vec4(WIDTH* 0.5f,1.0f,1.0f,1.0f));
    projection.scaleEntity(vec4(1.0f,HEIGHT*0.5f,1.0f,1.0f));
    

    

    
    for(int i = 0; i < testTri.getTriCount(); i ++ ) {
        SDL_RenderDrawLine(renderer,projection[i].getP1().x(),projection[i].getP1().y(),projection[i].getP2().x(), projection[i].getP2().y());
        SDL_RenderDrawLine(renderer,projection[i].getP1().x(),projection[i].getP1().y(),projection[i].getP3().x(), projection[i].getP3().y());
        SDL_RenderDrawLine(renderer,projection[i].getP3().x(),projection[i].getP3().y(),projection[i].getP2().x(), projection[i].getP2().y());

        std::cout << "\nTri number: " << i ;
        std::cout << "\nPoint 1: " << projection[i].getP1() << "\n";
        std::cout << "Point 2: " << projection[i].getP2() << "\n";
        std::cout << "Point 3: " << projection[i].getP3() << "\n";

    }

    


    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);//black background
    SDL_RenderPresent(renderer);

}

bool Input(entity &test) {
    SDL_Event e;

    while(SDL_PollEvent(&e) != 0) {
        if(e.type == SDL_QUIT) {
                printf("Quiting Window.\n");
                return true;
        } else if(e.type == SDL_KEYDOWN) {


            //transforms in terms of world space
            //test.translateEntity(vec4(40,40,40,1));
            test.rotateEntityY(.04f);
            test.rotateEntityZ(.01f);

            //handle projections wtih NDC

            std::cout << "Point 1: " << test[0].getP1() << "\n";
            std::cout << "Point 2: " << test[0].getP2() << "\n";
            std::cout << "Point 3: " << test[0].getP3() << "\n";
            printf("KeyDown\n");
        }
    }
    return false;
}