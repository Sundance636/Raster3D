#include "main.h"

int main() {

    const u_int32_t WIDTH = 640;
    const u_int32_t HEIGHT = 480;

    

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

    //first start with local coordinates in object space
    std::vector<triangle> cube = {triangle(vec4(0,0,0,1),vec4(0,1,0,1),vec4(1,0,0,1)),
                        triangle(vec4(1,1,0,1),vec4(1,0,0,1),vec4(0,1,0,1)),

                        triangle(vec4(0,0,1,1),vec4(1,0,1,1),vec4(0,1,1,1)),
                        triangle(vec4(1,1,1,1),vec4(0,1,1,1),vec4(1,0,1,1)),

                        triangle(vec4(0,1,0,1),vec4(0,0,0,1),vec4(0,1,1,1)),
                        triangle(vec4(0,0,1,1),vec4(0,1,1,1),vec4(0,0,0,1)),

                        triangle(vec4(1,0,0,1),vec4(1,1,0,1),vec4(1,0,1,1)),
                        triangle(vec4(1,1,1,1),vec4(1,0,1,1),vec4(1,1,0,1)),

                        triangle(vec4(0,1,0,1),vec4(0,1,1,1),vec4(1,1,0,1)),
                        triangle(vec4(1,1,1,1),vec4(1,1,0,1),vec4(0,1,1,1)),

                        triangle(vec4(0,0,0,1),vec4(1,0,0,1),vec4(0,0,1,1)),
                        triangle(vec4(1,0,1,1),vec4(0,0,1,1),vec4(1,0,0,1))};

    entity testTriangle = entity(cube);


    testTriangle.setTriCount(12);
    std::vector<triangle> flat = { triangle(vec4(0,0,0,1),vec4(1,0,0,1),vec4(0,0,1,1)),
                        triangle(vec4(1,0,1,1),vec4(0,0,1,1),vec4(1,0,0,1))};
    
    

    entity plane = entity(flat);

    plane.setTriCount(2);
    plane.scaleEntity(vec4(200.0f,200.0f,200.0f,1.0f));
    plane.translateEntity(vec4(-59.0f,50.0f,150.0f,0.0f));


    camera cam = camera();//camera init in constructor


    //transform from local to world space
    testTriangle.scaleEntity(vec4(50.0f,50.0f,50.0f,1.0f));
    testTriangle.translateEntity(vec4(0.0f,0.0f,200.0f,0.0f));

    entity ship;
    ship.loadObj("Models/Sora.obj");
    
    ship.scaleEntity(vec4(50.0f,50.0f,50.0f,1.0f));
    ship.translateEntity(vec4(0.0f,0.0f,300.0f,0.0f));
    

    u_int32_t frameStart = 0;
    
    u_int32_t framerate = 1000.0f/60.0f;
    int itt = 0;
    float totaltime = 0;
    
    while(!gQuit) {

        gQuit = Input(testTriangle, cam);


        //bind drawing rate to desired framerate
        u_int32_t frameEnd = SDL_GetTicks();
        if(frameEnd - frameStart >= framerate) {
            SDL_RenderClear(renderer);
            SDL_SetRenderDrawColor(renderer, 255, 242, 242, 255);//white line
            
            Draw(renderer, plane, cam);
            //Draw(renderer,testTriangle, cam);
            Draw(renderer, ship, cam);

            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);//black background
            SDL_RenderPresent(renderer);

            //float elapsed = (frameEnd - frameStart) ;
	        //std::cout << "Current FPS: " << 1000.0/elapsed << std::endl;

            frameStart = SDL_GetTicks();
            float elapsed = (frameStart - frameEnd);
            totaltime += elapsed;
	        std::cout << "Rendered In: " << elapsed << "ms\n" << std::endl;
            ++itt;
        }

    }
    if(itt != 0) {
        std::cout << "Average Time to render: " << totaltime/itt << "ms" << std::endl;
        std::cout << "Target: " << 1000.0f/60.0f << "ms\n" << std::endl;

    }



}


void Draw(SDL_Renderer *renderer,entity testTri, camera cam) {
    int WIDTH = 640;
    int HEIGHT = 480;

    entity projection = entity(testTri);

    //OPTIMIZE INTO ONE KERNEL CALL  LATER
    cam.viewTransformR(projection);
    cam.perspectiveProjectionR(projection);


    //part of coordinate conversion (screen space)
    projection.translateEntity(vec4(1.0f,1.0f,0.0f,0.0f));
    projection.scaleEntity(vec4(WIDTH* 0.5f,1.0f,1.0f,1.0f));
    projection.scaleEntity(vec4(1.0f,HEIGHT*0.5f,1.0f,1.0f));
    
    
    //calculating normals for each tri and drawing
    for(int i = 0; i < testTri.getTriCount(); i ++ ) {
        
        vec4 eyeLine =  cam.getPosition( ) - testTri[i].getP3();
        eyeLine.sety(cam.getPosition().y() -  -1.0f*testTri[i].getP3().y() );
        eyeLine = unit_vector4(eyeLine);
        eyeLine.setx(-eyeLine.x());
        eyeLine.setz(-eyeLine.z());
        eyeLine.setw(0);

        float view = dot_product4(testTri[i].getSurfaceNormal(), eyeLine);
        
        if( view < 0.0) {
                //std::cout << "\n Surface Normal skip: " << testTri[i].getSurfaceNormal() << "\n";
            
            //std::cout << "\n Face Ratio" << i << ": "<< view << '\n';

            SDL_SetRenderDrawColor(renderer, 150 * -view, 150*-view, 150*-view,  150*-view);//white line
            flatShading(renderer, projection[i]);

            SDL_RenderDrawLine(renderer,projection[i].getP1().x(),projection[i].getP1().y(),projection[i].getP2().x(), projection[i].getP2().y());
            SDL_RenderDrawLine(renderer,projection[i].getP1().x(),projection[i].getP1().y(),projection[i].getP3().x(), projection[i].getP3().y());
            SDL_RenderDrawLine(renderer,projection[i].getP3().x(),projection[i].getP3().y(),projection[i].getP2().x(), projection[i].getP2().y());

        }
        
    }
    
}

bool Input(entity &test, camera &cam) {
    SDL_Event e;

    while(SDL_PollEvent(&e) != 0) {
        if(e.type == SDL_QUIT) {
                printf("Quiting Window.\n");
                return true;
        } else if(e.type == SDL_KEYDOWN) {


            //transforms in terms of world space
            if(e.key.keysym.sym == SDLK_LEFT) {
                vec4 direction = vec4(-3.0,0.0,0.0,1.0);
                rotationY(cam.getLookAngle(),direction);
                direction.setw(0);
                cam.movecam(direction);
            }
            else if((e.key.keysym.sym == SDLK_RIGHT)) {
                vec4 direction = vec4(3.0,0.0,0.0,1.0);
                rotationY(cam.getLookAngle(),direction);
                direction.setw(0);
                cam.movecam(direction);
            }
            else if((e.key.keysym.sym == SDLK_UP)) {
                vec4 direction = vec4(0.0,3.0,0.0,1.0);
                rotationY(-cam.getLookAngle(),direction);
                direction.setw(0);
                cam.movecam(direction);
            }
            else if((e.key.keysym.sym == SDLK_DOWN)) {
                vec4 direction = vec4(0.0,-3.0,0.0,1.0);
                rotationY(-cam.getLookAngle(),direction);
                direction.setw(0);
                cam.movecam(direction);
            }
            else if((e.key.keysym.sym == SDLK_e)) {
                vec4 direction = vec4(0.0,0.0,8.5,1.0);
                rotationY(cam.getLookAngle(),direction);
                direction.setw(0);
                cam.movecam(direction);
            }
            else if((e.key.keysym.sym == SDLK_q)) {
                vec4 direction = vec4(0.0,0.0,-8.5,1.0);
                rotationY(cam.getLookAngle(),direction);
                direction.setw(0);
                cam.movecam(direction);
            }
            else if(e.key.keysym.sym == SDLK_d) {
                cam.rotateLook(0.04f);
            }
            else if(e.key.keysym.sym == SDLK_a) {
                cam.rotateLook(-0.04f);
            }
            else if(e.key.keysym.sym == SDLK_w) {
                cam.rotateUp(0.04f);
            }
            else if(e.key.keysym.sym == SDLK_s) {
                cam.rotateUp(-0.04f);
            }
            else if(e.key.keysym.sym == SDLK_y) {
                test.translateEntity(vec4(0.0f,20.0f,0.0f,0.0f));
            }
            

            std::cout << "Point 1: " << test[0].getP1() << "\n";
            std::cout << "Point 2: " << test[0].getP2() << "\n";
            std::cout << "Point 3: " << test[0].getP3() << "\n";
            std::cout << "Look Angle: " << cam.getLookAngle() << "\n";
            std::cout << "Look Vec: " << cam.getLookVec() << "\n";
            std::cout << "Up Angle: " << cam.getUpAngle() << "\n";
            std::cout << "Up Vec: " << cam.getUpVec() << "\n";
            //std::cout << "\n Surface Normal: " << test[0].getSurfaceNormal() << "\n";
            std::cout << "\n Camera Direction: " << cam.direction() << "\n";
            std::cout << "Camera Position: " << cam.getPosition() << "\n";

            for(int i = 0; i < test.getTriCount(); i++) {
                //std::cout << "\n Surface Normal: " << test[i].getSurfaceNormal();
            }

            printf("KeyDown\n");
        }
    }
    return false;
}

void flatShading(SDL_Renderer* renderer, triangle tri) {
    
    vec4 sorted[3] = {tri.getP1(),tri.getP2(),tri.getP3()};

    auto compareY = [](const vec4& a, const vec4& b) {
        return a.y() < b.y();
    };

    std::sort(std::begin(sorted), std::end(sorted), compareY);

    if( sorted[1].y() == sorted[2].y() ) {
        fillBottom(renderer, sorted);
    }
    else if(sorted[0].y() == sorted[1].y()) {
        fillTop(renderer, sorted);
    }
    else {
        vec4 v4 = vec4(
      (sorted[0].x() + ((float)(sorted[1].y() - sorted[0].y()) / (float)(sorted[2].y() - sorted[0].y())) * (sorted[2].x() - sorted[0].x())), sorted[1].y(),0.0f,0.0f);
      vec4 sorted2[3] = {sorted[0],sorted[1],v4};
        fillBottom(renderer, sorted2);
        vec4 sorted3[3] = {sorted[1],v4,sorted[2]};
        fillTop(renderer, sorted3);
    }

}

void fillBottom( SDL_Renderer* renderer , vec4 sorted[]) {
    float invslope1 = (sorted[1].x() - sorted[0].x()) / (sorted[1].y() - sorted[0].y());
    float invslope2 = (sorted[2].x() - sorted[0].x()) / (sorted[2].y() - sorted[0].y());

    float curx1 = sorted[0].x();
    float curx2 = sorted[0].x();

    for (int scanlineY = sorted[0].y(); scanlineY <= sorted[1].y(); scanlineY++)
    {
        SDL_RenderDrawLine(renderer, curx1, scanlineY, (int)curx2, scanlineY);
        curx1 += invslope1;
        curx2 += invslope2;
    }
}

void fillTop( SDL_Renderer* renderer , vec4 sorted[]) {
    float invslope1 = (sorted[2].x() - sorted[0].x()) / (sorted[2].y() - sorted[0].y());
    float invslope2 = (sorted[2].x() - sorted[1].x()) / (sorted[2].y() - sorted[1].y());

    float curx1 = sorted[2].x();
    float curx2 = sorted[2].x();

    for (int scanlineY = sorted[2].y(); scanlineY > sorted[0].y(); scanlineY--)
    {
        SDL_RenderDrawLine(renderer, curx1, scanlineY, (int)curx2, scanlineY);
        curx1 -= invslope1;
        curx2 -= invslope2;
    }
}