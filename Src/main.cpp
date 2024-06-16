#include "main.h"

//Branch has no hardware acceleration
int main() {

//Initialize SDL stuff
    SDL_Window *applicationWindow;
    SDL_Renderer* renderer;
    const u_int32_t WIDTH = 640;
    const u_int32_t HEIGHT = 480;


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

    mainLoop(renderer);


    //cleaning and quit routine
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(applicationWindow);
    SDL_Quit();

    return 0;
}

void mainLoop(SDL_Renderer *renderer) {

    bool gQuit = false;
    camera cam = camera();//camera init in constructor

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

    //transform from local to world space
    testTriangle.scaleEntity(vec4(50.0f,50.0f,50.0f,1.0f));
    testTriangle.translateEntity(vec4(0.0f,0.0f,200.0f,0.0f));


    entity plane;
    plane.loadObj("Models/flatPlane.obj");
    plane.scaleEntity(vec4(50.0f,50.0f,50.0f,1.0f));
    plane.translateEntity(vec4(0.0f,0.0f,300.0f,0.0f));

    entity ship;
    ship.loadObj("Models/sphere.obj");
    ship.scaleEntity(vec4(50.0f,50.0f,50.0f,1.0f));
    ship.translateEntity(vec4(0.0f,0.0f,400.0f,0.0f));
    

    u_int32_t frameStart = 0;
    
    float framerate = 1000.0f/60.0f;
    u_int32_t itt = 0;
    float totaltime = 0;

    int WIDTH = 640;
    int HEIGHT = 480;

    SDL_Texture* texture = SDL_CreateTexture(renderer,SDL_PIXELFORMAT_ARGB8888,SDL_TEXTUREACCESS_STREAMING,WIDTH,HEIGHT);
    u_int32_t* frameBuffer = new u_int32_t[WIDTH * HEIGHT];
    float* depthBuffer = new float[WIDTH * HEIGHT];

    u_int32_t t = 0; 
    
    while(!gQuit) {

        gQuit = Input(testTriangle, cam);

        //bind drawing rate to desired framerate
        u_int32_t frameEnd = SDL_GetTicks();
        if(frameEnd - frameStart >= framerate) {
            SDL_RenderClear(renderer);
            SDL_SetRenderDrawColor(renderer, 255, 242, 242, 255);//white line

            //presumably clean buffers each frame
            for(int i = 0; i < WIDTH * HEIGHT; i++) {
                frameBuffer[i] = 0u; // Black color
                depthBuffer[i] = std::numeric_limits<float>::infinity();
            }

            Draw(renderer, texture, plane, cam, frameBuffer, depthBuffer, t);
            Draw(renderer,texture, testTriangle, cam, frameBuffer,depthBuffer, t);
            Draw(renderer, texture, ship, cam, frameBuffer, depthBuffer, t);

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
        std::cout << "Target: " << framerate << "ms\n" << std::endl;

    }

    delete[] frameBuffer;
    delete[] depthBuffer;

}


void Draw(SDL_Renderer *renderer, SDL_Texture* texture, entity testTri, camera cam, u_int32_t* frameBuffer, float* depthBuffer, u_int32_t t) {
    int WIDTH = 640;
    int HEIGHT = 480;
    int count = 0;

    std::vector<float> facingRatios = std::vector<float>(testTri.getTriCount(),0);

    entity projection = entity(testTri);

    //apply sine wave
    //projection.applyWave();

    //OPTIMIZE INTO ONE KERNEL CALL  LATER
    cam.viewTransformR(projection);

    //calculate normals/facing ratios for backface culling
    cam.faceCulling( facingRatios, testTri );

    //frustum culling for each triangle
    cam.frustumCulling(facingRatios, projection);

    //should probably rewrite later so this ACTUALLY benefits from early culling
    cam.perspectiveProjectionR(facingRatios, projection);

    //part of coordinate conversion (screen space)
    projection.translateEntity(vec4(1.0f,1.0f,0.0f,0.0f));
    projection.scaleEntity(vec4(WIDTH* 0.5f,1.0f,1.0f,1.0f));
    projection.scaleEntity(vec4(1.0f,HEIGHT*0.5f,1.0f,1.0f));
  

     for(int i = 0; i < testTri.getTriCount(); i ++ ) {

        if( facingRatios[i] < 0.0) {
            ++count;

            //get bounding box for current triangle
            float boxMinX = std::min(std::min(projection[i].getP1().x(),projection[i].getP2().x()),projection[i].getP3().x());
            float boxMaxX = std::max(std::max(projection[i].getP1().x(),projection[i].getP2().x()),projection[i].getP3().x());

            float boxMinY = std::min(std::min(projection[i].getP1().y(),projection[i].getP2().y()),projection[i].getP3().y());
            float boxMaxY = std::max(std::max(projection[i].getP1().y(),projection[i].getP2().y()),projection[i].getP3().y());
            
            projection[i].hitTest(boxMinX, boxMaxX, boxMinY, boxMaxY, WIDTH, HEIGHT,frameBuffer, depthBuffer, -facingRatios[i]);
            
            //rendering bounding box
            //SDL_RenderDrawLine(renderer,boxMinX,boxMinY,boxMaxX, boxMinY);
            //SDL_RenderDrawLine(renderer,boxMaxX,boxMinY,boxMaxX, boxMaxY);


            //wireframe
            //SDL_RenderDrawLine(renderer,projection[i].getP1().x(),projection[i].getP1().y(),projection[i].getP2().x(), projection[i].getP2().y());
            //SDL_RenderDrawLine(renderer,projection[i].getP1().x(),projection[i].getP1().y(),projection[i].getP3().x(), projection[i].getP3().y());
            //SDL_RenderDrawLine(renderer,projection[i].getP3().x(),projection[i].getP3().y(),projection[i].getP2().x(), projection[i].getP2().y());
        }

    }
    
    
    std::cout << "Tris Rendered: " << count << " / " << testTri.getTriCount() << "\n";

    //texture stuff
    SDL_UpdateTexture(texture,nullptr,frameBuffer, WIDTH* sizeof(u_int32_t));
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    
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
            /*
            std::cout << "Look Angle: " << cam.getLookAngle() << "\n";
            std::cout << "Look Vec: " << cam.getLookVec() << "\n";
            std::cout << "Up Angle: " << cam.getUpAngle() << "\n";
            std::cout << "Up Vec: " << cam.getUpVec() << "\n";
            std::cout << "\n Camera Direction: " << cam.direction() << "\n";
            std::cout << "Camera Position: " << cam.getPosition() << "\n";

            std::cout << "KeyDown\n";
            */
        }
    }
    return false;
}