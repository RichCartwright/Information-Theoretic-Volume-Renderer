/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Volume rendering sample

    This sample loads a 3D volume from disk and displays it using
    ray marching and 3D textures.

    Note - this is intended to be an example of using 3D textures
    in CUDA, not an optimized volume renderer.

    Changes
    sgg 22/3/2010
    - updated to use texture for display instead of glDrawPixels.
    - changed to render from front-to-back rather than back-to-front.
*/

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>


// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>

#include "entropy/Entropy.h"

// Socket and learning stuff
#include "socket.h"
#include <sys/socket.h>
#include <arpa/inet.h>
using namespace serversock;
struct serversock::objectData data;

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "util/stb_image_write.h"

#include <iostream>
#include <fstream>

typedef unsigned int uint;
typedef unsigned char uchar;

unsigned int* pVolumeDataHist = nullptr;    // This is the data after transfer function - the ray marches 
unsigned int* pRawDataHist = nullptr;       // The raw data

float entropyA = 0.f, entropyB = 0.f, jointEntropy = 0.f;
float mutualInformation = 0.f;
float scale = 0.001f; // This is for scaling the histogram renders - there are many smarter ways to do this

size_t BIN_COUNT = 511;              // This crashes at 512, has to be *2-1, I think
size_t histSize = sizeof(unsigned int) * BIN_COUNT;
size_t histSizeCache = 0;
float DataRange[2] = {0.f,0.f}; 
float highestMI = 0.0f;
bool LOG_FLAG = false;
bool LOG_FILE_WRITTEN = false;

Entropy* Entropy::instance = 0;
Entropy* entropyHelper = entropyHelper->getInstance(); // Static instance

GLint *windowID = nullptr; 

#define MAX_EPSILON_ERROR 5.00f
#define THRESHOLD         0.30f

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "volume.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_volume.ppm",
    NULL
};

const char *sSDKsample = "CUDA 3D Volume Render";

const char *volumeFilename = "Bucky.raw";
cudaExtent volumeSize = make_cudaExtent(32, 32, 32);
typedef unsigned char VolumeType;

//char *volumeFilename = "mrt16_angio.raw";
//cudaExtent volumeSize = make_cudaExtent(416, 512, 112);
//typedef unsigned short VolumeType;

uint width = 512, height = 512;
uint statsWidth = width, statsHeight = 512;
dim3 blockSize(16, 16);
dim3 gridSize;

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

float density           = 0.05f;
float brightness        = 1.0f;
float transferOffset    = 0.0f;
float transferScale     = 1.0f;
bool linearFiltering    = true;

// Socket stuff
struct sockaddr_in server; 
int sock;
bool serverFailed = false;

std::ofstream* outputFile = nullptr;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint _tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface *timer = 0;

// Auto-Verification Code
const int frameCheckNumber  = 2;
int fpsCount                = 0;    // FPS count for averaging
int fpsLimit                = 1;    // FPS limit for sampling
int g_Index                 = 0;
unsigned int frameCount     = 0;

int *pArgc;
char **pArgv;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern "C" void setTextureFilterMode(bool bLinearFilter);
extern "C" void initCuda(void *h_volume, cudaExtent volumeSize);
extern "C" void freeCudaBuffers();
extern "C" void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                              float density, float brightness, float transferOffset, float transferScale, uint* pVolumeDataDist, size_t histSize);
extern "C" void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);

void dirtyDrawBitmapString(int x, int y, const char* string, GLfloat* colour = nullptr, void* font = GLUT_BITMAP_TIMES_ROMAN_24);
void dirtyDrawBitmapString(float x, float y, const char* string, GLfloat* colour = nullptr, void* font = GLUT_BITMAP_TIMES_ROMAN_24);
void initPixelBuffer();

void SendToServer(char* message = nullptr);
void ListenToServer();

void SetupServerConnection(char* addr = "127.0.0.1", int port = 8888)
{
    close(sock);
    //std::cout << "[CLIENT]: Connecting to server" << std::endl; 
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if(sock == -1)
    {
        std::cout << "[CLIENT]: Could not create socket for some reason" << std::endl;
    }
    else
    {
        std::cout << "[CLIENT]: Socket created with code " << sock << std::endl;
    }
    
    if(addr)
        server.sin_addr.s_addr = inet_addr(addr);
    else
        server.sin_addr.s_addr = inet_addr("127.0.0.1");

    server.sin_family = AF_INET;
    server.sin_port = htons(port);

    //Connect to remote server
    if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0)
    {
        perror("[CLIENT]: Connect failed. Error");
        // This is a bit of a temp hack, if the server doesn't connect we will never listen for it again
        serverFailed = true;
        return; 
    }

    //SendToServer((char*)"[CLIENT]: Connected to server\n");
}

void SendToServer(char* message)
{
    int i=0;
    
    if(message == nullptr) //then well post the MI information
    {
        // for the sake of ease, ill just set the timestep to zero 
        float MI_arr[5];
        MI_arr[0] = 0;
        MI_arr[1] = viewRotation.x;
        MI_arr[2] = viewRotation.y;
        MI_arr[3] = viewTranslation.z;
        MI_arr[4] = mutualInformation;
        char const * p = reinterpret_cast<char const *>(MI_arr); 
        std::string s(p, p + sizeof MI_arr);
        //std::cout << s << std::endl;
        write(sock, &s[0], sizeof(MI_arr));
        ListenToServer();
    }
    else
    {
        write(sock,message,strlen(message));
    }
    SetupServerConnection();
}

void ListenToServer()
{
    int i;
    char server_reply[2000] = {0};
    ssize_t len;
    
    //std::cout << "[CLIENT]: Waiting for reply\n" << std::endl; 
    if( (len = recv(sock, server_reply, 2000, 0)) < 0 )
    {
        std::cout << "[CLIENT]: Recv Failed\n" << std::endl; 
        close(sock);
    }
    
    // This is crazy unsafe - look into using protobuffs
    float* unpacked = (float*)server_reply; 
    printf("[SERVER]: %f, %f, %f\n", unpacked[0], unpacked[1], unpacked[2]);
    viewRotation.x = unpacked[0];
    viewRotation.y = unpacked[1];
    viewTranslation.z = unpacked[2];
    for (i=0; i< strlen(server_reply); i++)
    {
      server_reply[i] = '\0';
    }
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        char fps[256];
        float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        sprintf(fps, "Volume Render: %3.1f fps", ifps);

        glutSetWindowTitle(fps);
        fpsCount = 0;

        fpsLimit = (int)MAX(1.f, ifps);
        sdkResetTimer(&timer);
    }
}

//
void dirtyDrawBitmapString(int x, int y, const char* string, GLfloat* colour, void* font )
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, width, height, 0, -100.0, 100.0);
    (colour) ? glColor4fv(colour) : glColor4f(1.f,1.f,1.f,1.f); // Just make it white if no pointer is passed
    glRasterPos2i(x,y);
    const char* character = string;

    while(*character != '\0')
    {
        glutBitmapCharacter(font, *character);
        character++;
    }
}

void dirtyDrawBitmapString(float x, float y, const char* string, GLfloat* colour, void* font )
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -100.0, 100.0);
    (colour) ? glColor4fv(colour) : glColor4f(1.f,1.f,1.f,1.f); // Just make it white if no pointer is passed
    glRasterPos2f(x,y);
    const char* character = string;

    while(*character != '\0')
    {
        glutBitmapCharacter(font, *character);
        character++;
    }
}

// render image using CUDA
void render()
{
    // Not really needed here, but if the bin count changes, we need to reallocate
    histSize = sizeof(uint)*BIN_COUNT; 
    if(histSizeCache != histSize)
    {
        printf("Allocating Volume Data Histogram of size %li\n", histSize);
        checkCudaErrors(cudaFree(pVolumeDataHist));
        checkCudaErrors(cudaMallocManaged(&pVolumeDataHist, histSize));
        histSizeCache = histSize;
    }

    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);

    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
                                                         cuda_pbo_resource));
    //printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width*height*4));

    // call CUDA kernel, writing results to PBO
    for(int i = 0; i < BIN_COUNT + 1; i++)
    {
        pVolumeDataHist[i] = 0;
    }
    render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, pVolumeDataHist, histSize);
    cudaDeviceSynchronize();
    getLastCudaError("kernel failed");
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

}

// display results using OpenGL (called by GLUT)
void display()
{    
    sdkStartTimer(&timer);

    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    render();

    // display results
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // draw using texture
    // copy from pbo to texture
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, _tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -100.0, 100.0);

    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_TRIANGLE_STRIP);
        glColor3f(1,1,1);
        glTexCoord2f(0, 0);
        glVertex2f(0, 0);
        glTexCoord2f(1, 0);
        glVertex2f(1, 0);
        glTexCoord2f(0, 1);
        glVertex2f(0, 1);
        glTexCoord2f(1, 1);
        glVertex2f(1, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glScalef(0.07, 0.5, 1.0);
    glBegin(GL_TRIANGLE_STRIP);
        glColor3f(1,0,0);
        glVertex2f(0.1, ((0.9-0.1)/(6))*1);
        glVertex2f(1,   ((0.9-0.1)/(6))*1);
        
        glColor3f(1,0.5,0);
        glVertex2f(0.1, ((0.9-0.1)/(6))*2);
        glVertex2f(1,   ((0.9-0.1)/(6))*2);

        glColor3f(1,1,0);
        glVertex2f(0.1, ((0.9-0.1)/(6))*3);
        glVertex2f(1,   ((0.9-0.1)/(6))*3);

        glColor3f(0,1,0);
        glVertex2f(0.1, ((0.9-0.1)/(6))*4);
        glVertex2f(1,   ((0.9-0.1)/(6))*4);

        glColor3f(0,1,1);
        glVertex2f(0.1, ((0.9-0.1)/(6))*5);
        glVertex2f(1,   ((0.9-0.1)/(6))*5);

        glColor3f(0,0,1);
        glVertex2f(0.1, ((0.9-0.1)/(6))*6);
        glVertex2f(1,   ((0.9-0.1)/(6))*6);

        glColor3f(1,0,1);
        glVertex2f(0.1, ((0.9-0.1)/(6))*7);
        glVertex2f(1,   ((0.9-0.1)/(6))*7);
    glEnd();


    if(pVolumeDataHist != nullptr)
    {
        // These shouldn't really be hardcoded but...
        float bottom = 0.1f, top = 0.9f;
        float difference = top - bottom;
        float step = difference / BIN_COUNT;
        for(size_t i = 0; i < BIN_COUNT + 1; ++i)
        {
            float barY = 0.137f + (step * i);
            glLineWidth(4.0f);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
            glEnable( GL_BLEND ); 

            glBegin(GL_LINES);
                glColor4f(1.f, 1.f, 1.f, 1.f); 
                glVertex2f(1.f, barY);
                glVertex2f((float)pVolumeDataHist[i]*scale + 1.f, barY);
            glEnd();
        }
    }

    glPopMatrix();

    glClearColor(.5f, .5, .5f, 1.0f);
    glutSwapBuffers();
    glutReportErrors();

    sdkStopTimer(&timer);

    computeFPS();
}

void DisplayStatsWindow()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glLoadIdentity();

    // Stats draw
    //drawHistograms();
    //displaySettings();

    glLineWidth(2.0f);
    if(pRawDataHist != nullptr)
    {
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, width, height, 0, -100.0, 100.0);


        float left = ((width/4)) + 35.f, right = (left + 200);
        float bottom = 350.f;
        float difference = right - left;
        float step = difference / BIN_COUNT;

        for(size_t i = 0; i < BIN_COUNT; ++i)
        {
            GLfloat white[4] = {1.f, 1.f, 1.f, 1.f};
            glColor4fv(white);
            float barX = left + (step * i);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable( GL_BLEND ); 

            glBegin(GL_LINES);
                glVertex2f(barX, bottom);
                glVertex2f(barX, bottom - ((float)pRawDataHist[i]*0.1f));
            glEnd();
        }
    }

    glutSwapBuffers();
    glutReportErrors();
    glClearColor(.5f, .5f, .5f, 1.f);
}

void idle()
{
    entropyHelper->GetEntropy(pVolumeDataHist, pRawDataHist, BIN_COUNT, &entropyA, &entropyB, &jointEntropy, &mutualInformation);
   // std::cout << "Raw entropy = " << entropyA << " | Volume Entropy = " << entropyB << " | Joint Entropy = " << jointEntropy << " | MI = " << mutualInformation << std::endl;
    //std::cout << viewRotation.x << "," << viewRotation.y << "," << viewTranslation.z << "," << mutualInformation << "," << std::endl;
    if(LOG_FLAG)
    {
        if(outputFile->is_open())
        {
            *outputFile << viewRotation.x << "," << viewRotation.y << "," << viewTranslation.z << "," << mutualInformation << "," << std::endl;
        }

        if(viewRotation.y++ > 360.f)
        {
            viewRotation.x++;
            std::cout << viewRotation.x << std::endl;
            viewRotation.y = 0.f;
        }

        if(viewRotation.x > 360.f)
        {
            // sanity check
            if(outputFile->is_open())
                outputFile->close();
            exit(EXIT_SUCCESS);
        }
    }

    if(!serverFailed)
        SendToServer();
    for(GLint i = 0; i < 2; ++i)
    {
        glutSetWindow(windowID[i]);
        glutPostRedisplay();
    }
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 27:
            for(GLint i = 0; i < 2; i++)
            {
                glutSetWindow(windowID[i]);
                #if defined (__APPLE__) || defined(MACOSX)
                    exit(EXIT_SUCCESS);
                #else
                    glutDestroyWindow(glutGetWindow());
                    return;
                #endif
            }
            break;

        case 'f':
            linearFiltering = !linearFiltering;
            setTextureFilterMode(linearFiltering);
            break;

        case '+':
            density += 0.01f;
            break;

        case '-':
            density -= 0.01f;
            break;

        case ']':
            brightness += 0.1f;
            break;

        case '[':
            brightness -= 0.1f;
            break;

        case ';':
            transferOffset += 0.01f;
            break;

        case '\'':
            transferOffset -= 0.01f;
            break;

        case '.':
            transferScale += 0.01f;
            break;

        case ',':
            transferScale -= 0.01f;
            break;

        default:
            break;
    }

    for(GLint i = 0; i < 2; ++i)
    {
        glutSetWindow(windowID[i]);
        glutPostRedisplay();
    }
}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        buttonState  |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        buttonState = 0;
    }

    ox = x;
    oy = y;

    for(GLint i = 0; i < 2; ++i)
    {
        glutSetWindow(windowID[i]); 
        glutPostRedisplay();
    }
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - ox);
    dy = (float)(y - oy);

    if (buttonState == 4)
    {
        // right = zoom
        viewTranslation.z += dy / 100.0f;
    }
    else if (buttonState == 2)
    {
        // middle = translate
        viewTranslation.x += dx / 100.0f;
        viewTranslation.y -= dy / 100.0f;
    }
    else if (buttonState == 1)
    {
        // left = rotate
        viewRotation.x += dy / 5.0f;
        viewRotation.y += dx / 5.0f;
    }

    ox = x;
    oy = y;

    for(GLint i = 0; i < 2; ++i)
    {
        glutSetWindow(windowID[i]); 
        glutPostRedisplay();
    }
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
    if(glutGetWindow() == windowID[0])
    {
        width = w;
        height = h;
        initPixelBuffer();

        // calculate new grid size
        gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

        glViewport(0, 0, w, h);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    freeCudaBuffers();

    if (pbo)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &_tex);
    }

    checkCudaErrors(cudaFree(pVolumeDataHist));
    free(windowID);
    delete pRawDataHist;

    if(LOG_FLAG || outputFile)
        delete outputFile; 
    cudaDeviceReset();
}

void initGL(int *argc, char **argv)
{
    // initialize GLUT callback functions
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    windowID[0] = glutCreateWindow("CUDA volume rendering");
    glutInitWindowPosition(100+width, 100);
    windowID[1] = glutCreateWindow("Information Theoretic Statistics");

    glewInit();

    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions missing.");
        exit(EXIT_SUCCESS);
    }
}

void initPixelBuffer()
{
    if (pbo)
    {
        // unregister this buffer object from CUDA C
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

        // delete old buffer
        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &_tex);
    }

    // create pixel buffer object for display
    glGenBuffersARB(1, &pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &_tex);
    glBindTexture(GL_TEXTURE_2D, _tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void initHistgramBuffers()
{
    // The raw data hist never needs to be sent to a kernel - we can just normally allocate it.
    pRawDataHist = new unsigned int[BIN_COUNT];
    for(int i = 0; i < BIN_COUNT; ++i)
    {
        pRawDataHist[i] = 0; 
    }

    //pRawDataHist = (unsigned int*)malloc(BIN_COUNT*sizeof(unsigned int));
    // We need to allocate this as cuda managed memory - good balance of accessibility and speed
    checkCudaErrors(cudaMallocManaged(&pVolumeDataHist, histSize));
}

void NormaliseAndBin(void* data, size_t dataSize)
{
    // We will use the histogram and ranges stored in local scope
    //      just pass in the data that needs to be binned
    // FIRST - we need to normalize (0,1) the whole dataset based on the ranges
    float normalizedData[dataSize] = {0.0f}; 

    for(int i = 0; i < dataSize; i++)
    {
        // Dodgy as hell casting, not safe but okay since we know its type
        normalizedData[i] = ( ((float)((unsigned char*)data)[i])-DataRange[0] ) / ( DataRange[1]-DataRange[0] );
        // Now we need to bin the value
        float step = 1.f/BIN_COUNT;
        int idx = (int)(normalizedData[i]/step);

        pRawDataHist[idx] += 1;
    }

    for(int i = 0; i < BIN_COUNT; i++)
    {
        std::cout << pRawDataHist[i] << " ";
    }
    std::cout << std::endl;
}

// Load raw data from disk
void *loadRawFile(char *filename, size_t size)
{
    // First we want to allocate the histograms 
    initHistgramBuffers();

    int lowest = std::numeric_limits<int>::max();
    int highest = std::numeric_limits<int>::min();

    FILE *fp = fopen(filename, "rb");

    if (!fp)
    {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

    void *data = malloc(size);
    size_t read = fread(data, 1, size, fp);

    // We need to get the highest and lowest value for normalisation - I figured this was easier than using high-level vectors
    for(int i = 0; i < (size); i++)
    {
        int val = ((int)((unsigned char*)data)[i]);
        (val < lowest) ? DataRange[0] = val, lowest = val : (val > highest) ? DataRange[1] = val, highest = val:  NULL;
    }
    
    NormaliseAndBin(data, size);
    fclose(fp);

#if defined(_MSC_VER_)
    printf("Read '%s', %Iu bytes\n", filename, read);
#else
    printf("Read '%s', %zu bytes\n", filename, read);
#endif

    return data;
}

// General initialization call for CUDA Device
int chooseCudaDevice(int argc, const char **argv, bool bUseOpenGL)
{
    int result = 0;

    if (bUseOpenGL)
    {
        result = findCudaGLDevice(argc, argv);
    }
    else
    {
        result = findCudaDevice(argc, argv);
    }

    return result;
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    // this is just the pointer for the two windows
    windowID = (GLint*)malloc(2*sizeof(GLint));
    pArgc = &argc;
    pArgv = argv;

    char *ref_file = NULL;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    //start logs
    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
        fpsLimit = frameCheckNumber;
    }

    if (ref_file)
    {
        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        chooseCudaDevice(argc, (const char **)argv, false);
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL(&argc, argv);

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        chooseCudaDevice(argc, (const char **)argv, true);
    }

    // parse arguments
    char *filename;

    if (getCmdLineArgumentString(argc, (const char **) argv, "volume", &filename))
    {
        volumeFilename = filename;
    }

    int n;

    if (checkCmdLineFlag(argc, (const char **) argv, "size"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "size");
        volumeSize.width = volumeSize.height = volumeSize.depth = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "xsize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "xsize");
        volumeSize.width = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "ysize"))
    {
        n = getCmdLineArgumentInt(argc, (const char **) argv, "ysize");
        volumeSize.height = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "zsize"))
    {
        n= getCmdLineArgumentInt(argc, (const char **) argv, "zsize");
        volumeSize.depth = n;
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "-l"))
    {
        LOG_FLAG = true;
        outputFile = new std::ofstream();
        outputFile->open("ValidationData.csv", std::ios::out | std::ios::trunc);
    }

    if (checkCmdLineFlag(argc, (const char **) argv, "-h"))
    {
        std::cout << "================= Mutual Information Volume Renderer =================" << std::endl;
        std::cout << "``` Simple CUDA accelerated volume renderer (CUDA sample) with added mutual information metrics ```" << std::endl;
        std::cout << "Flags: " << std::endl;
        std::cout << "  -h     = Help (display this)" << std::endl; 
        std::cout << "  -l     = Logger mode, sample MI around the volume" << std::endl;
        std::cout << "======================================================================" << std::endl;
        exit(EXIT_WAIVED);
    }
    // load volume data
    char *path = sdkFindFilePath(volumeFilename, argv[0]);

    if (path == 0)
    {
        printf("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }

    size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
    void *h_volume = loadRawFile(path, size);

    initCuda(h_volume, volumeSize);
    free(h_volume);

    sdkCreateTimer(&timer);

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    // This is the normal rendering path for VolumeRender
    glutSetWindow(windowID[0]);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    stbi_flip_vertically_on_write(1);

    glutSetWindow(windowID[1]);
    glutReshapeWindow(statsWidth, statsHeight);
    glutInitWindowPosition(width, height);
    glutDisplayFunc(DisplayStatsWindow);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);
    glutSetWindow(windowID[0]); //Change back to the main window once we've handled

    SetupServerConnection();

    for(GLint i = 0; i < 2; i++)
    {
        glutSetWindow(windowID[i]);
        initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        glutMainLoop();
    }
    
}
