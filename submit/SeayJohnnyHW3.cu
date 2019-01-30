//nvcc SeayJohnnyHW3.cu -o temp -lglut -lGL -lm

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "../helpers/helper.h"

// Dimensions and scale
#define DIM     768    // window size = DIMxDIM

/* ========================================================================== */
/*                         SECTION: Globals                                   */
/* ========================================================================== */
double gpuTimer;
double cpuTimer;

int displayCPUTrigger = 0;
float *pixels;

unsigned int seedChoice;
__constant__ float2 d_Seed;
float2 h_Default;
float2 h_Nicholas;
float2 h_Broccoli;
float2 h_Wyatt;
float2 h_Custom;

int drag = 0;
int mouse_i = 0;
int mouse_step = 10;
int oldX = 0;
int oldY = 0;
float shiftX = 0.0;
float shiftY = 0.0;
double scale = 0.7;
int tick = 0;

/* ========================================================================== */
/*                    SECTION: Device Functions and Kernel                    */
/* ========================================================================== */
__device__ float4 dGetColor(double x, double y, int seedChoice, double scale, int tick) 
{
    float4 color;
	double mag,maxMag,t1;
    double maxCount = 255 + 2.0*tick*logf(scale);
	double count = 0;
	maxMag = 10.0;
	mag = 0.0;
        
	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;	
		x = x*x - y*y + d_Seed.x;
		y = (2.0 * t1 * y) + d_Seed.y;
		mag = sqrtf(x*x + y*y);
		count++;
    }

    float v = count/maxCount;
    if(seedChoice==3){v = v;}

    color = dHSVToRGB(make_float3(100.0/360.0, 1.0, v));
    return(color);
	
}

__global__ void kernel(float *px, int seedChoice, 
                       float shiftX, float shiftY, 
                       float scale, int tick)
{

    double x = threadIdx.x + blockIdx.x*blockDim.x;
    double y = threadIdx.y + blockIdx.y*blockDim.y;
    int id = x + y*gridDim.x*blockDim.x;
    int idx = id*3;

    x = (((2.0*x/DIM) - 1 )/scale) - shiftX/DIM;
    y = (((2.0*y/DIM) - 1 )/scale) + shiftY/DIM;

    float4 color = dGetColor(x, y, seedChoice, scale, tick);
    
    px[idx] = color.x;
    px[idx + 1] = color.y;
    px[idx + 2] = color.z;

}


/* ========================================================================== */
/*                           SECTION: Host Functions                          */
/* ========================================================================== */
__host__ float2 hGetSeed(int seedChoice)
{
    switch(seedChoice)
    {
        case 0:
            return(h_Default);
        case 1:
            return(h_Nicholas);
        case 2:
            return(h_Broccoli);
        case 3:
            return(h_Wyatt);
        case 4:
            return(h_Custom);
        default:
            return(h_Default);
    }
}


__host__ float4 hGetColor(float x, float y, int seedChoice) 
{
    
	float mag,maxMag,t1;
	float maxCount = 10000;
	float count = 0;
	maxMag = 10;
	mag = 0.0;
    
    float2 seed  = hGetSeed(seedChoice);
    
	while (mag < maxMag && count < maxCount) 
	{
		t1 = x;	
		x = x*x - y*y + seed.x;
		y = (2.0 * t1 * y) + seed.y;
		mag = sqrt(x*x + y*y);
		count++;
    }

    float v = 20*count/maxCount;
    if(seedChoice==3){v = v/20;}

    float4 color = hHSVToRGB(make_float3(200.0/360.0, 1.0, v));
	return(color);
}

/* ========================================================================== */
/*                          SECTION: OpenGL Functions                         */
/* ========================================================================== */

void displayGPU(void) 
{ 
    // Clear window to background color
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush(); 

    // Start timer
    startTimer(&gpuTimer);

    float *px_CPU;
    float *px_GPU; 

    dim3 dimBlock;
    dim3 dimGrid;

    // Setup grid layout
    //  - Using 2-D blocks and grid for easier coordinate transformations.
    //  - Using multiples of 32 for more efficient warps. 
    dimBlock.x = 32;
    dimBlock.y = 32;
    dimBlock.z = 1;

    dimGrid.x = DIM/32;
    dimGrid.y = DIM/32;
    dimGrid.z = 1;

    // Allocate memory for pixel data on host and device
	px_CPU = (float*)malloc(DIM*DIM*3*sizeof(float));
    cudaMalloc(&px_GPU, DIM*DIM*3*sizeof(float));
    
    // Run the kernel
    kernel<<<dimGrid, dimBlock>>>(px_GPU, seedChoice, shiftX, shiftY, scale, tick);

    // Copy pixel data from device to host
    //  - HANDLE_ERROR (...) is a very convenient function used to catch
    //    most errors that happen on the device. Can be found in the
    //    helper.h file.
    HANDLE_ERROR( cudaPeekAtLastError() );
    HANDLE_ERROR( cudaMemcpy(px_CPU, px_GPU, DIM*DIM*3*sizeof(float), cudaMemcpyDeviceToHost) );

    // Draw pixels and free pixel data from host and device memory
	glDrawPixels(DIM, DIM, GL_RGB, GL_FLOAT, px_CPU);
    glFlush();

    // End timer
    endTimer(&gpuTimer);

    char *title = (char*)malloc(100*sizeof(char));
    sprintf(title, "GPU | Time to render:\t %.5f s\n", gpuTimer/1000000);
    glutSetWindowTitle(title);

    free(px_CPU);
    cudaFree(px_GPU); 
}

void displayCPU(void) 
{ 
    // Clear window to background color
    glClear(GL_COLOR_BUFFER_BIT);
    glFlush();

    if (displayCPUTrigger == 0)
    {
        printf("displayCPUTrigger = %d\n", displayCPUTrigger);
        displayCPUTrigger = 1;
         // Start timer
         startTimer(&cpuTimer);

         // Set zoom level
         float xMin = -1/scale;
         float xMax =  1/scale;
         float yMin = -1/scale;
         float yMax =  1/scale;
 
         // Transformation of pixel coordinates to fractal-space coordinates
         float stepSizeX = (xMax - xMin)/((float)DIM);
         float stepSizeY = (yMax - yMin)/((float)DIM);
 
         
         float x, y;
         float4 px_color;
         int k;
 
         // Allocate memory for pixel data
         pixels = (float *)malloc(DIM*DIM*3*sizeof(float));
 
         // Iterate through and set the pixel data
         k = 0;
         y = yMin;
         while(y < yMax) 
         {
             x = xMin;
             while(x < xMax) 
             {
                 px_color = hGetColor(x, y, seedChoice);
                 pixels[k] = px_color.x;	//Red on or off returned from color
                 pixels[k+1] = px_color.y; 	//Green off
                 pixels[k+2] = px_color.z;	//Blue off
                 k=k+3;			//Skip to next pixel
                 x += stepSizeX;
             }
             y += stepSizeY;
         }
 
         // Draw pixels and free pixel data from memory
         glDrawPixels(DIM, DIM, GL_RGB, GL_FLOAT, pixels); 
         glFlush();
 
 
         // End timer
         endTimer(&cpuTimer);
 
         char *title = (char*)malloc(100*sizeof(char));
         sprintf(title, "CPU | Time to render:\t %.5f s\n", cpuTimer/1000000);
         glutSetWindowTitle(title);
         
    }
    else
    {
        // Draw pixels and free pixel data from memory
        glDrawPixels(DIM, DIM, GL_RGB, GL_FLOAT, pixels); 
        glFlush();
        free(pixels);
    }

}

void keypress(unsigned char key, int x, int y)
{
    switch(key)
    {
        case 27:
            exit(0);

    }
}

void startDrag(int mx, int my)
{
    drag = 1;
}

void mouse_move(int mx, int my)
{
    mouse_i++;
    if(mouse_i > mouse_step)
    {
        mouse_i = 0;
    }

    if(drag && mouse_i/mouse_step)
    {
        shiftX +=  (mx - oldX)/scale;
        shiftY += (my - oldY)/scale;
        glutPostRedisplay();
    }

}

void mouse(int button, int state, int mx, int my)
{

    switch(button)
    {
        case GLUT_LEFT_BUTTON:
            if(state==GLUT_DOWN)
            {
                    drag = 1;
                    oldX = mx;
                    oldY = my;
            }

            if(state==GLUT_UP)
            {
                drag = 0;
                oldX = mx;
                oldY = my;
            }
            break;
        case 3:
            scale += scale*0.5;
            tick += 1;
            glutPostRedisplay();
            break;
        case 4:
            scale -= scale*0.5;
            tick -= 1;
            glutPostRedisplay();
            break;
        
    }
}


/* ========================================================================== */
/*                                SECTION: Main                               */
/* ========================================================================== */

int main(int argc, char** argv)
{ 

    // Store predefined seeds
    h_Default = make_float2(-0.7531930315709545, 0.05331999448114999);
    h_Nicholas = make_float2(-0.3740480961923849, -0.6066666719669807);
    h_Broccoli = make_float2(-0.3948897795591184, -0.5863460624863006);
    h_Wyatt = make_float2(-0.824, -0.1711);
    h_Custom = make_float2(0.0, 0.0);

    // Prompt for seed choice
    printf("\n Enter seed choice:\n");
    printf("\t0: Default\n");
    printf("\t1: Nicholas\n");
    printf("\t2: Broccoli\n");
    printf("\t3: Wyatt\n");
    printf("\t4: Enter Custom Seed\n\n\t> ");
    scanf("%d", &seedChoice);

    // Load seed data onto device
    switch(seedChoice)
    {
        case 0:
            cudaMemcpyToSymbol(d_Seed, &h_Default, sizeof(float2), 0, cudaMemcpyHostToDevice);
            break;
        case 1:
            cudaMemcpyToSymbol(d_Seed, &h_Nicholas, sizeof(float2), 0, cudaMemcpyHostToDevice);
            break;
        case 2:
            cudaMemcpyToSymbol(d_Seed, &h_Broccoli, sizeof(float2), 0, cudaMemcpyHostToDevice);
            break;
        case 3:
            cudaMemcpyToSymbol(d_Seed, &h_Wyatt, sizeof(float2), 0, cudaMemcpyHostToDevice);
            break;
        case 4:
            scanf("\n Enter real: %f", &h_Custom.x);
            scanf("\n Enter imaginary: %f", &h_Custom.y);
            cudaMemcpyToSymbol(d_Seed, &h_Custom, sizeof(float2), 0, cudaMemcpyHostToDevice);
            break;
        default:
            cudaMemcpyToSymbol(d_Seed, &h_Default, sizeof(float2), 0, cudaMemcpyHostToDevice);
    }

    // TODO : printf("Loaded seed information on GPU in ")

    // Initialize OpenGL
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowSize(DIM, DIM);
    glutInitWindowPosition((50+1822+1680-DIM)/2,
                       (500+1050+1050-DIM)/2);

    // Create first window
   	glutCreateWindow("GPU | Time to render:\t---");
    glutDisplayFunc(displayGPU);
    glutKeyboardFunc(keypress);
    glutMouseFunc(mouse);
    glutMotionFunc(mouse_move);

    // Store position of the GPU window in order to 
    // initialize the CPU window next to the GPU window
    int posX, posY;
    posX = glutGet(GLUT_WINDOW_X);
    posY = glutGet(GLUT_WINDOW_Y);

    // Create second window
    glutInitWindowPosition(posX+DIM,posY);
    glutCreateWindow("CPU | Time to render:\t---");
    glutDisplayFunc(displayCPU);
    glutKeyboardFunc(keypress);

    glClearColor(0.1, 0.1, 0.1, 0.1);
    glutMainLoop();

    return(0);

}
