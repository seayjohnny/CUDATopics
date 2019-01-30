//nvcc OpenGLInterop.cu -o OpenGLInterop -lglut -lGL -lm; ./'OpenGLInterop'
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <cuda_gl_interop.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


static void HandleError( cudaError_t err, const char *file, int line )
{
    if(err != cudaSuccess)
	{
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
#define GET_PROC_ADDRESS( str ) glXGetProcAddress( (const GLubyte *)str )
#define DIM    1024
#define A  -0.3948897795591184 //real
#define B  -0.5863460624863006   //imaginary

// default: -0.8 0.156
// broccoli: -0.3948897795591184  -0.5863460624863006
// nicholas: -0.3740480961923849  - 0.6066666719669807

PFNGLBINDBUFFERARBPROC    glBindBuffer     = NULL;
PFNGLDELETEBUFFERSARBPROC glDeleteBuffers  = NULL;
PFNGLGENBUFFERSARBPROC    glGenBuffers     = NULL;
PFNGLBUFFERDATAARBPROC    glBufferData     = NULL;


struct GPUAnimBitmap {

    GLuint  bufferObj;
    cudaGraphicsResource *resource;
    int     width, height;
    void    *dataBlock;
    void (*fAnim)(uchar4*,void*,int);
    void (*animExit)(void*);
    void (*clickDrag)(void*,int,int,int,int);
    int     dragStartX, dragStartY;

    GPUAnimBitmap( int w, int h, void *d = NULL ) {
        width = w;
        height = h;
        dataBlock = d;
        clickDrag = NULL;

	/*
        // first, find a CUDA device and set it to graphic interop
        cudaDeviceProp  prop;
        memset( &prop, 0, sizeof( cudaDeviceProp ) );
		int dev;
        prop.major = 1;
        prop.minor = 0;
        HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
	*/

        cudaGLSetGLDevice( 0 );


		// a bug in the Windows GLUT implementation prevents us from
		// passing zero arguments to glutInit()
		int c=1;
		char* dummy = "";
		glutInit( &c, &dummy );
	
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
        glutInitWindowSize( width, height );
        glutCreateWindow( "bitmap" );

        glBindBuffer    = (PFNGLBINDBUFFERARBPROC)GET_PROC_ADDRESS("glBindBuffer");
        glDeleteBuffers = (PFNGLDELETEBUFFERSARBPROC)GET_PROC_ADDRESS("glDeleteBuffers");
        glGenBuffers    = (PFNGLGENBUFFERSARBPROC)GET_PROC_ADDRESS("glGenBuffers");
        glBufferData    = (PFNGLBUFFERDATAARBPROC)GET_PROC_ADDRESS("glBufferData");

        glGenBuffers( 1, &bufferObj );
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
        glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4,
                      NULL, GL_DYNAMIC_DRAW_ARB );

        HANDLE_ERROR( cudaGraphicsGLRegisterBuffer( &resource, bufferObj, cudaGraphicsMapFlagsNone ) );
    }

    ~GPUAnimBitmap() {
        free_resources();
    }

    void free_resources( void ) {
        HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
        glDeleteBuffers( 1, &bufferObj );
    }


    long image_size( void ) const { return width * height * 4; }

    void click_drag( void (*f)(void*,int,int,int,int)) {
        clickDrag = f;
    }

    void anim_and_exit( void (*f)(uchar4*,void*,int), void(*e)(void*) ) {
        GPUAnimBitmap**   bitmap = get_bitmap_ptr();
        *bitmap = this;
        fAnim = f;
        animExit = e;

        glutKeyboardFunc( Key );
        glutDisplayFunc( Draw );
        if (clickDrag != NULL)
            glutMouseFunc( mouse_func );
        glutIdleFunc( idle_func );
        glutMainLoop();
    }

    // static method used for glut callbacks
    static GPUAnimBitmap** get_bitmap_ptr( void ) {
        static GPUAnimBitmap*   gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func( int button, int state,
                            int mx, int my ) {
        if (button == GLUT_LEFT_BUTTON) {
            GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            } else if (state == GLUT_UP) {
                bitmap->clickDrag( bitmap->dataBlock,
                                   bitmap->dragStartX,
                                   bitmap->dragStartY,
                                   mx, my );
            }
        }
    }

    // static method used for glut callbacks
    static void idle_func( void ) {
        static int ticks = 1;
        GPUAnimBitmap*  bitmap = *(get_bitmap_ptr());
        uchar4*         devPtr;
        size_t  size;

        HANDLE_ERROR( cudaGraphicsMapResources( 1, &(bitmap->resource), NULL ) );
        HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, bitmap->resource) );

        bitmap->fAnim( devPtr, bitmap->dataBlock, ticks++ );

        HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &(bitmap->resource), NULL ) );

        glutPostRedisplay();
    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
            case 27:
                GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
                if (bitmap->animExit)
                    bitmap->animExit( bitmap->dataBlock );
                bitmap->free_resources();
                exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw( void ) {
        GPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->width, bitmap->height, GL_RGBA,
                      GL_UNSIGNED_BYTE, 0 );
        glutSwapBuffers();
    }
};

__device__ float4 hsv2rgb(float3 hsv)
{
	
	float h,s,v;
	h = hsv.x;
	s = hsv.y;
	v = hsv.z;	

    if(s == 0)
	{
		float clr = v*255;
		return make_float4(clr, clr, clr, 255);
	}
	else
	{
		h = h*6.0;
		int i = (int)h;
		float f = h - i;
		float p = v*(1-s);
		float q = v*(1-s*f);
		float t = v*(1-s*(1-f));
		i = i%6;
		if		(i==0){return(make_float4(v*255, t*255, p*255, 255));}
		else if	(i==1){return(make_float4(q*255, v*255, p*255, 255));}
		else if	(i==2){return(make_float4(p*255, v*255, t*255, 255));}
		else if	(i==3){return(make_float4(p*255, q*255, v*255, 255));}
		else if	(i==4){return(make_float4(t*255, p*255, v*255, 255));}
		else	{return(make_float4(v*255, p*255, q*255, 255));}
	}
}

__device__ uchar4 color(int x, int y, int ticks) {

    const double scale = powf(1.01, -ticks);
    const double shiftX = -0.001;
    const double shiftY = 0.0;
    double fx = (scale*2.0*(double)(x-DIM/2)/DIM) + shiftX;//scale;
    double fy = (scale*2.0*(double)(y-DIM/2)/DIM) + shiftY;//scale;
    
	double mag, maxMag, t1;
	double maxCount = 360.0*(1 + ticks/360.0);
	double count = 0;

	maxMag = 10.0;
	mag = 0.0;
	

	if ( x==DIM-1 && y==DIM/2 && (ticks%100 == 0) )
		printf("fx: %f\n", fx);


	while (mag < maxMag && count < maxCount)
	{
		t1 = fx;	
		fx = fx*fx - fy*fy + A;
		fy = (2.0 * t1 * fy) + B;
		mag = sqrtf(fx*fx + fy*fy);
		count++;
	}

	float4 rgba_f = hsv2rgb(make_float3(count/maxCount, 1.0, 1.0));
	uchar4 rgba = make_uchar4((int)rgba_f.x,(int)rgba_f.y,(int)rgba_f.z,(int)rgba_f.w); 
    //int clr = (int)((count/maxCount) * 255);
    
    return( rgba );
}

__global__ void kernel( uchar4 *ptr, int ticks) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

	//if(offset==3000){printf("%f\n", 1.01*ticks);}
    // now calculate the value at that position	
    //int clr = color(x, y, ticks);
	uchar4 clr = color(x, y, ticks);

    // accessing uchar4 vs unsigned char*
	ptr[offset] = clr;
/*
    ptr[offset].x = 0;
    ptr[offset].y = clr;
    ptr[offset].z = 0;
    ptr[offset].w = 255;
*/
}



/*
__global__ void kernel( uchar4 *ptr, int ticks )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float fx = x - DIM/2;
	float fy = y - DIM/2;
	float d = sqrtf(x*x + y*y);
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f -ticks/7.0f)/(d/10.0f + 1.0f));
	
	ptr[offset].x = grey;
	ptr[offset].y = grey;
	ptr[offset].z = grey;
	ptr[offset] .w = 255;
}
*/

void generate_frame( uchar4 *pixels, void*, int ticks )
{
	
	dim3 grids(DIM/16,DIM/16);
	dim3 threads(16,16);
	kernel<<<grids,threads>>>(pixels, ticks);
}


int main( void )
{

	GPUAnimBitmap bitmap( DIM, DIM, NULL );

	bitmap.anim_and_exit( (void (*)(uchar4*,void*,int))generate_frame, NULL );
	

}
