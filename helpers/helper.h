#include <sys/time.h>
#include <stdio.h>


static void HandleError( cudaError_t err, const char *file, int line )
{
    if(err != cudaSuccess)
	{
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__device__ float4 dHSVToRGB(float3 hsv)
{
	
	float h,s,v;
	h = hsv.x;
	s = hsv.y;
	v = hsv.z;	

    if(s == 0)
	{
		float clr = v;
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
		if		(i==0){return(make_float4(v, t, p, 1.0));}
		else if	(i==1){return(make_float4(q, v, p, 1.0));}
		else if	(i==2){return(make_float4(p, v, t, 1.0));}
		else if	(i==3){return(make_float4(p, q, v, 1.0));}
		else if	(i==4){return(make_float4(t, p, v, 1.0));}
		else	{return(make_float4(v, p, q, 1.0));}
	}
}

float4 hHSVToRGB(float3 hsv)
{
	
	float h,s,v;
	h = hsv.x;
	s = hsv.y;
	v = hsv.z;	

    if(s == 0)
	{
		float clr = v;
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
		if		(i==0){return(make_float4(v, t, p, 1.0));}
		else if	(i==1){return(make_float4(q, v, p, 1.0));}
		else if	(i==2){return(make_float4(p, v, t, 1.0));}
		else if	(i==3){return(make_float4(p, q, v, 1.0));}
		else if	(i==4){return(make_float4(t, p, v, 1.0));}
		else	{return(make_float4(v, p, q, 1.0));}
	}
}


void startTimer(double *timer)
{
	timeval temp;
	gettimeofday(&temp, NULL);
	*timer = (double)(temp.tv_sec * 1000000 + temp.tv_usec);
}

double getTimer(double *timer)
{
	timeval temp;
	gettimeofday(&temp, NULL);
	return((double)(temp.tv_sec * 1000000 + temp.tv_usec) - *timer);
}

void endTimer(double *timer)
{
	timeval temp;
	gettimeofday(&temp, NULL);
	*timer = (double)(temp.tv_sec * 1000000 + temp.tv_usec) - *timer;
}
