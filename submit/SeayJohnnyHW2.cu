// nvcc SeayJohnnyHW2.cu -o SeayJohnnyHW2; ./'SeayJohnnyHW2'

#include <sys/time.h>
#include <stdio.h>

#define N 67043328

__global__ void addition(float *A, float *B, float *C, int n, int maxThreads) {

    int id = threadIdx.x + blockIdx.x*maxThreads;

    if(id < n) {
        C[id] = A[id] + B[id];
    }
}

int main ( void ) {
    
    /*  
    Getting the device properties in order to create the 
    minimum requried amount of blocks regardless of device.
    */
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;

    double sum, gpuTime, totalTime;
    float *A_CPU, *B_CPU, *C_CPU; //Pointers for memory on the host
    float *A_GPU, *B_GPU, *C_GPU; //Pointers for memory on the device
    timeval start, end;

    dim3 dimBlock; 
    dim3 dimGrid;
    
    //Threads in a block
    dimBlock.x = maxThreads;
    dimBlock.y = 1;
    dimBlock.z = 1;

    //Blocks in a grid
    dimGrid.x = ( (N-1)/maxThreads ) + 1;;
    dimGrid.y = 1;
    dimGrid.z = 1;

    printf("\n Length of vector:\t\t\t%d\n", N);
    printf(" Max number of threads per block:\t%d\n", maxThreads);
    printf(" Number of blocks created:\t\t%d\n\n", dimGrid.x);

    //Allocate Host (CPU) Memory
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaMalloc(&B_GPU,N*sizeof(float));
    cudaMalloc(&C_GPU,N*sizeof(float));
    
    //Loads values into vectors that we will add.
	for(long id = 0; id < N; id++)
	{		
		A_CPU[id] = 1;	
		B_CPU[id] = 0;
	}

    //Move A and B vectors from CPU to GPU
    gettimeofday(&start, NULL);
    cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
    gettimeofday(&end, NULL);
    gpuTime = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    printf(" Time to copy A and B to GPU \t=  %.10f ms\n", (gpuTime/1000.0));
    totalTime = gpuTime;
    
    //Add the two vectors together on the GPU
    gettimeofday(&start, NULL);
    addition<<<dimGrid, dimBlock>>>(A_GPU, B_GPU, C_GPU, N, maxThreads);
    gettimeofday(&end, NULL);
    gpuTime = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf(" Time to add A and B on GPU \t=  %.10f ms\n", (gpuTime/1000.0));
    totalTime += gpuTime;

    //Move the results from the GPU to the CPU
    gettimeofday(&start, NULL);
    cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);
    gpuTime = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	printf(" Time to get results from GPU \t=  %.10f ms\n", (gpuTime/1000.0));
    totalTime += gpuTime;

    for(int i=0;i<50;i++){printf("-");}
	printf("\n Total time spent \n  interacting with GPU \t\t=  %.10f ms\n", (totalTime/1000.0));        

    sum = 0.0;
	for(long id = 0; id < N; id++)
	{ 
		sum += C_CPU[id];
    }

    printf("\n Sum of C_CPU from GPU addition = %.10f\n", sum);

    free(A_CPU); free(B_CPU); free(C_CPU);
    cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);
    
    return(0);
}
