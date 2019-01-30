// To compile: nvcc HW4.cu -o temp
#include <sys/time.h>
#include <stdio.h>
#include "../helpers/helper.h"

// 22529
#define N 1000000
#define BLOCKSIZE 1024
#define A_VAL 2.0
#define B_VAL 1.0

__device__ void multiply(float sharedVec[BLOCKSIZE], float *A, float *B, long n)
{
	long id = threadIdx.x + blockIdx.x*BLOCKSIZE;
	if(id < n)
	{
		sharedVec[threadIdx.x] = A[id]*B[id];
	}
	else
	{
		sharedVec[threadIdx.x] = 0;
	}
}

__device__ void add(float sharedVec[BLOCKSIZE], int *offset)
{
	int id = threadIdx.x;
	*offset = *offset/2;
	
	if(id < *offset)
	{
		sharedVec[id] = sharedVec[id] + sharedVec[id + *offset];
	}
}

__device__ void copySharedToGlobal(float sharedVec[BLOCKSIZE], float *A)
{
	A[blockIdx.x] = sharedVec[threadIdx.x];

}

__global__ void dot(float *A, float *B, long n)
{
	__device__ __shared__ float sharedVec[BLOCKSIZE];
	
	int offset = BLOCKSIZE;
	
	multiply(sharedVec, A, B, n);
	__syncthreads();

	for(int i = 0; i < 10;i++)
	{
		add(sharedVec, &offset);
		__syncthreads();
	}
	
	if(threadIdx.x == 0){
		//printf("%d\t\t%f\n", blockIdx.x, sharedVec[0]);
		copySharedToGlobal(sharedVec, A);
	}
	__syncthreads();
}

int main()
{
	long id;
	float *A_CPU, *B_CPU; //Pointers for memory on the Host
	unsigned long n = N;
	
	// Your variables start here.
	float *A_GPU, *B_GPU;
	cudaMalloc(&A_GPU,n*sizeof(float));
	cudaMalloc(&B_GPU,n*sizeof(float));
	// Your variables stop here.
	
	//Allocating and loading Host (CPU) Memory
	A_CPU = (float*)malloc(n*sizeof(float));
	B_CPU = (float*)malloc(n*sizeof(float));
	for(id = 0; id < n; id++) {A_CPU[id] = 1.0; B_CPU[id] = 2.0;}
	
	//Allocating Device (GPU) Memory
	HANDLE_ERROR(cudaMemcpy(A_GPU, A_CPU, n*sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(B_GPU, B_CPU, n*sizeof(float), cudaMemcpyHostToDevice));


/* --------------------------- Dot Product on GPU --------------------------- */
	int numOfBlocks = (N-1)/BLOCKSIZE + 1;
	printf("======================================\n");
	printf("Size of Vectors: \t%ld\n", n);
	printf("Number of Blocks: \t%d\n", numOfBlocks);
	printf("======================================\n");
	//printf("-------------------------\n");
	//printf("\nBlock ID\tsharedVec[0]\t\n");

	double gpuTime;
	startTimer(&gpuTime);
	dot<<<numOfBlocks, BLOCKSIZE>>>(A_GPU, B_GPU, n);
	HANDLE_ERROR(cudaPeekAtLastError());
	HANDLE_ERROR(cudaMemcpy(A_CPU, A_GPU, n*sizeof(float), cudaMemcpyDeviceToHost));
	endTimer(&gpuTime);

	double cpuTime0;
	startTimer(&cpuTime0);
	double s = 0;
	for(int i = 0; i < numOfBlocks; i++)
	{
		s += A_CPU[i];
	}
	endTimer(&cpuTime0);
	printf("   vvv      GPU Run Results     vvv   \n");
	printf("--------------------------------------\n");
	printf("\nDot Product  = \t\t%.4f\n", s);
	printf("Expected Result = \t%.4f\n", N*A_VAL*B_VAL);
	printf("--------------------------------------\n");
	printf("GPU Time = \t\t%.1f ms\n", gpuTime);
	printf("CPU Reduction Time = \t%.1f ms\n", cpuTime0);
	printf("Total Time = \t\t%.1f ms\n", gpuTime + cpuTime0);
	printf("\n======================================\n");
/* --------------------------- Dot Product on CPU --------------------------- */
	for(id = 0; id < n; id++) {A_CPU[id] = 1.0; B_CPU[id] = 2.0;}

	double cpuTime1;
	startTimer(&cpuTime1);

	//Multiply
	for(id = 0; id < n; id++) {A_CPU[id] = A_CPU[id]*B_CPU[id];}
	//Add
	s = 0;
	for(id = 0; id < n; id++) {s += A_CPU[id];}
	endTimer(&cpuTime1);
	printf("   vvv      CPU Run Results     vvv   \n");
	printf("--------------------------------------\n");
	printf("\nDot Product  = \t\t%.1f\n", s);
	printf("Expected Result = \t%.1f\n", N*A_VAL*B_VAL);
	printf("--------------------------------------\n");
	printf("CPU Time = \t\t%.1f ms\n", cpuTime1);
	printf("\n======================================\n");
	printf("CPU Time - GPU Total Time = %.1f ms\n", cpuTime1 - (gpuTime + cpuTime0));
	// Your code stops here.
	
	free(A_CPU); free(B_CPU);
    cudaFree(A_GPU); cudaFree(B_GPU);

	return(0);
}
