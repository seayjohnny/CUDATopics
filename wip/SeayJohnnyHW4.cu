// To compile: nvcc HW4.cu -o temp
#include <sys/time.h>
#include <stdio.h>
#include "../helpers/helper.h"

#define N 100000

__device__ void multiply(float sharedVec[1024], float *A, float *B, long n)
{
	long id = threadIdx.x + blockIdx.x*1024;
	if(id < n)
	{
		sharedVec[threadIdx.x] = A[id]*B[id];
	}
	else
	{
		sharedVec[threadIdx.x] = 0;
	}
}

__device__ void add(float sharedVec[1024], int *offset)
{
	int id = threadIdx.x;
	*offset = *offset/2;
	
	if(id < *offset)
	{
		
		float temp = sharedVec[id] + sharedVec[id + *offset];
		if(id == 0 && blockIdx.x == 1)
		{
			//printf("offset=%d\tsharedVec[%d]+sharedVec[%d+%d]=%f\n", *offset, 
			//	id, id, *offset, temp);
		}
		sharedVec[id] = temp;
	}
}

__device__ void copySharedToGlobal(float sharedVec[1024], float *A)
{
	A[blockIdx.x] = sharedVec[threadIdx.x];

}

__global__ void dot(float *A, float *B, long n)
{
	__device__ __shared__ float sharedVec[1024];
	__device__ __shared__ int offset;

	int id = threadIdx.x + blockIdx.x*1024;
	if(threadIdx.x==0)
	{
		offset = 1024;
	}
	__syncthreads();

	multiply(sharedVec, A, B, n);
	__syncthreads();

	for(int i = 0; i < 10;i++)
	{
		add(sharedVec, &offset);
		__syncthreads();
	}
	
	if(threadIdx.x == 0){
		printf("%d\tsharedVec[0] = %f\n", blockIdx.x, sharedVec[0]);
		copySharedToGlobal(sharedVec, A);
	}
	__syncthreads();
}

int main()
{
	long id;
	float *A_CPU, *B_CPU, *C_CPU; //Pointers for memory on the Host
	long n = N;
	
	// Your variables start here.
	float *A_GPU, *B_GPU, *C_GPU;
	cudaMalloc(&A_GPU,n*sizeof(float));
	cudaMalloc(&B_GPU,n*sizeof(float));
    cudaMalloc(&C_GPU,n*sizeof(float));
	// Your variables stop here.
	
	//Allocating and loading Host (CPU) Memory
	A_CPU = (float*)malloc(n*sizeof(float));
	B_CPU = (float*)malloc(n*sizeof(float));
	C_CPU = (float*)malloc(n*sizeof(float));
	for(id = 0; id < n; id++) {A_CPU[id] = 1.0; B_CPU[id] = 2.0;}
	
	// Your code starts here.
	cudaMemcpy(A_GPU, A_CPU, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, n*sizeof(float), cudaMemcpyHostToDevice);

	int numOfBlocks = (N-1)/1024 + 1;
	printf("%d\n", (N-1)/1024 + 1);
	dot<<<numOfBlocks, 1024>>>(A_GPU, B_GPU, n);
	HANDLE_ERROR(cudaPeekAtLastError());
	HANDLE_ERROR(cudaMemcpy(A_CPU, A_GPU, n*sizeof(float), cudaMemcpyDeviceToHost));

	long s = 0;
	for(int i = 0; i < numOfBlocks; i++)
	{
		s += A_CPU[i];
	}

	printf("Dot Product  = %ld\n", s);
	// Your code stops here.
	
	free(A_CPU); free(B_CPU); free(C_CPU);
    cudaFree(A_GPU); cudaFree(B_GPU); cudaFree(C_GPU);

	return(0);
}
