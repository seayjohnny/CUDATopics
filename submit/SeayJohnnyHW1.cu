// nvcc SeayJohnnyHW1.cu -o SeayJohnnyHW1; ./'SeayJohnnyHW1'

#include <stdio.h>

int main ( void ) {
	
/*
	The CUDA runtime returns the properties of devices in a	structure of type 
	cudaDeviceProp (pg. 28). In other words, this struct is built into the 
	CUDA runtime and we can reference it directly. Visit 
	https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html 
	for more documentation.
*/
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount( &count ); 

	for(int i=0; i<count; i++) {
		cudaGetDeviceProperties( &prop, i);		// cudaGetDeviceProperties( ptr cudaDeviceProp struct, int device )
		printf("   --- General Information for device %d ---\n", i );
		printf("Name:\t\t\t\t%s\n", prop.name );
		printf("Compute capability:\t\t%d.%d\n", prop.major, prop.minor );
		printf("Integrated:\t\t\t%s\n", prop.integrated ? "Yes":"Not Integrated");
		printf("Total global memory:\t\t%lu B\n", prop.totalGlobalMem );
		printf("Max shared memory per block:\t%lu B\n", prop.sharedMemPerBlock );
		printf("Max threads per block:\t\t%d\n", prop.maxThreadsPerBlock );
		printf("Clock rate:\t\t\t%d kHz\n", prop.clockRate );
		printf("Number of multiprocessors:\t%d\n", prop.multiProcessorCount);
		printf("Shared memory avaible per \n  multiprocessor:\t\t%lu B\n", prop.sharedMemPerMultiprocessor);
		printf("\n");
	}
}
