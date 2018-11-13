#include "cfd.h"
#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void initialize(double* a, double* oA, double* x, double totalSize, int n, int ghosts){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	printf("%d initting\n", i);
	for(int j = 0; blockDim.x*j + i < n + 2*ghosts; j++){
		int index = blockDim.x*j + i;
		a[index] = 0;
		oA[index] = 0;
		x[index] = totalSize/n;
		printf("%d %f %f %f\n", index, a[index], oA[index], x[index]);
	}
}

__device__ void setA(int x, double init, double* a){
	a[x] = init;
}

__device__ double linInterp(double* in){	//dangerous function, need to make sure you're only using it on the in-bounds parts of array
	return (*(in+1) - (*(in+1) - *in)/2) - (*in - (*in - *(in-1))/2);
}


__global__ void advect(double dt, double* a, double* oA, double* x, double u, int n, int ghosts){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = j*blockDim.x+i+ghosts;
		oA[index] = a[index] - dt*u*linInterp(&a[index])/x[index];
		printf("%d %f %f\n", a[index], oA[index]);
	}
}

__global__ void initSinusoid(double* a, double* x, double totalX, int n, int ghosts, double shift, double amp){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = j*blockDim.x+i;
		double temp = 0;
		for(int z = 0; z < index; j++){
			temp += x[z+ghosts];
		}
		a[index+ghosts] = sinpi((temp/totalX)*2)*amp + shift;
	}
	__syncthreads();

	if(i==0){	//copy over for boundary conditions
		for(int j = 0; j < ghosts; j++){
			a[j] = a[j+n];
			a[n+ghosts+j] = a[ghosts+j];
		}
	}
}

CFD::CFD(int x, double size){
	dim = x;
	totalX = size;
	a = new double[dim];
	cout<<"Creating CFD 1D\n";
	cudaMalloc((void**)&d_a, dim*sizeof(double));
	cudaMalloc((void**)&d_x, dim*sizeof(double));
	cudaMalloc((void**)&d_oA, dim*sizeof(double));
	initialize<<<1, dim/20>>>(d_a, d_oA, d_x, totalX, dim, 2);
	initSinusoid<<<1, dim/20>>>(d_a, d_x, totalX, dim, 2, 1, 1);
}

void CFD::setInitial(int x, double init){

}

void CFD::step(double dt){
	advect<<<1, dim/20>>>(dt, d_a, d_oA, d_x, 1.0, dim, 1);
}