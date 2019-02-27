#include "cfd.h"


__global__ void initialize(double* a, double* oA, double* x, double totalSize, int n, int ghosts){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = blockDim.x*j + i;
		a[index] = 0;
		oA[index] = 0;
		x[index] = totalSize/n;
	}
	for(int j = n; j < n+2*ghosts; j++){
		oA[j] = 0;
	}
}

__device__ void setA(int x, double init, double* a){
	a[x] = init;
}

__device__ double linInterp(double* in){	//dangerous function, need to make sure you're only using it on the in-bounds parts of array
	return ((*(in+1) + *in)/2) - ((*in + *(in-1))/2);
}

__device__ double colellaEvenInterp(double* in){
	double aRj = 7.0f/12*(*in + *(in+1)) - 1.0f/12*(*(in+2) + *(in-1));
	if((aRj > *in && aRj < *(in+1)) || (aRj < *in && aRj > *(in+1)))
		return aRj;
	else return *in;
}

__device__ double colellaEvenInterp(double ai, double air1, double ail1, double air2, double ail2){
	return (7.0/12)*(air1-ail1) - (1.0/12)*(air2 + ail1) - (air1 + ail2);
}


__global__ void advect(double* a, double* oA, double* x, double* prev, double u, int n, int ghosts, double* minDx, double* dt, double* timeElapsed, int* counter, double* error, double tmax){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int l_n = n;
	double l_dt, l_tElapsed = 0, l_dx = x[0], l_u = u, l_tmax = tmax;
	int l_ghosts = ghosts;
	int step = 0;
	const double courantFactor = 0.01;
	l_dt = (l_dx/l_u) * courantFactor;
	__syncthreads();
	const int maxStep = 99999;


	while(l_tElapsed < l_tmax && step < maxStep){
		__syncthreads();
		if(i == 0)
			if(step % 1000 == 0 || l_tElapsed + l_dt > l_tmax)
				for(int j = 0; j < l_n; j++){
					printf("%d %d %f %f\n", j, step, l_tElapsed, a[j]);
				}
		__syncthreads();
		for(int j = threadIdx.x; j < l_n; j+=blockDim.x){
			prev[j+l_ghosts] = a[j];
		}
		__syncthreads();
		if(i == 0){
			for(int j = 0; j < l_ghosts; j++){
				prev[j] = prev[l_ghosts];
				prev[j + l_n + l_ghosts] = prev[l_n + l_ghosts - 1];
			}
		}
		__syncthreads();
		// if(i == 0){
		// 	for(int j = 0; j < l_n + 2*l_ghosts; j++){
		// 		printf("prev %d %d %f\n", j, step, prev[j]);
		// 	}
		// 	// for(int j = 0; j < l_n; j++){
		// 	// 	printf("a %d %d %f\n", j, step, a[j]);
		// 	// }
		// }
		__syncthreads();
		for(int j = threadIdx.x; j < l_n; j+=blockDim.x){
			// int index = j*blockDim.x+i+ghosts;

			// copy to local register for quick processing... this copy actually is less efficient than global memory refs if vars are only used once
			// double ai = a[index], xi = x[index], air1 = a[index+1], air2 = a[index+2], ail1 = a[index-1], ail2 = a[index-2];
			// a[index] = ai - l_dt*u*colellaEvenInterp(ai, air1, ail1, air2, ail2)/xi;
		
			oA[j+l_ghosts] = colellaEvenInterp(prev+j+l_ghosts);
		}

		__syncthreads();

		if(i==0){	//copy over for boundary conditions
			for(int j = 0; j < l_ghosts; j++){
				oA[j] = oA[l_ghosts];
				oA[l_n+l_ghosts+j] = oA[l_n+l_ghosts - 1];
			}
		}		
		__syncthreads();
		
		for(int j = threadIdx.x; j < l_n; j+=blockDim.x){
			a[j] = a[j] + l_u * l_dt * (oA[j-1 + l_ghosts] - oA[j + l_ghosts]) / l_dx;
		}
		__syncthreads();
		l_tElapsed += l_dt;
		step++;
	}
}

__global__ void initSinusoid(double* a, double* x, double totalX, int n, int ghosts, double shift, double amp){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = j*blockDim.x+i;
		double temp = 0;
		for(int z = 0; z < index; z++){
			temp += x[z];
		}
		a[index] = sinpi((temp/totalX)*2)*amp + shift;
	}
	__syncthreads();
}

__global__ void initSquare(double* a, double* x, double totalX, int n, int ghosts){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = j*blockDim.x+i;
		if(index > n/3 && index < 2*n/3)
			a[index] = 1.5;
		else a[index] = .5;
	}
	__syncthreads();

}


CFD::CFD(int x, double size, double uIn){
	u = uIn;
	ghosts = 3;
	dim = x;
	totalX = size;
	a = new double[dim+2*ghosts];
	numBlocks = 1;
	cudaMalloc((void**)&d_a, (dim)*sizeof(double));
	cudaMalloc((void**)&d_x, (dim)*sizeof(double));
	cudaMalloc((void**)&d_oA, (dim+ghosts*2)*sizeof(double));
	cudaDeviceSynchronize();
	initialize<<<numBlocks, 1024>>>(d_a, d_oA, d_x, totalX, dim, ghosts);
	cudaDeviceSynchronize();
	// initSinusoid<<<numBlocks, 1024>>>(d_a, d_x, totalX, dim, ghosts, 1, 0.5);
	initSquare<<<numBlocks, 1024>>>(d_a, d_x, totalX, dim, ghosts);
	cudaDeviceSynchronize();
}

double* CFD::getA(){
	cudaDeviceSynchronize();
	cudaMemcpy(a, d_a, (dim)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	return a;
}

int CFD::getDim(){
	return dim;
}

void CFD::setInitial(int x, double init){

}

void CFD::step(double maxtime){
	double *dt,*te, *minDx, *error, *prevA;
	int* counter;
	cudaMalloc((void**)&dt, sizeof(double));
	cudaMalloc((void**)&te, sizeof(double));
	cudaMalloc((void**)&counter, sizeof(int));
	cudaMalloc((void**)&minDx, sizeof(double));
	cudaMalloc((void**)&error, sizeof(double));
	cudaMalloc((void**)&prevA, sizeof(double)*dim+2*ghosts);
	cudaDeviceSynchronize();
	advect<<<1, 512>>>(d_a, d_oA, d_x, prevA, u, dim, ghosts, minDx, dt, te, counter, error, maxtime);
	cudaDeviceSynchronize();
}
