#include "cfd.h"


__global__ void initialize(double* a, double* oA, double* x, double totalSize, int n, int ghosts){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int j = 0; blockDim.x*j + i < n + 2*ghosts; j++){
		int index = blockDim.x*j + i;
		a[index] = 0;
		oA[index] = 0;
		x[index] = totalSize/n;
	}
}

__device__ void setA(int x, double init, double* a){
	a[x] = init;
}

__device__ double linInterp(double* in){	//dangerous function, need to make sure you're only using it on the in-bounds parts of array
	return ((*(in+1) + *in)/2) - ((*in + *(in-1))/2);
}

__device__ double colellaEvenInterp(double*in){
	return (7.0/12)*(*in + *(in+1)) - (1.0/12)*(*(in+2) + *(in-1));
}


__global__ void advect(double* a, double* oA, double* x, double u, int n, int ghosts, double* dtt){
	__shared__ double dt;
	__shared__ double minDx;
	__shared__ bool* areYouLessThan;

	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(i==0){
		minDx = x[0+ghosts];
		areYouLessThan = new bool[n];
	}
	__syncthreads();
	
	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = j*blockDim.x+i;
		if(x[index+ghosts] < minDx)
			areYouLessThan[index] = true;
		else
			areYouLessThan[index] = false;
	}

	__syncthreads();

	if(i == 0){
		for(int j = 0; j < n; j++){
			if(areYouLessThan[j]){
				if(x[j+ghosts] < minDx)
					minDx = x[j+ghosts];
			}
		}

		dt = (minDx/u)/1000;
		// printf("dt: %f\n", dt);
		delete[] areYouLessThan;
	}

	__syncthreads();



	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = j*blockDim.x+i+ghosts;
		oA[index] = a[index] - dt*u*colellaEvenInterp(&a[index])/x[index];
		// oA[index] = a[index] - dt*u*linInterp(&a[index])/x[index];
		// printf("%d %f %f\n", index, a[index], oA[index]);
		a[index] = oA[index];
	}

	__syncthreads();
	// printf("%d here\n", i);
	if(i==0){	//copy over for boundary conditions
		for(int j = 0; j < ghosts; j++){
			a[j] = a[j+n];
			a[n+ghosts+j] = a[ghosts+j];
			// a[j] = a[ghosts];
			// a[n+ghosts+j] = a[n+ghosts-1];
		}
		// printf("%f\n",dt);
		*dtt = dt;
		// printf("%p %f\n", dtt, *dtt);
	}
	__syncthreads();
}

__global__ void initSinusoid(double* a, double* x, double totalX, int n, int ghosts, double shift, double amp){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = j*blockDim.x+i;
		double temp = 0;
		for(int z = 0; z < index; z++){
			temp += x[z+ghosts];
		}
		a[index+ghosts] = sinpi((temp/totalX)*2)*amp + shift;
	}
	__syncthreads();

	if(i==0){	//copy over for boundary conditions
		for(int j = 0; j < ghosts; j++){
			a[j] = a[j+n];
			a[n+ghosts+j] = a[ghosts+j];
			// a[j] = a[ghosts];
			// a[n+ghosts+j] = a[n+ghosts-1];
		}
		// for(int z = 0; z < n+2*ghosts; z++){
		// 	printf("%5d %10f\n", z, a[z]);
		// }
	}
}

CFD::CFD(int x, double size, double uIn){
	u = uIn;
	ghosts = 2;
	dim = x;
	totalX = size;
	a = new double[dim+2*ghosts];
	cudaMalloc((void**)&d_a, (dim+ghosts*2)*sizeof(double));
	cudaMalloc((void**)&d_x, (dim+ghosts*2)*sizeof(double));
	cudaMalloc((void**)&d_oA, (dim+ghosts*2)*sizeof(double));
	cudaDeviceSynchronize();
	initialize<<<1, dim/numHandle+1>>>(d_a, d_oA, d_x, totalX, dim, ghosts);
	cudaDeviceSynchronize();
	initSinusoid<<<1, dim/numHandle+1>>>(d_a, d_x, totalX, dim, ghosts, 1, 0.5);
	cudaDeviceSynchronize();
}

double* CFD::getA(){
	cudaDeviceSynchronize();
	cudaMemcpy(a, d_a, (dim+2*ghosts)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	return a;
}

int CFD::getDim(){
	return dim;
}

void CFD::setInitial(int x, double init){

}

double CFD::step(){
	double temp, *d_dt;
	cudaMalloc((void**)&d_dt, sizeof(double));
	cudaDeviceSynchronize();
	advect<<<1, dim/numHandle+1>>>(d_a, d_oA, d_x, u, dim, ghosts, d_dt);
	cudaDeviceSynchronize();
	cudaMemcpy(&temp, d_dt, sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(d_dt);
	return temp;
}