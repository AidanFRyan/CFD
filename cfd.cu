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
	return (7.0/12)*(*(in+1) - *(in-1)) - (1.0/12)*((*(in+2) + *(in-1))-(*(in+1) + *(in-2)));
}


__global__ void advect(double* a, double* oA, double* x, double u, int n, int ghosts, double* minDx, double* dt, double* timeElapsed, int* counter, double tmax){
	// __shared__ bool* areYouLessThan;
	__shared__ int maxN;
	grid_group g = this_grid();

	int i = threadIdx.x + blockDim.x*blockIdx.x;

	if(i == 0){
		*timeElapsed = 0;
		*minDx = x[0];
		*dt = (*minDx/u)/500;
		*counter = 0;
	}
	if(threadIdx.x == 0){
		if(i + blockDim.x > n)
			maxN = n-i;
		else maxN = blockDim.x;
	}
	// __syncthreads();
	// g.sync();

	while(*timeElapsed < tmax){
		// if(i==0){
		// 	minDx = x[0+ghosts];
		// 	areYouLessThan = new bool[n];
		// }
		// __syncthreads();
		
		// for(int j = 0; blockDim.x*j + i < n; j++){
		// 	int index = j*blockDim.x+i;
		// 	if(x[index+ghosts] < minDx)
		// 		areYouLessThan[index] = true;
		// 	else
		// 		areYouLessThan[index] = false;
		// }

		// __syncthreads();

		// if(i == 0){
		// 	for(int j = 0; j < n; j++){
		// 		if(areYouLessThan[j]){
		// 			if(x[j+ghosts] < minDx)
		// 				minDx = x[j+ghosts];
		// 		}
		// 	}

			// dt = (minDx/u)/1000;
		// 	// printf("dt: %f\n", dt);
		// 	delete[] areYouLessThan;
		// }

		// __syncthreads();
		g.sync();

		for(int j = 0; blockDim.x*j + threadIdx.x < maxN; j++){
			int index = j*blockDim.x+i+ghosts;
			oA[index] = a[index] - (*dt)*u*colellaEvenInterp(&a[index])/x[index];
			// oA[index] = a[index] - dt*u*linInterp(&a[index])/x[index];
			// printf("%d %f %f\n", index, a[index], oA[index]);
			a[index] = oA[index];
		}

		// __syncthreads();
		g.sync();
		// printf("%d here\n", i);
		if(*counter == 0){
			for(int j = 0; blockDim.x*j + threadIdx.x < maxN; j++){
				int index = j*blockDim.x+i+ghosts;
				printf("%10f\t%10d\t%f\n", *timeElapsed, index-ghosts, a[index]);
			}
		}
		if(i==0){	//copy over for boundary conditions
			for(int j = 0; j < ghosts; j++){
				a[j] = a[j+n];
				a[n+ghosts+j] = a[ghosts+j];
				// a[j] = a[ghosts];
				// a[n+ghosts+j] = a[n+ghosts-1];
			}
			
			// printf("%f\n",dt);
			// printf("%p %f\n", dtt, *dtt);
			*timeElapsed += *dt;
			(*counter)++;
			if(*counter == 10000)
				*counter = 0;
		}
	}
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

__global__ void initSquare(double* a, double* x, double totalX, int n, int ghosts){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int j = 0; blockDim.x*j + i < n; j++){
		int index = j*blockDim.x+i;
		if(index > n/3 && index < 2*n/3)
			a[index+ghosts] = 1.5;
		else a[index+ghosts] = .5;
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
	numBlocks = dim/maxThreads;
	if(dim%maxThreads != 0)
		numBlocks++;
	cudaMalloc((void**)&d_a, (dim+ghosts*2)*sizeof(double));
	cudaMalloc((void**)&d_x, (dim+ghosts*2)*sizeof(double));
	cudaMalloc((void**)&d_oA, (dim+ghosts*2)*sizeof(double));
	cudaDeviceSynchronize();
	initialize<<<1, 1024>>>(d_a, d_oA, d_x, totalX, dim, ghosts);
	cudaDeviceSynchronize();
	// initSinusoid<<<numBlocks, 1024>>>(d_a, d_x, totalX, dim, ghosts, 1, 0.5);
	initSquare<<<1, 1024>>>(d_a, d_x, totalX, dim, ghosts);
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

void CFD::step(double maxtime){
	double *dt,*te, *minDx;
	int* counter;
	cudaMalloc((void**)&dt, sizeof(double));
	cudaMalloc((void**)&te, sizeof(double));
	cudaMalloc((void**)&counter, sizeof(int));
	cudaMalloc((void**)&minDx, sizeof(double));
	cudaDeviceSynchronize();
	advect<<<numBlocks, 1024>>>(d_a, d_oA, d_x, u, dim, ghosts, minDx, dt, te, counter, maxtime);
	cudaDeviceSynchronize();
}