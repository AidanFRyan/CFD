#include <iostream>
#include <stdio.h>
#include <cooperative_groups.h>
using namespace std;
using namespace cooperative_groups;
__global__ void initialize(double* a, double* oA, double* x, double totalSize, int n, int ghosts);
__global__ void advect(double dt, double* a, double* oA, double* x, double u, int n, int ghosts, double tmax);
__global__ void initSinusoid(double* a, double* x, double totalX, int n, int ghosts, double shift, double amp);
__global__ void initSquare(double* a, double* x, double totalX, int n, int ghosts);
__device__ void setA(int x, double init, double* a);
__device__ double linInterp(double* in);
__device__ double colellaEvenInterp(double*in);

class CFD{
public:
	CFD(int x, double size, double uIn);	//number of cells, size of cells, velocity
	void setInitial(int x, double init);
	void step(double maxtime);	//solve until maxtime
	double* getA();
	int getDim();
private:
	int dim, ghosts;
	double* a, u, *x;
	double* d_a, *d_x, *d_oA;
	double totalX;
	const int maxThreads = 1024;
	int numBlocks;
};