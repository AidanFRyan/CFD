__global__ void initialize(double* a, double* oA, double* x, double totalSize, int n, int ghosts);
__global__ void advect(double dt, double* a, double* oA, double* x, double u, int n, int ghosts);
__global__ void initSinusoid(double* a, double* x, double totalX, int n, int ghosts, double shift, double amp);
__device__ void setA(int x, double init, double* a);
__device__ double linInterp(double* in);

class CFD{
public:
	CFD(int x, double size);
	void setInitial(int x, double init);	//create point of energy at specific cell
	void step(double dt);	//solve for time step dt

private:
	int dim;
	double* a, u, *x;
	double* d_a, *d_x, *d_oA;
	double totalX;
};