#include "cfd.h"
#include <cmath>

int main(){
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	CFD cfd = CFD(1000, 10, 10);
	double maxTime = 1;
	cfd.step(maxTime);
	return 0;
}