#include "cfd.h"
#include <cmath>

int main(){
	//cudaSetDevice(1);
	cudaDeviceSynchronize();
	CFD cfd = CFD(19, 1, 1);
	float maxTime = 10;
	cfd.step(maxTime);
	return 0;
}
