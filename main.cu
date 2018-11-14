#include "cfd.h"
#include <cmath>

int main(){
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	CFD cfd = CFD(1000, 10, 12);
	double ts = 0;
	double maxTime = .01;
	for(double totaltime = 0; totaltime < maxTime;){
		ts = cfd.step();
		// cout<<totaltime<<endl;
		totaltime += ts;
		// cout<<fmod(totaltime, .1)<<endl;
		if(fmod(totaltime, .0001) <= 0.0000001){
			 double* temp = cfd.getA();
			 for(int i = 0; i < cfd.getDim(); i++){
			 	printf("%10f\t%10d\t%f\n", totaltime, i, temp[i]);
			 }
		}
	}
	return 0;
}