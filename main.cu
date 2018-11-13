#include "cfd.h"

int main(){
	CFD cfd = CFD(1000, 10);
	cfd.step(.0001);
	return 0;
}