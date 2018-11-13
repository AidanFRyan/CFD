main: main.o cfd.o
	nvcc -arch=sm_61 -o main main.o cfd.o
main.o: main.cu cfd.h
	nvcc -arch=sm_61 -c main.cu
cfd.o: cfd.cu cfd.h
	nvcc -arch=sm_61 -c cfd.cu
clean:
	rm *.o main