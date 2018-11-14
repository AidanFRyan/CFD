main: main.o cfd.o
	nvcc -arch=sm_61 -O3 -o main main.o cfd.o
main.o: main.cu cfd.h
	nvcc -arch=sm_61 -O3 -c main.cu
cfd.o: cfd.cu cfd.h
	nvcc -arch=sm_61 -O3 -c cfd.cu
debug: dbg1.o dbg2.o
	nvcc -g -G -arch=sm_61 -o debug dbg1.o dbg2.o
dbg1.o: main.cu cfd.h
	nvcc -g -G -arch=sm_61 -c -o dbg1.o main.cu
dbg2.o: cfd.cu cfd.h
	nvcc -g -G -arch=sm_61 -c -o dbg2.o cfd.cu
clean:
	rm *.o main