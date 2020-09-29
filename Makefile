NVCC=nvcc

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/local/include/opencv4

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda-10.1/include

NVCC_OPTS=-O3 -Xcompiler -Wall -Xcompiler -Wextra -m64 -arch=sm_75

GCC_OPTS=-O3 -Wall -Wextra -m64 -Wmaybe-uninitialized -Wunused-parameter

main: main.o calculations.o calculationsSequential.o Makefile
	$(NVCC) -o main main.o calculations.o calculationsSequential.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS)

main.o: main.cpp utils.h timer.h process.cpp 
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH) -I .

calculations.o: calculations.cu utils.h
	nvcc -c calculations.cu $(NVCC_OPTS)

calculationsSequential.o: calculationsSequential.cpp utils.h
	g++ -c calculationsSequential.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH) -I .

clean:
	rm -f *.o main ./Output/*.JPG ./Light/Stacked/*.JPG ./Dark/Stacked/*.JPG ./Bias/Stacked/*.JPG log.txt
