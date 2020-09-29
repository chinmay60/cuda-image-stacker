#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "utils.h"

__global__
void calculateAverageLightFrames(int width, int height, int numberOfImages, uchar4* d_lightFrames, uchar4* d_outputFrame)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  	if(index_y > width || index_x > height)
  		return;
  	int grid_width = gridDim.x * blockDim.x;
  	int index = index_y * grid_width + index_x;
  	float red = 0, green = 0, blue = 0;
  	for(int i = 0; i < numberOfImages; i++)
  	{
  		uchar4 currentPixelValue = d_lightFrames[index + ((width * height) * i)];
  		red = (red + (int)currentPixelValue.x);
  		green = (green + (int)currentPixelValue.y);
  		blue = (blue + (int)currentPixelValue.z);
  		// if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)
  		// 	printf("- %d - %d - %d - %f\n", numberOfImages, width, height, red);
  	}
  	uchar4 outputPixel = make_uchar4(red/numberOfImages, green/numberOfImages, blue/numberOfImages, 255);
  	d_outputFrame[index] = outputPixel;
}

__global__
void subtractFrames(int width, int height, uchar4* d_LightFrame, uchar4* d_DarkFrame, uchar4* d_FinalImage)
{
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  	if(index_y > width || index_x > height)
  		return;
  	int grid_width = gridDim.x * blockDim.x;
  	int index = index_y * grid_width + index_x;

  	uchar4 lightFramePixel = d_LightFrame[index];
  	uchar4 darkFramePixel = d_DarkFrame[index];

  	unsigned char red = lightFramePixel.x - darkFramePixel.x;
  	unsigned char green = lightFramePixel.y - darkFramePixel.y;
  	unsigned char blue = lightFramePixel.z - darkFramePixel.z;
  	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  	d_FinalImage[index] = outputPixel;
}

__global__
void divideFrames(int width, int height, uchar4* d_LightFrame, uchar4* d_FlatFrame, uchar4* d_FinalImage, uchar4 avgValue)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(index_y > width || index_x > height)
        return;
    int grid_width = gridDim.x * blockDim.x;
    int index = index_y * grid_width + index_x;

    uchar4 lightFramePixel = d_LightFrame[index];
    uchar4 flatFramePixel = d_FlatFrame[index];

    unsigned char red = (lightFramePixel.x/flatFramePixel.x) * avgValue.x;
    unsigned char green = (lightFramePixel.y/flatFramePixel.y) * avgValue.y;
    unsigned char blue = (lightFramePixel.z/flatFramePixel.z) * avgValue.z;
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    d_FinalImage[index] = outputPixel;
}

void processFrames(uchar4* d_lightFrames, uchar4* d_outputLightFrame, int width, int height, int numberOfImages)
{
	std::cout << "Calculating Average of the Frames" << std::endl;
	//std::cout << width << " - "<< height << " - " << numberOfImages << std::endl;
	const int thread = 16;
	const dim3 blockSize(thread, thread);
	const dim3 gridSize(ceil(height/(float)thread), ceil(width/(float)thread));
	calculateAverageLightFrames<<<gridSize, blockSize>>>(width, height, numberOfImages, d_lightFrames, d_outputLightFrame);
	cudaDeviceSynchronize(); 
	checkCudaErrors(cudaGetLastError());
}

void subtractFrames(uchar4* d_outputLightFrame, uchar4* d_outputDarkFrame, uchar4* d_FinalImage, int width, int height)
{
	const int thread = 16;
	const dim3 blockSize(thread, thread);
	const dim3 gridSize(ceil(height/(float)thread), ceil(width/(float)thread));
	subtractFrames<<<gridSize, blockSize>>>(width, height, d_outputLightFrame, d_outputDarkFrame, d_FinalImage);
	cudaDeviceSynchronize(); 
	checkCudaErrors(cudaGetLastError());
}

void applyFlatFramesDivision(uchar4* d_outputLightFrame, uchar4* d_outputDarkFrame, uchar4* d_FinalImage, uchar4 avgValue, int width, int height)
{
    const int thread = 16;
    const dim3 blockSize(thread, thread);
    const dim3 gridSize(ceil(height/(float)thread), ceil(width/(float)thread));
    divideFrames<<<gridSize, blockSize>>>(width, height, d_outputLightFrame, d_outputDarkFrame, d_FinalImage, avgValue);
    cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());
}