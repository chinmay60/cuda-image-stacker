#include <iostream>
#include "utils.h"

void stackFramesSeq(uchar4* h_lightFrames, uchar4* h_outputFrame, int width, int height, int numberOfImages)
{
    std::clock_t start;
    double duration;
    start = std::clock();
  	float red = 0, green = 0, blue = 0;
  	for(int index = 0; index < width*height ; index++)
    {
        for(int i = 0; i < numberOfImages; i++) 
        {
            uchar4 currentPixelValue = h_lightFrames[index + (width * height)*i];
            red = (red + (int)currentPixelValue.x);
            green = (green + (int)currentPixelValue.y);
            blue = (blue + (int)currentPixelValue.z);
  	    }
  	    uchar4 outputPixel = make_uchar4(red/numberOfImages, green/numberOfImages, blue/numberOfImages, 255);
  	    h_outputFrame[index] = outputPixel;
    }
    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC ) * 1000;
    printf("Time for stacking frames (Sequential): %f msecs.\n", duration);
}

void subtractFramesSeq(int width, int height, uchar4* h_firstFrame, uchar4* h_secondFrame, uchar4* h_FinalImage)
{
    std::clock_t start;
    start = std::clock();
    double duration;
	for(int index = 0; index < width*height ; index++)
    {
	   uchar4 lightFramePixel = h_firstFrame[index];
       uchar4 darkFramePixel = h_secondFrame[index];
       unsigned char red = lightFramePixel.x - darkFramePixel.x;
       unsigned char green = lightFramePixel.y - darkFramePixel.y;
       unsigned char blue = lightFramePixel.z - darkFramePixel.z;
       uchar4 outputPixel = make_uchar4(red, green, blue, 255);
       h_FinalImage[index] = outputPixel;
    }
    duration = (( std::clock() - start ) / (double) CLOCKS_PER_SEC ) * 1000;
    printf("Time for subtracting frames (Parallel): %f msecs.\n", duration);
}

