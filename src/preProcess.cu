#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sm_20_atomic_functions.h>
#include <math.h>
#include <stdio.h>


__global__ void cudaPreProcessKenerl(float* img_dst, unsigned char* img_source, int width, int height, int channel, int num)
{
    for (size_t i = 0; i < num; i++)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x >= width)
        {
            return;
        }
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y >= height)
        {
            return;
        }
        int c = blockIdx.z;
        int dst = i * width * height * channel + c * width * height + y * width + x;
        // BGR cvtColor to RGB div 255
        if (c == 0)
        {
            int source = i * width * height * channel + (y * width + x) * channel + 2;
            img_dst[dst] = img_source[source] / 255.0;
        }
        else if (c == 1)
        {
            int source = i * width * height * channel + (y * width + x) * channel + 1;
            img_dst[dst] = img_source[source] / 255.0;
        }
        else if (c == 2)
        {
            int source = i * width * height * channel + (y * width + x) * channel + 0;
            img_dst[dst] = img_source[source] / 255.0;
        }
    }
}


extern "C" void cudaPreProcess(float* img_dst, unsigned char* img_source, int width, int height, int channel, int num, cudaStream_t stream)
{
    dim3 threadPerBlock(16, 16);
    dim3 blockPerGrid((width + 15) / 16, (height + 15) / 16, channel);
    cudaPreProcessKenerl << <blockPerGrid, threadPerBlock, 0, stream >> > (img_dst, img_source, width, height, channel, num);
}