#include "gpu_utils.h"

namespace snuqs {
namespace gpu {

using stream_t = cudaStream_t;
using event_t = cudaEvent_t;

size_t getNumDevices() 
{
	int count;
	gpuErrchk(cudaGetDeviceCount(&count));
	return count;
}

int getDevice() 
{
	int d;
	gpuErrchk(cudaGetDevice(&d));
	return d;
}

void setDevice(int d) {
	gpuErrchk(cudaSetDevice(d));
}

void Malloc(void **buf, size_t count)
{
	gpuErrchk(cudaMalloc(buf, count));
}

void Free(void *b)
{
	gpuErrchk(cudaFree(b));
}

void MallocHost(void **buf, size_t count)
{
	gpuErrchk(cudaMallocHost(buf, count));
}

void FreeHost(void *b)
{
	gpuErrchk(cudaFreeHost(b));
}

void checkLastError()
{
	gpuErrchk(cudaGetLastError());
}

void MemcpyAsyncH2D(void *dst, const void *src, size_t count, stream_t s=0)
{
	gpuErrchk(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, s));
}

void MemcpyAsyncD2H(void *dst, const void *src, size_t count, stream_t s=0)
{
	gpuErrchk(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, s));
}

void streamCreate(stream_t *stream)
{
	gpuErrchk(cudaStreamCreate(stream));
}

void streamDestroy(stream_t stream)
{
	gpuErrchk(cudaStreamDestroy(stream));
}

void streamSynchronize(stream_t stream)
{
	gpuErrchk(cudaStreamSynchronize(stream));
}

void eventCreate(event_t *event)
{
	gpuErrchk(cudaEventCreate(event));
}

void eventDestroy(event_t event)
{
	gpuErrchk(cudaEventDestroy(event));
}

void enqueueEvent(stream_t stream, event_t event)
{
	gpuErrchk(cudaEventRecord(event, stream));
}

void streamWaitEvent(stream_t stream, event_t event)
{
	gpuErrchk(cudaStreamWaitEvent(stream, event));
}

void deviceSynchronize()
{
	gpuErrchk(cudaDeviceSynchronize());
}

void launchHostFunc(stream_t stream, cudaHostFn_t fn, void *data)
{
	gpuErrchk(cudaLaunchHostFunc(stream,fn, data));
}

} // gpu
} // snuqs
