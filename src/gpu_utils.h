#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>

namespace {

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

} // namespace 


namespace snuqs {
namespace gpu {

using stream_t = cudaStream_t;
using event_t = cudaEvent_t;

static size_t getNumDevices() 
{
	int count;
	gpuErrchk(cudaGetDeviceCount(&count));
	return count;
}

static int getDevice() 
{
	int d;
	gpuErrchk(cudaGetDevice(&d));
	return d;
}

static void setDevice(int d) 
{
	gpuErrchk(cudaSetDevice(d));
}

static void Malloc(void **buf, size_t count)
{
	gpuErrchk(cudaMalloc(buf, count));
}

static void Free(void *b)
{
	gpuErrchk(cudaFree(b));
}

static void MallocHost(void **buf, size_t count)
{
	gpuErrchk(cudaMallocHost(buf, count));
}

static void FreeHost(void *b)
{
	gpuErrchk(cudaFreeHost(b));
}

static void checkLastError()
{
	gpuErrchk(cudaGetLastError());
}

static void MemcpyAsyncH2D(void *dst, const void *src, size_t count, stream_t s=0)
{
	gpuErrchk(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, s));
}

static void MemcpyAsyncD2H(void *dst, const void *src, size_t count, stream_t s=0)
{
	gpuErrchk(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, s));
}

static void streamCreate(stream_t *stream)
{
	gpuErrchk(cudaStreamCreate(stream));
}

static void streamDestroy(stream_t stream)
{
	gpuErrchk(cudaStreamDestroy(stream));
}

static void streamSynchronize(stream_t stream)
{
	gpuErrchk(cudaStreamSynchronize(stream));
}

static void eventCreate(event_t *event)
{
	gpuErrchk(cudaEventCreate(event));
}

static void eventDestroy(event_t event)
{
	gpuErrchk(cudaEventDestroy(event));
}

static void enqueueEvent(stream_t stream, event_t event)
{
	gpuErrchk(cudaEventRecord(event, stream));
}

static void streamWaitEvent(stream_t stream, event_t event)
{
	gpuErrchk(cudaStreamWaitEvent(stream, event));
}

static void deviceSynchronize()
{
	gpuErrchk(cudaDeviceSynchronize());
}

static void launchHostFunc(stream_t stream, cudaHostFn_t fn, void *data)
{
	gpuErrchk(cudaLaunchHostFunc(stream,fn, data));
}

} // gpu
} // snuqs
