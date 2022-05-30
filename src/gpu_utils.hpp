#pragma once

#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

namespace {

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

size_t getNumDevices(); 
int getDevice();
void setDevice(int d); 
void Malloc(void **buf, size_t count);
void Free(void *b);
void MallocHost(void **buf, size_t count);
void FreeHost(void *b);
void checkLastError();
void MemcpyAsyncH2D(void *dst, const void *src, size_t count, stream_t s);
void MemcpyAsyncD2H(void *dst, const void *src, size_t count, stream_t s);
void streamCreate(stream_t *stream);
void streamDestroy(stream_t stream);
void streamSynchronize(stream_t stream);
void eventCreate(event_t *event);
void eventDestroy(event_t event);
void enqueueEvent(stream_t stream, event_t event);
void streamWaitEvent(stream_t stream, event_t event);
void deviceSynchronize();
void launchHostFunc(stream_t stream, cudaHostFn_t fn, void *data);

} // gpu
} // snuqs
