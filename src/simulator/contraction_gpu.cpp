#include "simulator.hpp"
#include <cuda_runtime.h>
#include <cutensornet.h>

#define HANDLE_ERROR(x)                                               \
do { \
	const auto err = (x);                                                  \
	if( err != CUTENSORNET_STATUS_SUCCESS ) {\
		printf("Error: %s in line %d\n", cutensornetGetErrorString(err), __LINE__);\
		exit(err);\
	} \
} while(0)

#define HANDLE_CUDA_ERROR(x)                                      \
do { \
	const auto err = (x);                                                  \
	if( err != cudaSuccess ) {\
		printf("Error: %s in line %d\n", cudaGetErrorString(err), __LINE__);\
		exit(err);\
	} \
} while(0)

namespace snuqs {

namespace {

struct GPUTimer
{
   GPUTimer(cudaStream_t stream): stream_(stream)
   {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
   }

   ~GPUTimer()
   {
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
   }

   void start()
   {
      cudaEventRecord(start_, stream_);
   }

   float seconds()
   {
      cudaEventRecord(stop_, stream_);
      cudaEventSynchronize(stop_);
      float time;
      cudaEventElapsedTime(&time, start_, stop_);
      return time * 1e-3;
   }

   private:
   cudaEvent_t start_, stop_;
   cudaStream_t stream_;
};


void simulate(std::vector<Tensor*> &tensors) {
   typedef std::complex<double> doubleType;
   cudaDataType_t typeData = CUDA_C_64F;
   cutensornetComputeType_t typeCompute = CUTENSORNET_COMPUTE_64F;

   // Sphinx: #2
   // Computing: D_{m,x,n,y} = A_{m,h,k,n} B_{u,k,h} C_{x,u,y}


   std::cout << "numTensors: " << tensors.size() << "\n";
   //int32_t numTensors = tensors.size();
   int32_t numTensors = 10;
   int32_t numInputs = numTensors;

   std::vector<std::vector<int32_t>> modes(numTensors);
   std::vector<std::vector<int64_t>> extents(numTensors);

   for (int32_t i = 0; i < numTensors; ++i) {
   	   auto t = tensors[i];
   	   for (auto e : t->inedges_) {
   	   	   modes[i].push_back(e);
   	   	   extents[i].push_back(2);
	   }
   	   for (auto e : t->outedges_) {
   	   	   modes[i].push_back(e);
   	   	   extents[i].push_back(2);
	   }
   }

   //
   // Allocate workspace
   //

   size_t freeMem, totalMem;
   HANDLE_CUDA_ERROR( cudaMemGetInfo(&freeMem, &totalMem ));

   uint64_t worksize = freeMem * 0.9;

   void *work = nullptr;
   HANDLE_CUDA_ERROR( cudaMalloc(&work, worksize) );

   // Sphinx: #3
   // Allocating data
   std::vector<void*> rawDataIn_d;
   for (int32_t i = 0; i < numTensors; ++i) {
   	   size_t elements = 1;
   	   for (auto e : extents[i])
   	   	   elements *= e;
   	   size_t size = sizeof(doubleType) * elements;
   	   HANDLE_CUDA_ERROR(cudaMalloc(&(tensors[i]->d_mem_), size));
	   HANDLE_CUDA_ERROR(cudaMemcpy(tensors[i]->d_mem_, tensors[i]->mem_.data(), size, cudaMemcpyHostToDevice));
	   rawDataIn_d.push_back(tensors[i]->d_mem_);
   }


   //
   // Initialize data
   //

   printf("Allocate memory for data and workspace, and initialize data.\n");

   // Sphinx: #4
   //
   // cuTensorNet
   //

   cudaStream_t stream;
   cudaStreamCreate(&stream);

   cutensornetHandle_t handle;
   HANDLE_ERROR(cutensornetCreate(&handle));

   //
   // Create Network Descriptor
   //

   std::vector<int32_t*> modesIn;
   std::vector<int32_t> numModesIn;
   std::vector<int64_t*> extentsIn;

   for (int32_t i = 0; i < numTensors; ++i) {
	   modesIn.push_back(modes[i].data());
	   numModesIn.push_back(static_cast<int32_t>(modes[i].size()));
	   extentsIn.push_back(extents[i].data());
   }

   void *D_d;
   int32_t nmodeD = numInputs;
   std::vector<int32_t> modesD;
   std::vector<int64_t> extentD;
   size_t sizeD =  sizeof(doubleType);
   for (int32_t i = 0; i < numInputs; ++i) {
   	   modesD.push_back(i);
   	   extentD.push_back(2);
   }
   HANDLE_CUDA_ERROR(cudaMalloc(&D_d, sizeD));
   HANDLE_CUDA_ERROR(cudaMemset(D_d, 0, sizeD));

   auto getMaximalPointerAlignment = [](const void* ptr) {
      const uint64_t ptrAddr  = reinterpret_cast<uint64_t>(ptr);
      uint32_t alignment = 1;
      while(ptrAddr % alignment == 0 &&
            alignment < 256) // at the latest we terminate once the alignment reached 256 bytes (we could be going, but any alignment larger or equal to 256 is equally fine)
      {
         alignment *= 2;
      }
      return alignment;
   };
   const uint32_t alignmentOut = getMaximalPointerAlignment(D_d);
   std::vector<uint32_t> alignmentsIn;

   for (int32_t i = 0; i < numTensors; ++i) {
   	   alignmentsIn.push_back(getMaximalPointerAlignment(rawDataIn_d[i]));
   }

   // setup tensor network
   cutensornetNetworkDescriptor_t descNet;
   HANDLE_ERROR(cutensornetCreateNetworkDescriptor(handle,
                                                numInputs, numModesIn.data(), extentsIn.data(), NULL, modesIn.data(), alignmentsIn.data(),
                                                nmodeD, extentD.data(), NULL, modesD.data(), alignmentOut,
                                                typeData, typeCompute,
                                                &descNet));

   printf("Initialize the cuTensorNet library and create a network descriptor.\n");

   // Sphinx: #5
   //
   // Find "optimal" contraction order and slicing
   //

   cutensornetContractionOptimizerConfig_t optimizerConfig;
   HANDLE_ERROR (cutensornetCreateContractionOptimizerConfig(handle, &optimizerConfig));

   // Set the value of the partitioner imbalance factor, if desired
   int imbalance_factor = 30;
   HANDLE_ERROR(cutensornetContractionOptimizerConfigSetAttribute(
                                                               handle,
                                                               optimizerConfig,
                                                               CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_IMBALANCE_FACTOR,
                                                               &imbalance_factor,
                                                               sizeof(imbalance_factor)));

   cutensornetContractionOptimizerInfo_t optimizerInfo;
   HANDLE_ERROR (cutensornetCreateContractionOptimizerInfo(handle, descNet, &optimizerInfo));

   HANDLE_ERROR (cutensornetContractionOptimize(handle,
                                             descNet,
                                             optimizerConfig,
                                             worksize,
                                             optimizerInfo));

   int64_t numSlices = 0;
   HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
               handle,
               optimizerInfo,
               CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_NUM_SLICES,
               &numSlices,
               sizeof(numSlices)));

   assert(numSlices > 0);

   printf("Find an optimized contraction path with cuTensorNet optimizer.\n");

   // Sphinx: #6
   //
   // Initialize all pair-wise contraction plans (for cuTENSOR)
   //
   cutensornetContractionPlan_t plan;

   cutensornetWorkspaceDescriptor_t workDesc;
   HANDLE_ERROR(cutensornetCreateWorkspaceDescriptor(handle, &workDesc));

   uint64_t requiredWorkspaceSize = 0;
   HANDLE_ERROR(cutensornetWorkspaceComputeSizes(handle,
                                          descNet,
                                          optimizerInfo,
                                          workDesc));

   HANDLE_ERROR(cutensornetWorkspaceGetSize(handle,
                                         workDesc,
                                         CUTENSORNET_WORKSIZE_PREF_MIN,
                                         CUTENSORNET_MEMSPACE_DEVICE,
                                         &requiredWorkspaceSize));
   if (worksize < requiredWorkspaceSize)
   {
      printf("Not enough workspace memory is available.");
   }
   else
   {
      HANDLE_ERROR (cutensornetWorkspaceSet(handle,
                                            workDesc,
                                            CUTENSORNET_MEMSPACE_DEVICE,
                                            work,
                                            worksize));
                                          
      HANDLE_ERROR( cutensornetCreateContractionPlan(handle,
                                                     descNet,
                                                     optimizerInfo,
                                                     workDesc,
                                                     &plan) );

      //
      // Optional: Auto-tune cuTENSOR's cutensorContractionPlan to pick the fastest kernel
      //
      cutensornetContractionAutotunePreference_t autotunePref;
      HANDLE_ERROR(cutensornetCreateContractionAutotunePreference(handle,
                              &autotunePref));

      const int numAutotuningIterations = 5; // may be 0
      HANDLE_ERROR(cutensornetContractionAutotunePreferenceSetAttribute(
                              handle,
                              autotunePref,
                              CUTENSORNET_CONTRACTION_AUTOTUNE_MAX_ITERATIONS,
                              &numAutotuningIterations,
                              sizeof(numAutotuningIterations)));

      // modify the plan again to find the best pair-wise contractions
      HANDLE_ERROR(cutensornetContractionAutotune(handle,
                              plan,
                              rawDataIn_d.data(),
                              D_d,
                              workDesc,
                              autotunePref,
                              stream));

      HANDLE_ERROR(cutensornetDestroyContractionAutotunePreference(autotunePref));

      printf("Create a contraction plan for cuTENSOR and optionally auto-tune it.\n");

      // Sphinx: #7
      //
      // Run
      //
	  GPUTimer timer{stream};
	  double minTimeCUTENSOR = 1e100;
	  cudaDeviceSynchronize();

	  //
	  // Contract over all slices.
	  //
	  // A user may choose to parallelize this loop across multiple devices.
	  // (Note, however, that as of cuTensorNet v1.0.0 the contraction must
	  // start from slice 0, see the cutensornetContraction documentation at
	  // https://docs.nvidia.com/cuda/cuquantum/cutensornet/api/functions.html#cutensornetcontraction )
	  //
	  for (int64_t sliceId=0; sliceId < numSlices; ++sliceId)
	  {
		  timer.start();

		  HANDLE_ERROR(cutensornetContraction(handle,
					  plan,
					  rawDataIn_d.data(),
					  D_d,
					  workDesc, sliceId, stream));

		  // Synchronize and measure timing
		  auto time = timer.seconds();
		  minTimeCUTENSOR = (minTimeCUTENSOR < time) ? minTimeCUTENSOR : time;
	  }

      printf("Contract the network, each slice uses the same contraction plan.\n");

      double flops = -1;

      HANDLE_ERROR( cutensornetContractionOptimizerInfoGetAttribute(
                  handle,
                  optimizerInfo,
                  CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_FLOP_COUNT,
                  &flops,
                  sizeof(flops)));

      printf("numSlices: %ld\n", numSlices);
      printf("%.2f ms / slice\n", minTimeCUTENSOR * 1000.f);
      printf("%.2f GFLOPS/s\n", flops/1e9/minTimeCUTENSOR );
   }

   HANDLE_ERROR(cutensornetDestroy(handle));
   HANDLE_ERROR(cutensornetDestroyNetworkDescriptor(descNet));
   HANDLE_ERROR(cutensornetDestroyContractionPlan(plan));
   HANDLE_ERROR(cutensornetDestroyContractionOptimizerConfig(optimizerConfig));
   HANDLE_ERROR(cutensornetDestroyContractionOptimizerInfo(optimizerInfo));
   HANDLE_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));


   for (int32_t i = 0; i < numInputs; ++i) {
   	   std::cout << *tensors[i] << "\n";
   }

   void *D = malloc(sizeD);
   HANDLE_CUDA_ERROR(cudaMemcpy(D, D_d, sizeD, cudaMemcpyDeviceToHost));
   for (size_t i = 0; i < sizeD / sizeof(doubleType); ++i) {
   	   std::cout << ((doubleType*)D)[i] << " ";
   }
   std::cout << "\n";

   if (D_d) cudaFree(D_d);
   if (work) cudaFree(work);

   printf("Free resource and exit.\n");
}

}


void ContractionGPUSimulator::init(size_t num_qubits) {
}

void ContractionGPUSimulator::deinit() {
}

void ContractionGPUSimulator::run(const std::vector<QuantumCircuit> &circs) {

	size_t num_qubits = circs[0].num_qubits();

	std::vector<unsigned int> index_map(num_qubits);
	std::vector<Tensor*> tensors;


	for (auto &circ: circs) {

		unsigned int nedges = 0;
		for (size_t i = 0; i < num_qubits; ++i) {
			tensors.push_back(new Tensor({}, {nedges}, {1, 0}));
			index_map[i] = nedges++;
		}

		for (auto g : circ.gates()) {
			Tensor *t = g->toTensor();
			tensors.push_back(t);

			std::vector<unsigned int> inedges;
			std::vector<unsigned int> outedges;
			for (auto q : g->qubits()) {
				inedges.push_back(index_map[q]);
			}

			for (auto q : g->qubits()) {
				outedges.push_back(nedges);
				index_map[q] = nedges;
				nedges++;
			}

			t->setInEdges(inedges);
			t->setOutEdges(outedges);
		}
		
		for (size_t i = 0; i < num_qubits; ++i) {
			tensors.push_back(new Tensor({nedges}, {}, {1, 0}));
			index_map[i] = nedges++;
		}

		simulate(tensors);
	}
}

} // namespace snuqs
