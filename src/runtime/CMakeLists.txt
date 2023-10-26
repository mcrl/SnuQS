add_library(runtime
    STATIC 
    rt_handle.cc
    rt_stream.cc
    rt_queue.cc
    rt_worker_pool.cc
    rt_task.cc
    rt_kernel.cc
    rt_host.cc
    rt_device.cc
    rt_storage.cc
    rt_soft_raid0.cc
    )

target_link_libraries(runtime CUDA::cudart)
set_target_properties(runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
