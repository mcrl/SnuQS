add_library(runtime
    STATIC 
    rt_handle.cpp
    rt_stream.cpp
    rt_queue.cpp
    rt_worker_pool.cpp
    rt_task.cpp
    rt_kernel.cpp
    rt_host.cpp
    rt_device.cpp
    rt_storage.cpp
    rt_soft_raid0.cpp
    )

target_link_libraries(runtime CUDA::cudart)
set_target_properties(runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
