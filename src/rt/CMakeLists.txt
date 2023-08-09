add_library(rt
    SHARED 
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

target_link_libraries(rt CUDA::cudart)
