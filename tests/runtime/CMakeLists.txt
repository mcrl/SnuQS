enable_testing()

add_executable(rt_test rt_test.cc
)

target_link_libraries(rt_test
    runtime
    GTest::gtest_main
    )

include(GoogleTest)
gtest_discover_tests(rt_test)
