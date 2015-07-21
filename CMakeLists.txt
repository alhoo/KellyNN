cmake_minimum_required(VERSION 2.8.5)
project(brain)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
#set(CMAKE_CXX_FLAGS "-std=c++0x")
set(CMAKE_CXX_FLAGS "-std=c++11")

# Add all the code for the current module
file ( GLOB CODE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
include(FindOpenCL)
Find_Package ( OpenCL REQUIRED )
link_libraries(${OPENCL_LIBRARIES})
include_directories(${OPENCL_INCLUDE_DIR})
# Create the library
add_library ( brain STATIC ${CODE})
#target_compile_features(brain PRIVATE cxx_auto_type)

enable_testing()
include(ExternalProject)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
ExternalProject_Add(
  gtest-1.7.0
  PREFIX ${CMAKE_BINARY_DIR}/vendor/gtest
  URL http://localhost/files/gtest-1.7.0.zip
  #URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG:PATH=DebugLibs
             -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE:PATH=ReleaseLibs
              -Dgtest_force_shared_crt=ON
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
)
ExternalProject_Get_Property(gtest-1.7.0 source_dir)
ExternalProject_Get_Property(gtest-1.7.0 binary_dir)
include_directories(${source_dir}/include)

#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIR})
#neural_map_tests.cc       simple_opencl_class.cc    simple_opencl_test.cc
add_executable(neural-map-tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/neural_map_tests.cc )
add_dependencies(neural-map-tests gtest-1.7.0)
#target_compile_features(neural-map-tests PRIVATE cxx_auto_type)
#target_link_libraries(neural-map-tests brain ${GTEST_LIBRARY_DEBUG} ${GTEST_MAIN_LIBRARY_DEBUG})
if(MSVC)
  set(Suffix ".lib")
else()
  set(Suffix ".a")
  set(Pthread "-pthread")
endif()
target_link_libraries(
  neural-map-tests brain
  debug ${binary_dir}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
  optimized ${binary_dir}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
  ${Pthread})
add_test(
    NAME neural-map-tests
    COMMAND neural-map-tests
)

add_executable(brain-function-tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/brain_functions_tests.cc )
add_dependencies(brain-function-tests gtest-1.7.0)
if(MSVC)
  set(Suffix ".lib")
else()
  set(Suffix ".a")
  set(Pthread "-pthread")
endif()
target_link_libraries(
  brain-function-tests brain
  debug ${binary_dir}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
  optimized ${binary_dir}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
  ${Pthread})
add_test(
    NAME brain-function-tests
    COMMAND brain-function-tests
)
# Grab all the test cases 
#file ( GLOB TESTS_CODE ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc ) 
# Create test program
#add_executable ( BrainTests ${TESTS_CODE} ) 
#target_link_libraries ( BrainTests brain ) 
#add_test ( BrainTests BrainTests )
