cmake_minimum_required(VERSION 2.8.5)
project(brain)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Add all the code for the current module
file ( GLOB CODE ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
include(FindOpenCL)
Find_Package ( OpenCL REQUIRED )
link_libraries(${OPENCL_LIBRARIES})
include_directories(${OPENCL_INCLUDE_DIR})
# Create the library
add_library ( brain STATIC ${CODE})

enable_testing()
include(ExternalProject)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
ExternalProject_Add(
  gtest-1.7.0
  PREFIX ${CMAKE_BINARY_DIR}/vendor/gtest
  URL https://googletest.googlecode.com/files/gtest-1.7.0.zip
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
include_directories(${source_dir}/include)

#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIR})
add_executable(brain-tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/brain_functions_tests.cc )
add_dependencies(brain-tests gtest-1.7.0)
ExternalProject_Get_Property(gtest-1.7.0 binary_dir)
#target_link_libraries(brain-tests brain ${GTEST_LIBRARY_DEBUG} ${GTEST_MAIN_LIBRARY_DEBUG})
if(MSVC)
  set(Suffix ".lib")
else()
  set(Suffix ".a")
  set(Pthread "-pthread")
endif()
target_link_libraries(
  brain-tests brain
  debug ${binary_dir}/DebugLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
  optimized ${binary_dir}/ReleaseLibs/${CMAKE_FIND_LIBRARY_PREFIXES}gtest${Suffix}
  ${Pthread})
add_test(
    NAME brain-tests
    COMMAND brain-tests
)
# Grab all the test cases 
#file ( GLOB TESTS_CODE ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc ) 
# Create test program
#add_executable ( BrainTests ${TESTS_CODE} ) 
#target_link_libraries ( BrainTests brain ) 
#add_test ( BrainTests BrainTests )
