
PROJECT(gnoimi-impl)
CMAKE_MINIMUM_REQUIRED(VERSION 3.2)

FIND_PACKAGE(GFLAGS REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -std=c++14 -m64")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -g -std=c++14 -m64")

SET(MKL_DIR /home/bismarckdd/intel/mkl/include)
SET(MKL_LIB /home/bismarckdd/intel/mkl/lib/intel64)
SET(YAEL_DIR /usr/local/include)
SET(YAEL_LIB /usr/local/lib)

INCLUDE_DIRECTORIES(${YAEL_DIR})
INCLUDE_DIRECTORIES(${MKL_DIR})
LINK_DIRECTORIES(${YAEL_LIB})
LINK_DIRECTORIES(${MKL_LIB})

SET(DEP_LIBS yael mkl_rt gflags)

#########################################################################################
ADD_EXECUTABLE (training_gnoimi training_gnoimi.cpp)
TARGET_LINK_LIBRARIES (training_gnoimi ${DEP_LIBS})

#########################################################################################
ADD_EXECUTABLE (searching_gnoimi searching_gnoimi.cpp)
TARGET_LINK_LIBRARIES (searching_gnoimi ${DEP_LIBS})

