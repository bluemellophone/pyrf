cmake_minimum_required(VERSION 2.8)
SET(CLANG2 Off)

#######################################
# Clang2 = Clang + OpenMP built for OSX Mavericks
# http://goo.gl/1Tg0Zj
if (APPLE)
    message("Detected APPLE system")
    SET(CLANG2 Off)
endif()

if (APPLE AND CLANG2)
	message("Using clang2")
    set(CMAKE_C_COMPILER "clang2")
    set(CMAKE_CXX_COMPILER "clang2++")
endif()

#######################################
project(pyrf) #### !!!!IMPORTANT!!!! THIS MUST BE DOWN HERE FOR CLANG2
set(CMAKE_BUILD_TYPE "Release")

if (APPLE)
    #MacPorts
    message("USING MACPORTS")
    include_directories(/opt/local/include)
    link_directories(/opt/local/lib)
endif()

#######################################
message(STATUS "Checking OpenCV")
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

#######################################
if(APPLE AND CLANG2)    
    # Add flags to support clang2
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -stdlib=libstdc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -stdlib=libstdc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lstdc++ -stdlib=libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++ -stdlib=libstdc++")
endif()


#######################################
message(STATUS "Checking OpenMP")
find_package(OpenMP)
IF(OPENMP_FOUND)
    message("Found OpenMP! ^_^")
    # add flags for OpenMP
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_SHARED_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    message("Missed OpenMP! x_x")
ENDIF()

#######################################
set(SOURCE_FILES 
    src/CRForestDetector.cpp
    src/CRPatch.cpp
    src/CRTree.cpp
    src/HoG.cpp)

add_library(pyrf SHARED 
	src/pyrf.cpp
	${SOURCE_FILES})

#######################################
target_link_libraries(pyrf ${OpenCV_LIBS})
