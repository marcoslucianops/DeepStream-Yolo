cmake_minimum_required(VERSION 3.5.1)

PROJECT(cmake_and_cuda 
    LANGUAGES CXX
)

### LIBAPC All Source Files ###
FILE(GLOB CXX_SOURCES_FILES CONFIGURE_DEPENDS
    nvdsinfer_custom_impl_Yolo/*.cpp
    nvdsinfer_custom_impl_Yolo/layers/*.cpp
)

FILE(GLOB CU_SOURCES_FILES CONFIGURE_DEPENDS
    nvdsinfer_custom_impl_Yolo/*.cu
)


### INCLUDE DIRECTORY ###
INCLUDE_DIRECTORIES(
    nvdsinfer_custom_impl_Yolo
    nvdsinfer_custom_impl_Yolo/layers
    ${CUDA_PATH}/include
    ${INCLUDE_PATH}/opencv4
)


### Make STATIC library (.so) ###
ADD_LIBRARY(nvdsinfer_custom_impl_Yolo SHARED ${CXX_SOURCES_FILES} ${CU_SOURCES_FILES})


FIND_LIBRARY( CUDART 
	NAMES cudart
	PATHS ${SDK_PATH}/usr/local/cuda-10.2/lib/
)


### Link Library ###
TARGET_LINK_LIBRARIES(nvdsinfer_custom_impl_Yolo
    nvinfer
    nvinfer_plugin
    nvparsers
	${CUDART}
)

