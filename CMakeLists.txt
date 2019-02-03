cmake_minimum_required(VERSION 3.10)
project(chipsYolov3_libtorch)

set(CMAKE_CXX_STANDARD 14)

set(LIBRARY_NAME "chipsYolo")
set(EXECUTABLE_NAME "chipsYoloRunner")

# include openCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

message(${OpenCV_INCLUDE_DIRS})

# get pytorch, if not already available
message("${CMAKE_CURRENT_SOURCE_DIR}")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch/build-version")
    message("pytorch already installed.")
else()
    execute_process(COMMAND wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip)
    message("downloaded pytorch lib.")
    execute_process(COMMAND unzip libtorch-shared-with-deps-latest.zip -d ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
    message("unpacked pytorch lib zip-file.")
endif()

# include pytorch using the cmake prefix path
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/libtorch")
find_package(Torch REQUIRED)

# adds library as interface (requires cmake >= 0
add_library(${LIBRARY_NAME} INTERFACE)
# includes the header libraries
target_include_directories(${LIBRARY_NAME} INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )
# installs the library correctly
install(TARGETS ${LIBRARY_NAME} EXPORT ${LIBRARY_NAME}
        DESTINATION lib) # ignored
install(DIRECTORY include/${LIBRARY_NAME} DESTINATION include)

include_directories(./include)

add_executable(${EXECUTABLE_NAME} main.cpp include/chipsYolo/Yolo.hpp)
message(${OpenCV_LIBS})
target_link_libraries(${EXECUTABLE_NAME} chipsYolo ${TORCH_LIBRARIES} ${OpenCV_LIBS})
