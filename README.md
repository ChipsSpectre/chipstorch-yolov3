# Yolov3 in Libtorch and C++

Repository is based on https://github.com/walktree/libtorch-yolov3 with the following improvments:

- header-only implementation (just include yolo in your own projects)
- code redesign (separated classes into different files to aid adaptability)
- avoidance of raw pointers where possible
- Installation of cpu-based pytorch c++ frontend if no installtion is found

## Requirements

- OpenCV

- CUDA (optional, recommended for increased performance)

## Installation

If you have pytorch c++ frontend installed:

    mkdir build
    cd build
    
    cmake -DCMAKE_PREFIX_PATH=<path/to/libtorch> -DCMAKE_BUILD_PATH=Release ../
    make -j4
    
If you do not have installed the pytorch c++ frontend:

    mkdir build
    cd build
    
    cmake -DCMAKE_BUILD_TYPE=Release ../
    make -j4
    
 ## Contributions / suggestions welcome!
 