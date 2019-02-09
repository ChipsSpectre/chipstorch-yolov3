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
    
## Usage

- Cifar10 Sample

The first libtorch implementation of a Cifar10 data loader is used in this program to predict 
the labels of cifar10 images. Download the data using ***download_cifar10.py*** first.

    cd build
    ./cifar10 
    
For visualization run the following command:

    python3 view_cifar10.py build/
    
The network used is really simple because this an example repository - feel free to replace 
the class **Net** in ***cifar10.cpp*** with your more sophisticated model to eliminate errors.

- Yolov3 sample

In order to run yolov3 inference on an image of your choice, execute the following command.

    cd build
    ./yolov3 <path_to_your_image>
    
The result will be stored as the image **out.png** in the build folder.
    
 
 ## Contributions / suggestions welcome!
 