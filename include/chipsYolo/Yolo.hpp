//
// Created by chips on 02.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_MODEL_HPP
#define CHIPSYOLOV3_LIBTORCH_MODEL_HPP

#include <iostream>
#include <torch/torch.h>
#include <torch/torch.h>

/**
 * Yolo is a class that encapsulates the torch::nn Modules that are needed to run the yolov3 and related configurations.
 *
 * By providing the wrapper class, the usability of the yolo library is enhanced.
 */
class Yolo {
public:
    Yolo(const std::string& cfgFile, const torch::Device& device)
        : _device(device) {
    }
private:
    const torch::Device& _device;
};

#endif //CHIPSYOLOV3_LIBTORCH_MODEL_HPP
