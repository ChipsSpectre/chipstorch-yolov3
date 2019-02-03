#include <iostream>
#include <chipsYolo/Yolo.hpp>
#include <chipsYolo/ConfigLoader.hpp>

int main() {
    std::cout << "Hello, World!" << std::endl;

    torch::DeviceType device_type;

    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    std::string configPath = "../cfg/yolov3-tiny.cfg";
    Yolo yolo(configPath, device);

    yolo.loadWeights("../models/yolov3-tiny.weights");

    for(auto& layer : yolo.model()->modules()) {
        std::cout << layer->name() << std::endl;
    }

    std::cout << yolo.size() << std::endl;
    return 0;
}
