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

    std::string configPath = "../cfg/yolov3.cfg";
    Yolo yolo(configPath, device, 416);

    yolo.loadWeights("../models/yolov3.weights");
    std::cout << "loaded weights." << std::endl;

    int i = 0;
    for(auto& layer : yolo.model()->modules()) {
        std::cout << i << "\t" << layer->name() << std::endl;
        i++;

    }

    std::cout << yolo.size() << std::endl;

    yolo.toDevice();

    torch::NoGradGuard no_grad;
    yolo.eval();

    yolo.infer("/home/chips/Schreibtisch/person.jpg");
    yolo.infer("/home/chips/Schreibtisch/person.jpg");
    yolo.infer("/home/chips/Schreibtisch/person.jpg");
    yolo.infer("/home/chips/Schreibtisch/person.jpg");
    yolo.infer("/home/chips/Schreibtisch/person.jpg");

    return 0;
}
