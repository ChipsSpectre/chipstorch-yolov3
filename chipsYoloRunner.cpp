#include <iostream>
#include <chipsYolo/Yolo.hpp>
#include <chipsYolo/ConfigLoader.hpp>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./chipsYoloRunner <image path>" << std::endl;
        return -1;
    }

    torch::DeviceType device_type;

    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    std::string configPath = "../cfg/yolov3-tiny.cfg";
    Yolo yolo(configPath, device, 416);


    int i = 0;
    for(auto& layer : yolo.model()->modules()) {
        std::cout << i << "\t" << layer->name() << std::endl;
        i++;
    }

    yolo.loadWeights("../models/yolov3-tiny.weights");
    std::cout << "loaded weights." << std::endl;

    std::cout << yolo.size() << std::endl;

    yolo.toDevice();

    torch::NoGradGuard no_grad;
    yolo.eval();

    yolo.infer(argv[1], "out.jpg");

    return 0;
}
