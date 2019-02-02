#include <iostream>
#include <chipsYolo/Yolo.hpp>
#include <chipsYolo/ConfigLoader.hpp>

int main() {
    std::cout << "Hello, World!" << std::endl;

    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    Yolo yolo("../cfg/yolov3.cfg", device);

    return 0;
}
