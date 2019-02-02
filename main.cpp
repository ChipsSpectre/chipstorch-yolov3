#include <iostream>
#include <chipsYolo/Model.hpp>
#include <chipsYolo/ConfigLoader.hpp>

int main() {
    std::cout << "Hello, World!" << std::endl;

    ConfigLoader configLoader;

    configLoader.loadFromConfig("../cfg/yolov3.cfg");

    int x = 5;
    return 0;
}
