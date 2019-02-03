//
// Created by chips on 03.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_IOHANDLER_HPP
#define CHIPSYOLOV3_LIBTORCH_IOHANDLER_HPP

#include <torch/torch.h>

/**
 * The IOHandler is used to handle disk operations.
 */
class IOHandler {
public:
    static
    at::Tensor readTensorFromFile(std::string fileName) {
        std::ifstream fs(fileName, std::ios::binary);

        // header info: 5 * int32_t
        int32_t header_size = sizeof(int32_t) * 5;

        fs.seekg(0, fs.end);
        int64_t length = fs.tellg();
        // skip header
        length = length - header_size;

        fs.seekg(header_size, fs.beg);
        float *weights_src = (float *) malloc(length);
        fs.read(reinterpret_cast<char *>(weights_src), length);

        fs.close();

        at::TensorOptions options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .is_variable(true);
        at::Tensor weights = torch::CPU(torch::kFloat32).tensorFromBlob(weights_src, {length / 4});
        return weights;
    }
};

#endif //CHIPSYOLOV3_LIBTORCH_IOHANDLER_HPP
