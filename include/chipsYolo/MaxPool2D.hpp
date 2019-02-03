//
// Created by chips on 03.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_MAXPOOL2D_HPP
#define CHIPSYOLOV3_LIBTORCH_MAXPOOL2D_HPP

#include <torch/torch.h>

class MaxPool2DImpl : public torch::nn::Module {
private:
    int _kernel_size;
    int _stride;
public:
    MaxPool2DImpl(int kernel_size, int stride) {
        _kernel_size = kernel_size;
        _stride = stride;
    }

    torch::Tensor forward(torch::Tensor x) {
        if (_stride != 1) {
            x = torch::max_pool2d(x, {_kernel_size, _kernel_size}, {_stride, _stride});
        } else {
            int pad = _kernel_size - 1;

            torch::Tensor padded_x = torch::replication_pad2d(x, {0, pad, 0, pad});
            x = torch::max_pool2d(padded_x, {_kernel_size, _kernel_size}, {_stride, _stride});
        }

        return x;
    }
};

TORCH_MODULE(MaxPool2D);

#endif //CHIPSYOLOV3_LIBTORCH_MAXPOOL2D_HPP
