//
// Created by chips on 02.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_MODEL_HPP
#define CHIPSYOLOV3_LIBTORCH_MODEL_HPP

#include <iostream>
#include <torch/torch.h>

struct UpsampleLayer : torch::nn::Module
{
    int _stride;
    UpsampleLayer(int stride){
        _stride = stride;
    }

    torch::Tensor forward(torch::Tensor x) {

        torch::IntList sizes = x.sizes();

        int64_t w, h;

        if (sizes.size() == 4)
        {
            w = sizes[2] * _stride;
            h = sizes[3] * _stride;

            x = torch::upsample_nearest2d(x, {w, h});
        }
        else if (sizes.size() == 3)
        {
            w = sizes[2] * _stride;
            x = torch::upsample_nearest1d(x, {w});
        }
        return x;
    }
};

#endif //CHIPSYOLOV3_LIBTORCH_MODEL_HPP
