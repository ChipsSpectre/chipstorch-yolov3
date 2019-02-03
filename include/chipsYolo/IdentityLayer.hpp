//
// Created by chips on 03.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_IDENTITYLAYER_HPP
#define CHIPSYOLOV3_LIBTORCH_IDENTITYLAYER_HPP

#include <torch/torch.h>

class IdentityLayerImpl : public torch::nn::Module
{
public:
    IdentityLayerImpl(){

    }

    torch::Tensor forward(torch::Tensor x) {
        return x;
    }
};

TORCH_MODULE(IdentityLayer);

#endif //CHIPSYOLOV3_LIBTORCH_IDENTITYLAYER_HPP
