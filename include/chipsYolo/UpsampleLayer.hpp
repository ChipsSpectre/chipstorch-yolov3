//
// Created by chips on 03.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_UPSAMPLELAYER_HPP
#define CHIPSYOLOV3_LIBTORCH_UPSAMPLELAYER_HPP
#include <torch/torch.h>

/**
 * Implementation of an upsample layer that supports 3D and 4D input vectors at once.
 *
 * Uses 1D and 2D upsampling from pytorch internally.
 */
class UpsampleLayerImpl : public torch::nn::Module
{
private:
    int _stride;
public:
    UpsampleLayerImpl(int stride){
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

TORCH_MODULE(UpsampleLayer);

#endif //CHIPSYOLOV3_LIBTORCH_UPSAMPLELAYER_HPP
