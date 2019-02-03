//
// Created by chips on 03.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_DETECTIONLAYER_HPP
#define CHIPSYOLOV3_LIBTORCH_DETECTIONLAYER_HPP

#include <torch/torch.h>

class DetectionLayerImpl : public torch::nn::Module {
private:
    std::vector<float> _anchors;
public:
    DetectionLayerImpl(std::vector<float> anchors) {
        _anchors = anchors;
    }

    torch::Tensor forward(torch::Tensor prediction, int inp_dim, int num_classes, torch::Device device) {
        return predict_transform(prediction, inp_dim, _anchors, num_classes, device);
    }

    torch::Tensor predict_transform(torch::Tensor prediction, int inp_dim, std::vector<float> anchors, int num_classes,
                                    torch::Device device) {
        int batch_size = prediction.size(0);
        int stride = floor(inp_dim / prediction.size(2));
        int grid_size = floor(inp_dim / stride);
        int bbox_attrs = 5 + num_classes;
        int num_anchors = anchors.size() / 2;

        for (int i = 0; i < anchors.size(); i++) {
            anchors[i] = anchors[i] / stride;
        }
        torch::Tensor result = prediction.view({batch_size, bbox_attrs * num_anchors, grid_size * grid_size});
        result = result.transpose(1, 2).contiguous();
        result = result.view({batch_size, grid_size * grid_size * num_anchors, bbox_attrs});

        result.select(2, 0).sigmoid_();
        result.select(2, 1).sigmoid_();
        result.select(2, 4).sigmoid_();

        auto grid_len = torch::arange(grid_size);

        std::vector <torch::Tensor> args = torch::meshgrid({grid_len, grid_len});

        torch::Tensor x_offset = args[1].contiguous().view({-1, 1});
        torch::Tensor y_offset = args[0].contiguous().view({-1, 1});

        x_offset = x_offset.to(device);
        y_offset = y_offset.to(device);

        auto x_y_offset = torch::cat({x_offset, y_offset}, 1).repeat({1, num_anchors}).view({-1, 2}).unsqueeze(0);
        result.slice(2, 0, 2).add_(x_y_offset);

        torch::Tensor anchors_tensor = torch::from_blob(anchors.data(), {num_anchors, 2});
        //if (device != nullptr)
        anchors_tensor = anchors_tensor.to(device);
        anchors_tensor = anchors_tensor.repeat({grid_size * grid_size, 1}).unsqueeze(0);

        result.slice(2, 2, 4).exp_().mul_(anchors_tensor);
        result.slice(2, 5, 5 + num_classes).sigmoid_();
        result.slice(2, 0, 4).mul_(stride);

        return result;
    }
};
TORCH_MODULE(DetectionLayer);

#endif //CHIPSYOLOV3_LIBTORCH_DETECTIONLAYER_HPP
