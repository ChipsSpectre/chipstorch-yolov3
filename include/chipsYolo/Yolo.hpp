//
// Created by chips on 02.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_MODEL_HPP
#define CHIPSYOLOV3_LIBTORCH_MODEL_HPP

#include <iostream>
#include <torch/torch.h>
#include "ConfigLoader.hpp"
#include "DetectionLayer.hpp"
#include "IdentityLayer.hpp"
#include "MaxPool2D.hpp"
#include "Splitter.hpp"
#include "UpsampleLayer.hpp"

/**
 * Yolo is a class that encapsulates the torch::nn Modules that are needed to run the yolov3 and related configurations.
 *
 * By providing the wrapper class, the usability of the yolo library is enhanced.
 */
class Yolo {
private:

    int get_int_from_cfg(std::map<string, string> block, string key, int default_value) {
        if (block.find(key) != block.end()) {
            return std::stoi(block.at(key));
        }
        return default_value;
    }

    string get_string_from_cfg(std::map<string, string> block, string key, string default_value) {
        if (block.find(key) != block.end()) {
            return block.at(key);
        }
        return default_value;
    }

    torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                          int64_t stride, int64_t padding, int64_t groups, bool with_bias=false){
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
        conv_options.stride_ = stride;
        conv_options.padding_ = padding;
        conv_options.groups_ = groups;
        conv_options.with_bias_ = with_bias;
        return conv_options;
    }

    torch::nn::BatchNormOptions bn_options(int64_t features){
        torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
        bn_options.affine_ = true;
        bn_options.stateful_ = true;
        return bn_options;
    }

    void
    addConvLayer(torch::nn::Sequential& module, string activation, int batch_normalize,
                 int filters, int padding, int kernel_size, int stride, int prev_filters) {
        int pad = padding > 0 ? (kernel_size - 1) / 2 : 0;
        bool with_bias = batch_normalize > 0 ? false : true;

        torch::nn::Conv2d conv = torch::nn::Conv2d(
                conv_options(prev_filters, filters, kernel_size, stride, pad, 1, with_bias));
        module->push_back(conv);

        if (batch_normalize > 0) {
            torch::nn::BatchNorm bn = torch::nn::BatchNorm(bn_options(filters));
            module->push_back(bn);
        }

        if (activation == "leaky") {
            module->push_back(torch::nn::Functional(torch::leaky_relu, /*slope=*/0.1));
        }
    }

/**
         * Loads the model from provided configuration information.
         *
         * @param blocks - configuration object, i.e. a typedef for a vector of maps from string to string.
         * @return sequential model containing all layers
         */
    torch::nn::Sequential loadModel(Config blocks) {
        int prev_filters = 3;

        std::vector<int> output_filters;

        int index = 0;
        int filters = 0;

        torch::nn::Sequential module;

        for (std::size_t i = 0, len = blocks.size(); i < len; i++) {
            std::map<string, string> block = blocks[i];

            string layer_type = block["type"];

            if (layer_type == "net")
                continue;

            if (layer_type == "convolutional") {
                string activation = get_string_from_cfg(block, "activation", "");
                int batch_normalize = get_int_from_cfg(block, "batch_normalize", 0);
                filters = get_int_from_cfg(block, "filters", 0);
                int padding = get_int_from_cfg(block, "pad", 0);
                int kernel_size = get_int_from_cfg(block, "size", 0);
                int stride = get_int_from_cfg(block, "stride", 1);

                addConvLayer(module, activation, batch_normalize, filters, padding, kernel_size, stride, prev_filters);
            }
            else if (layer_type == "upsample")
            {
                int stride = get_int_from_cfg(block, "stride", 1);

                UpsampleLayer uplayer(stride);
                module->push_back(uplayer);
            }
            else if (layer_type == "maxpool")
            {
                int stride = get_int_from_cfg(block, "stride", 1);
                int size = get_int_from_cfg(block, "size", 1);

                MaxPool2D poolLayer(size, stride);
                module->push_back(poolLayer);
            }
            else if (layer_type == "shortcut")
            {
                // skip connection
                int from = get_int_from_cfg(block, "from", 0);
                block["from"] = std::to_string(from);

                blocks[i] = block;

                // placeholder
                IdentityLayer layer;
                module->push_back(layer);
            }
            else if (layer_type == "route")
            {
                // L 85: -1, 61
                string layers_info = get_string_from_cfg(block, "layers", "");

                std::vector<string> layers;
                _splitter.split(layers_info, layers, ",");

                std::string::size_type sz;
                signed int start = std::stoi(layers[0], &sz);
                signed int end = 0;

                if (layers.size() > 1)
                {
                    end = std::stoi(layers[1], &sz);
                }

                if (start > 0)	start = start - index;

                if (end > 0) end = end - index;

                block["start"] = std::to_string(start);
                block["end"] = std::to_string(end);

                blocks[i] = block;

                // placeholder
                IdentityLayer layer;
                module->push_back(layer);

                if (end < 0)
                {
                    filters = output_filters[index + start] + output_filters[index + end];
                }
                else
                {
                    filters = output_filters[index + start];
                }
            }
            else if (layer_type == "yolo")
            {
                string mask_info = get_string_from_cfg(block, "mask", "");
                std::vector<int> masks;
                _splitter.split(mask_info, masks, ",");

                string anchor_info = get_string_from_cfg(block, "anchors", "");
                std::vector<int> anchors;
                _splitter.split(anchor_info, anchors, ",");

                std::vector<float> anchor_points;
                int pos;
                for (int i = 0; i< masks.size(); i++)
                {
                    pos = masks[i];
                    anchor_points.push_back(anchors[pos * 2]);
                    anchor_points.push_back(anchors[pos * 2+1]);
                }

                DetectionLayer layer(anchor_points);
                module->push_back(layer);
            }
            else
            {
                std::string errorMsg = "Unsupported operator: " + layer_type;
                throw std::runtime_error(errorMsg);
            }

            prev_filters = filters;
            output_filters.push_back(filters);

            char *module_key = new char[strlen("layer_") + sizeof(index) + 1];

            sprintf(module_key, "%s%d", "layer_", index);

            index += 1;
        }
        return module;
    }

public:
    Yolo(const std::string &cfgFile, const torch::Device &device)
            : _device(device),
              _splitter(),
              _configLoader(),
              _model(loadModel(_configLoader.loadFromConfig(cfgFile))) {
    }

    /**
     * Yields the number of layers.
     * @return number of layers, counting each activation function/batchnorm/cnn layer separately.
     */
    std::size_t size() {
        return _model->size();
    }

private:
    const torch::Device &_device;
    Splitter _splitter;
    ConfigLoader _configLoader;
    const torch::nn::Sequential _model;
};

#endif //CHIPSYOLOV3_LIBTORCH_MODEL_HPP
