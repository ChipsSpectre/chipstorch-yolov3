//
// Created by chips on 02.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_MODEL_HPP
#define CHIPSYOLOV3_LIBTORCH_MODEL_HPP

#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include "ConfigLoader.hpp"
#include "DetectionLayer.hpp"
#include "Drawer.hpp"
#include "IdentityLayer.hpp"
#include "MaxPool2D.hpp"
#include "Splitter.hpp"
#include "UpsampleLayer.hpp"

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
                                          int64_t stride, int64_t padding, int64_t groups, bool with_bias = false) {
        torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
        conv_options.stride_ = stride;
        conv_options.padding_ = padding;
        conv_options.groups_ = groups;
        conv_options.with_bias_ = with_bias;
        return conv_options;
    }

    torch::nn::BatchNormOptions bn_options(int64_t features) {
        torch::nn::BatchNormOptions bn_options = torch::nn::BatchNormOptions(features);
        bn_options.affine_ = true;
        bn_options.stateful_ = true;
        return bn_options;
    }

    void
    addConvLayer(torch::nn::Sequential &module, string activation, int batch_normalize,
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

            if (layer_type == "net") {
                _input_dim = std::stoi(block["width"]);
                continue;
            }

            if (layer_type == "convolutional") {
                string activation = get_string_from_cfg(block, "activation", "");
                int batch_normalize = get_int_from_cfg(block, "batch_normalize", 0);
                filters = get_int_from_cfg(block, "filters", 0);
                int padding = get_int_from_cfg(block, "pad", 0);
                int kernel_size = get_int_from_cfg(block, "size", 0);
                int stride = get_int_from_cfg(block, "stride", 1);

                addConvLayer(module, activation, batch_normalize, filters, padding, kernel_size, stride, prev_filters);
            } else if (layer_type == "upsample") {
                int stride = get_int_from_cfg(block, "stride", 1);

                UpsampleLayer uplayer(stride);
                module->push_back(uplayer);
            } else if (layer_type == "maxpool") {
                int stride = get_int_from_cfg(block, "stride", 1);
                int size = get_int_from_cfg(block, "size", 1);

                MaxPool2D poolLayer(size, stride);
                module->push_back(poolLayer);
            } else if (layer_type == "shortcut") {
                // skip connection
                int from = get_int_from_cfg(block, "from", 0);
                block["from"] = std::to_string(from);

                blocks[i] = block;

                // placeholder
                IdentityLayer layer;
                module->push_back(layer);
            } else if (layer_type == "route") {
                // L 85: -1, 61
                string layers_info = get_string_from_cfg(block, "layers", "");

                std::vector<string> layers;
                _splitter.split(layers_info, layers, ",");

                std::string::size_type sz;
                signed int start = std::stoi(layers[0], &sz);
                signed int end = 0;

                if (layers.size() > 1) {
                    end = std::stoi(layers[1], &sz);
                }

                if (start > 0) start = start - index;

                if (end > 0) end = end - index;

                block["start"] = std::to_string(start);
                block["end"] = std::to_string(end);

                blocks[i] = block;

                // placeholder
                IdentityLayer layer;
                module->push_back(layer);

                if (end < 0) {
                    filters = output_filters[index + start] + output_filters[index + end];
                } else {
                    filters = output_filters[index + start];
                }
            } else if (layer_type == "yolo") {
                string mask_info = get_string_from_cfg(block, "mask", "");
                std::vector<int> masks;
                _splitter.split(mask_info, masks, ",");

                string anchor_info = get_string_from_cfg(block, "anchors", "");
                std::vector<int> anchors;
                _splitter.split(anchor_info, anchors, ",");

                std::vector<float> anchor_points;
                int pos;
                for (int i = 0; i < masks.size(); i++) {
                    pos = masks[i];
                    anchor_points.push_back(anchors[pos * 2]);
                    anchor_points.push_back(anchors[pos * 2 + 1]);
                }

                int numClasses = get_int_from_cfg(block, "classes", 80);

                DetectionLayer layer(anchor_points, _input_dim, numClasses, _device);
                module->push_back(layer);
            } else {
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
            : _configPath(cfgFile),
              _device(device),
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

    void loadWeights(const std::string &fileName) {
        std::ifstream fs(fileName, std::ios::binary);

        // header info: 5 * int32_t
        int32_t header_size = sizeof(int32_t) * 5;

        int64_t index_weight = 0;

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

        Config config = _configLoader.loadConvFromConfig(_configPath);
        int configPos = 0; // points to the position of the current conv layer

        for (size_t i = 0; i < size(); i++) {
            auto layer = _model.ptr()->ptr(i);
            if (layer->name() != "torch::nn::Conv2dImpl") {
                continue; // only load weights for conv-layers!
            }
            torch::nn::Conv2dImpl *conv_imp = dynamic_cast<torch::nn::Conv2dImpl *>(layer.get());
            std::map<string, string> module_info = config[configPos];

            int batch_normalize = get_int_from_cfg(module_info, "batch_normalize", 0);
            if (batch_normalize > 0) {
                auto bn_module = _model.ptr()->ptr(i + 1);

                torch::nn::BatchNormImpl *bn_imp = dynamic_cast<torch::nn::BatchNormImpl *>(bn_module.get());

                int num_bn_biases = bn_imp->bias.numel();

                at::Tensor bn_bias = weights.slice(0, index_weight, index_weight + num_bn_biases);
                index_weight += num_bn_biases;

                at::Tensor bn_weights = weights.slice(0, index_weight, index_weight + num_bn_biases);
                index_weight += num_bn_biases;

                at::Tensor bn_running_mean = weights.slice(0, index_weight, index_weight + num_bn_biases);
                index_weight += num_bn_biases;

                at::Tensor bn_running_var = weights.slice(0, index_weight, index_weight + num_bn_biases);
                index_weight += num_bn_biases;

                bn_bias = bn_bias.view_as(bn_imp->bias);
                bn_weights = bn_weights.view_as(bn_imp->weight);
                bn_running_mean = bn_running_mean.view_as(bn_imp->running_mean);
                bn_running_var = bn_running_var.view_as(bn_imp->running_variance);

                bn_imp->bias.set_data(bn_bias);
                bn_imp->weight.set_data(bn_weights);
                bn_imp->running_mean.set_data(bn_running_mean);
                bn_imp->running_variance.set_data(bn_running_var);
            }else {
                int num_conv_biases = conv_imp->bias.numel();

                at::Tensor conv_bias = weights.slice(0, index_weight, index_weight + num_conv_biases);
                index_weight += num_conv_biases;

                conv_bias = conv_bias.view_as(conv_imp->bias);
                conv_imp->bias.set_data(conv_bias);
            }

            int num_weights = conv_imp->weight.numel();

            at::Tensor conv_weights = weights.slice(0, index_weight, index_weight + num_weights);
            index_weight += num_weights;

            conv_weights = conv_weights.view_as(conv_imp->weight);
            conv_imp->weight.set_data(conv_weights);
            configPos++;
        }
    }

    torch::nn::Sequential &model() {
        return _model;
    }

    void toDevice() {
        _model->to(_device);
    }

    void eval() {
        _model->eval();
    }

    void infer(const std::string imgFile) {
        cv::Mat origin_image, resized_image;

        origin_image = cv::imread(imgFile);

        // input image size for YOLO v3
        int input_image_size = 416;
        cv::cvtColor(origin_image, resized_image,  cv::COLOR_RGB2BGR);
        cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

        cv::Mat img_float;
        resized_image.convertTo(img_float, CV_32F, 1.0/255);

        auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, input_image_size, input_image_size, 3});
        img_tensor = img_tensor.permute({0,3,1,2});
        auto img_var = torch::autograd::make_variable(img_tensor, false).to(_device);

        auto start = std::chrono::high_resolution_clock::now();

        auto output = _model->forward(img_var);

        // filter result by NMS
        // class_num = 80
        // confidence = 0.6
        auto result = _drawer.write_results(output, 80, 0.6, 0.4);

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // It should be known that it takes longer time at first time
        std::cout << "inference taken : " << duration.count() << " ms" << std::endl;

        if (result.dim() == 1)
        {
            std::cout << "no object found" << std::endl;
        }
        else {
            int obj_num = result.size(0);

            std::cout << obj_num << " objects found" << std::endl;

            float w_scale = float(origin_image.cols) / input_image_size;
            float h_scale = float(origin_image.rows) / input_image_size;

            result.select(1, 1).mul_(w_scale);
            result.select(1, 2).mul_(h_scale);
            result.select(1, 3).mul_(w_scale);
            result.select(1, 4).mul_(h_scale);

            auto result_data = result.accessor<float, 2>();

            for (int i = 0; i < result.size(0); i++) {
                cv::rectangle(origin_image, cv::Point(result_data[i][1], result_data[i][2]),
                              cv::Point(result_data[i][3], result_data[i][4]), cv::Scalar(0, 0, 255), 1, 1, 0);
            }

            cv::imwrite("out-det.jpg", origin_image);
        }
    }
private:
    const std::string &_configPath;
    const torch::Device &_device;
    Splitter _splitter;
    ConfigLoader _configLoader;
    Drawer _drawer;
    torch::nn::Sequential _model;
    int _input_dim;
};

#endif //CHIPSYOLOV3_LIBTORCH_MODEL_HPP
