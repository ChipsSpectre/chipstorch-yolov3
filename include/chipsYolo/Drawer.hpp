//
// Created by chips on 03.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_DRAWER_HPP
#define CHIPSYOLOV3_LIBTORCH_DRAWER_HPP

#include <torch/torch.h>

class Drawer {
private:
    // returns the IoU of two bounding boxes
    inline torch::Tensor get_bbox_iou(torch::Tensor box1, torch::Tensor box2)
    {
        // Get the coordinates of bounding boxes
        torch::Tensor b1_x1, b1_y1, b1_x2, b1_y2;
        b1_x1 = box1.select(1, 0);
        b1_y1 = box1.select(1, 1);
        b1_x2 = box1.select(1, 2);
        b1_y2 = box1.select(1, 3);
        torch::Tensor b2_x1, b2_y1, b2_x2, b2_y2;
        b2_x1 = box2.select(1, 0);
        b2_y1 = box2.select(1, 1);
        b2_x2 = box2.select(1, 2);
        b2_y2 = box2.select(1, 3);

        // et the corrdinates of the intersection rectangle
        torch::Tensor inter_rect_x1 =  torch::max(b1_x1, b2_x1);
        torch::Tensor inter_rect_y1 =  torch::max(b1_y1, b2_y1);
        torch::Tensor inter_rect_x2 =  torch::min(b1_x2, b2_x2);
        torch::Tensor inter_rect_y2 =  torch::min(b1_y2, b2_y2);

        // Intersection area
        torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1 + 1,torch::zeros(inter_rect_x2.sizes()))*torch::max(inter_rect_y2 - inter_rect_y1 + 1, torch::zeros(inter_rect_x2.sizes()));

        // Union Area
        torch::Tensor b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1);
        torch::Tensor b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1);

        torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);

        return iou;
    }

public:
    torch::Tensor write_results(torch::Tensor prediction, int num_classes, float confidence, float nms_conf)
    {
        // get result which object confidence > threshold
        auto conf_mask = (prediction.select(2,4) > confidence).to(torch::kFloat32).unsqueeze(2);

        prediction.mul_(conf_mask);
        auto ind_nz = torch::nonzero(prediction.select(2, 4)).transpose(0, 1).contiguous();

        if (ind_nz.size(0) == 0)
        {
            return torch::zeros({0});
        }

        torch::Tensor box_a = torch::ones(prediction.sizes(), prediction.options());
        // top left x = centerX - w/2
        box_a.select(2, 0) = prediction.select(2, 0) - prediction.select(2, 2).div(2);
        box_a.select(2, 1) = prediction.select(2, 1) - prediction.select(2, 3).div(2);
        box_a.select(2, 2) = prediction.select(2, 0) + prediction.select(2, 2).div(2);
        box_a.select(2, 3) = prediction.select(2, 1) + prediction.select(2, 3).div(2);

        prediction.slice(2, 0, 4) = box_a.slice(2, 0, 4);

        int batch_size = prediction.size(0);
        int item_attr_size = 5;

        torch::Tensor output = torch::ones({1, prediction.size(2) + 1});
        bool write = false;

        int num = 0;

        for (int i = 0; i < batch_size; i++)
        {
            auto image_prediction = prediction[i];

            // get the max classes score at each result
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(image_prediction.slice(1, item_attr_size, item_attr_size + num_classes), 1);

            // class score
            auto max_conf = std::get<0>(max_classes);
            // index
            auto max_conf_score = std::get<1>(max_classes);
            max_conf = max_conf.to(torch::kFloat32).unsqueeze(1);
            max_conf_score = max_conf_score.to(torch::kFloat32).unsqueeze(1);

            // shape: n * 7, left x, left y, right x, right y, object confidence, class_score, class_id
            image_prediction = torch::cat({image_prediction.slice(1, 0, 5), max_conf, max_conf_score}, 1);

            // remove item which object confidence == 0
            auto non_zero_index =  torch::nonzero(image_prediction.select(1,4));
            auto image_prediction_data = image_prediction.index_select(0, non_zero_index.squeeze()).view({-1, 7});

            // get unique classes
            std::vector<torch::Tensor> img_classes;

            for (int m = 0, len = image_prediction_data.size(0); m < len; m++)
            {
                bool found = false;
                for (int n = 0; n < img_classes.size(); n++)
                {
                    auto ret = (image_prediction_data[m][6] == img_classes[n]);
                    if (torch::nonzero(ret).size(0) > 0)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found) img_classes.push_back(image_prediction_data[m][6]);
            }

            for (int k = 0; k < img_classes.size(); k++)
            {
                auto cls = img_classes[k];

                auto cls_mask = image_prediction_data * (image_prediction_data.select(1, 6) == cls).to(torch::kFloat32).unsqueeze(1);
                auto class_mask_index =  torch::nonzero(cls_mask.select(1, 5)).squeeze();

                auto image_pred_class = image_prediction_data.index_select(0, class_mask_index).view({-1,7});
                // ascend by confidence
                // seems that inverse method not work
                std::tuple<torch::Tensor,torch::Tensor> sort_ret = torch::sort(image_pred_class.select(1,4));

                auto conf_sort_index = std::get<1>(sort_ret);

                // seems that there is something wrong with inverse method
                // conf_sort_index = conf_sort_index.inverse();

                image_pred_class = image_pred_class.index_select(0, conf_sort_index.squeeze()).cpu();

                for(int w = 0; w < image_pred_class.size(0)-1; w++)
                {
                    int mi = image_pred_class.size(0) - 1 - w;

                    if (mi <= 0)
                    {
                        break;
                    }

                    auto ious = get_bbox_iou(image_pred_class[mi].unsqueeze(0), image_pred_class.slice(0, 0, mi));

                    auto iou_mask = (ious < nms_conf).to(torch::kFloat32).unsqueeze(1);
                    image_pred_class.slice(0, 0, mi) = image_pred_class.slice(0, 0, mi) * iou_mask;

                    // remove from list
                    auto non_zero_index = torch::nonzero(image_pred_class.select(1,4)).squeeze();
                    image_pred_class = image_pred_class.index_select(0, non_zero_index).view({-1,7});
                }

                torch::Tensor batch_index = torch::ones({image_pred_class.size(0), 1}).fill_(i);

                if (!write)
                {
                    output = torch::cat({batch_index, image_pred_class}, 1);
                    write = true;
                }
                else
                {
                    auto out = torch::cat({batch_index, image_pred_class}, 1);
                    output = torch::cat({output,out}, 0);
                }

                num += 1;
            }
        }

        if (num == 0)
        {
            return torch::zeros({0});
        }

        return output;
    }
};

#endif //CHIPSYOLOV3_LIBTORCH_DRAWER_HPP
