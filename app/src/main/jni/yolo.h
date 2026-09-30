// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLO_H
#define YOLO_H

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include <net.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};
struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};
class Yolo
{
public:
    Yolo();
    ~Yolo();

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu, int _num_class, const char* _input_name, const char* _output_name, bool _multi_output, bool _scores_sigmoided, bool _dfl_builtin, int _pixel_format, const std::vector<const char*>& _class_names);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

private:
    ncnn::Net yolo;
    ncnn::Layer* softmax;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
    int num_class;
    std::string input_name;
    std::string output_name;
    bool multi_output;
    bool scores_sigmoided;
    bool dfl_builtin;
    int pixel_format;
    std::vector<const char*> class_names;
    std::vector<unsigned char> model_buf;
};

#endif // NANODET_H
