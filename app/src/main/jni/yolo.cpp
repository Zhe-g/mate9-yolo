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

#include "yolo.h"

#include <cfloat>
#include <cstdio>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <android/log.h>

#include "cpu.h"

static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}
static float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}
static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects, ncnn::Layer* softmax, int num_class, bool scores_sigmoided, bool dfl_builtin)
{
    if (dfl_builtin)
    {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "dfl: pred c=%d h=%d w=%d num_class=%d", pred.c, pred.h, pred.w, num_class);
        // The model produces dims=2 output: h=13, w=num_points (or dims=3: c=13, h=1, w=num_points)
        //   rows/channels 0-3: cx, cy, w, h (pixel coords)
        //   rows/channels 4-12: class scores (already sigmoided)
        int num_points;
        int feat_dim;
        const float* pred_data;
        if (pred.c == 1 && pred.h > 1)
        {
            // dims=2 format: h=13, w=num_points
            num_points = pred.w;
            feat_dim = pred.h;
            pred_data = pred;
        }
        else
        {
            // dims=3 format: c=13, h=1, w=num_points
            num_points = pred.w;
            feat_dim = pred.c;
            pred_data = pred.channel(0);
        }

        if (feat_dim < 4 + num_class)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "dfl: ERROR feat_dim=%d < 4+num_class=%d", feat_dim, 4 + num_class);
            return;
        }

        float max_conf = -FLT_MAX, min_conf = FLT_MAX;
        for (int i = 0; i < num_points; i++)
        {
            float pb_cx = pred_data[0 * num_points + i];
            float pb_cy = pred_data[1 * num_points + i];
            float pb_w = pred_data[2 * num_points + i];
            float pb_h = pred_data[3 * num_points + i];

            int label = -1;
            float max_score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                float confidence = pred_data[(4 + k) * num_points + i];
                if (confidence > max_score)
                {
                    label = k;
                    max_score = confidence;
                }
            }
            if (max_score > max_conf) max_conf = max_score;
            if (max_score < min_conf) min_conf = max_score;

            if (max_score >= prob_threshold)
            {
                Object obj;
                obj.rect.x = pb_cx - pb_w / 2;
                obj.rect.y = pb_cy - pb_h / 2;
                obj.rect.width = pb_w;
                obj.rect.height = pb_h;
                obj.label = label;
                obj.prob = max_score;
                objects.push_back(obj);
            }
        }
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "dfl: min_conf=%f max_conf=%f proposals=%zu threshold=%f", min_conf, max_conf, objects.size(), prob_threshold);
    }
    else
    {
        const int num_points = grid_strides.size();
        for (int i = 0; i < num_points; i++)
        {
            float x0, y0, x1, y1;
            float box_prob;

            const int reg_max_1 = 16;
            const float* scores = pred.row(i) + 4 * reg_max_1;

            int label = -1;
            float score = -FLT_MAX;
            for (int k = 0; k < num_class; k++)
            {
                float confidence = scores[k];
                if (confidence > score)
                {
                    label = k;
                    score = confidence;
                }
            }
            box_prob = scores_sigmoided ? score : sigmoid(score);

            if (box_prob >= prob_threshold)
            {
                ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->forward_inplace(bbox_pred, opt);

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = 0.f;
                    const float* dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++)
                    {
                        dis += l * dis_after_sm[l];
                    }
                    pred_ltrb[k] = dis * grid_strides[i].stride;
                }

                float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
                float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

                x0 = pb_cx - pred_ltrb[0];
                y0 = pb_cy - pred_ltrb[1];
                x1 = pb_cx + pred_ltrb[2];
                y1 = pb_cy + pred_ltrb[3];

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = label;
                obj.prob = box_prob;

                objects.push_back(obj);
            }
        }
    }
}

Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
    softmax = 0;
}

Yolo::~Yolo()
{
    if (softmax)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        softmax->destroy_pipeline(opt);
        delete softmax;
        softmax = 0;
    }
}


int Yolo::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu, int _num_class, const char* _input_name, const char* _output_name, bool _multi_output, bool _scores_sigmoided, bool _dfl_builtin, int _pixel_format, const std::vector<const char*>& _class_names)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    snprintf(parampath, sizeof(parampath), "%s.param", modeltype);
    snprintf(modelpath, sizeof(modelpath), "%s.bin", modeltype);

    int ret;

    // Load param from manually read asset
    {
        AAsset* asset = AAssetManager_open(mgr, parampath, AASSET_MODE_BUFFER);
        if (!asset)
        {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "cannot open asset %s", parampath);
            return -1;
        }
        size_t len = AAsset_getLength(asset);
        const char* buf = (const char*)AAsset_getBuffer(asset);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "param %s size=%zu", parampath, len);
        std::string param_str(buf, len);
        AAsset_close(asset);

        ret = yolo.load_param_mem(param_str.c_str());
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "load_param_mem %s failed ret=%d", parampath, ret);
            return -1;
        }
    }

    // Load model from manually read asset (memory must persist)
    {
        AAsset* asset = AAssetManager_open(mgr, modelpath, AASSET_MODE_BUFFER);
        if (!asset)
        {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "cannot open asset %s", modelpath);
            return -1;
        }
        size_t len = AAsset_getLength(asset);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "model %s size=%zu", modelpath, len);
        model_buf.resize(len);
        memcpy(model_buf.data(), AAsset_getBuffer(asset), len);
        AAsset_close(asset);

        ret = yolo.load_model(model_buf.data());
        if (ret < 0)
        {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "load_model %s failed ret=%d", modelpath, ret);
            return -1;
        }
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "load_model %s ok bytes_consumed=%d", modelpath, ret);
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "model %s loaded successfully", modeltype);

    target_size = _target_size;
    num_class = _num_class;
    input_name = _input_name;
    output_name = _output_name;
    multi_output = _multi_output;
    scores_sigmoided = _scores_sigmoided;
    dfl_builtin = _dfl_builtin;
    pixel_format = _pixel_format;
    class_names = _class_names;

    if (!softmax)
    {
        softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        pd.set(0, 1);
        pd.set(1, 1);
        softmax->load_param(pd);

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;

        softmax->create_pipeline(opt);
    }
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int Yolo::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to target_size
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, pixel_format, width, height, w, h);

    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "detect: target_size=%d in_pad c=%d h=%d w=%d mean=%f,%f,%f norm=%f,%f,%f pixel_format=%d",
        target_size, in_pad.c, in_pad.h, in_pad.w,
        mean_vals[0], mean_vals[1], mean_vals[2],
        norm_vals[0], norm_vals[1], norm_vals[2],
        pixel_format);

    ncnn::Extractor ex = yolo.create_extractor();

    int ret_input = ex.input(input_name.c_str(), in_pad);
    if (ret_input != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, "ncnn", "ex.input(%s) failed ret=%d", input_name.c_str(), ret_input);
    }

    std::vector<Object> proposals;

    if (multi_output)
    {
        std::vector<const char*> out_names = {"out0", "out1", "out2"};
        std::vector<int> strides = {8, 16, 32};
        for (int i = 0; i < 3; i++)
        {
            ncnn::Mat out;
            ex.extract(out_names[i], out);
            std::vector<GridAndStride> gs;
            std::vector<int> single_stride = {strides[i]};
            generate_grids_and_stride(in_pad.w, in_pad.h, single_stride, gs);
            generate_proposals(gs, out, prob_threshold, proposals, softmax, num_class, scores_sigmoided, dfl_builtin);
        }
    }
    else
    {
        ncnn::Mat out;
        int ret = ex.extract(output_name.c_str(), out);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "extract %s ret=%d out c=%d h=%d w=%d", output_name.c_str(), ret, out.c, out.h, out.w);
        if (ret == 0 && out.w > 0)
        {
            std::vector<int> strides = {8, 16, 32};
            std::vector<GridAndStride> grid_strides;
            generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
            generate_proposals(grid_strides, out, prob_threshold, proposals, softmax, num_class, scores_sigmoided, dfl_builtin);
        }
        else
        {
            __android_log_print(ANDROID_LOG_ERROR, "ncnn", "extract failed or empty output");
        }
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
    struct
    {
        bool operator()(const Object& a, const Object& b) const
        {
            return a.rect.area() > b.rect.area();
        }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int Yolo::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
//        { 80, 175,  76},
//        { 74, 195, 139},
//        { 57, 220, 205},
//        { 59, 235, 255},
//        {  7, 193, 255},
//        {  0, 152, 255},
//        { 34,  87, 255},
//        { 72,  85, 121},
//        {158, 158, 158},
//        {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        snprintf(text, sizeof(text), "%s %.1f%%", class_names[(size_t)obj.label % class_names.size()], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }

    return 0;
}
