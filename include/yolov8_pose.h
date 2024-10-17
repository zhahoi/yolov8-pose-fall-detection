#ifndef YOLOV8_POSE_H
#define YOLOV8_POSE_H

#include "layer.h"
#include "net.h"

#include "opencv2/opencv.hpp"

#include <float.h>
#include <stdio.h>
#include <vector>

#include <algorithm> // 引入 std::min 和 std::max
#include <cmath> // 引入 std::round
#include <limits> // 引入 std::numeric_limits

#define MAX_STRIDE 32 // if yolov8-p6 model modify to 64

const int target_size = 320;
const float prob_threshold = 0.25f;
const float nms_threshold = 0.45f;

const std::vector<std::vector<unsigned int>> KPS_COLORS =
        {
            {0,   255, 0},
            {0,   255, 0},
            {0,   255, 0},
            {0,   255, 0},
            {0,   255, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {51,  153, 255},
            {51,  153, 255},
            {51,  153, 255},
            {51,  153, 255},
            {51,  153, 255},
            {51,  153, 255}
         };

const std::vector<std::vector<unsigned int>> SKELETON = 
        {
            {16, 14},
            {14, 12},
            {17, 15},
            {15, 13},
            {12, 13},
            {6,  12},
            {7,  13},
            {6,  7},
            {6,  8},
            {7,  9},
            {8,  10},
            {9,  11},
            {2,  3},
            {1,  2},
            {1,  3},
            {2,  4},
            {3,  5},
            {4,  6},
            {5,  7}
        };

const std::vector<std::vector<unsigned int>> LIMB_COLORS = 
        {
            {51,  153, 255},
            {51,  153, 255},
            {51,  153, 255},
            {51,  153, 255},
            {255, 51,  255},
            {255, 51,  255},
            {255, 51,  255},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {255, 128, 0},
            {0,   255, 0},
            {0,   255, 0},
            {0,   255, 0},
            {0,   255, 0},
            {0,   255, 0},
            {0,   255, 0},
            {0,   255, 0}
        };

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
	std::vector<float> kps;
};

class Yolov8Pose
{
public:
    Yolov8Pose(const char* param, const char* bin, bool useGPU);
    ~Yolov8Pose();

    static Yolov8Pose* yolov8_Detector;
    ncnn::Net* yolov8_Net;
    static bool hasGPU;

    int detect_yolov8(const cv::Mat& bgr, std::vector<Object>& objects); 
    void draw_objects(const cv::Mat &image, cv::Mat &res, const std::vector<Object> &objs, const std::vector<std::vector<unsigned int>> &SKELETON, const std::vector<std::vector<unsigned int>> &KPS_COLORS, const std::vector<std::vector<unsigned int>> &LIMB_COLORS); 
    void detect_objects(const cv::Mat &image, cv::Mat &res, const std::vector<Object> &objs, const std::vector<std::vector<unsigned int>> &SKELETON, const std::vector<std::vector<unsigned int>> &KPS_COLORS, const std::vector<std::vector<unsigned int>> &LIMB_COLORS); 

private:
    float sigmod(const float in);
    float softmax(const float* src, float* dst, int length);
    void generate_proposals(int stride, const ncnn::Mat& feat_blob, const float prob_threshold, std::vector<Object>& objects);
    float clamp(float val, float min = 0.f, float max = 1280.f);
    void non_max_suppression(std::vector<Object>& proposals, std::vector<Object>& results, int orin_h, int orin_w, float dh = 0, float dw = 0, float ratio_h = 1.0f, float ratio_w = 1.0f, float conf_thres = 0.25f, float iou_thres = 0.65f);
    bool fall_estimate(const std::vector<float>& kps);  // 摔倒检测
};


#endif // YOLOV8_POSE_H
