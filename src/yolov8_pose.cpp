#include "yolov8_pose.h"

bool Yolov8Pose::hasGPU = true;
Yolov8Pose* Yolov8Pose::yolov8_Detector = nullptr;


Yolov8Pose::Yolov8Pose(const char* param, const char* bin, bool useGPU)
{
	this->yolov8_Net = new ncnn::Net();
	// opt
#if NCNN_VULKAN
	this->hasGPU = ncnn::get_gpu_count() > 0;
#endif
	this->yolov8_Net->opt.use_vulkan_compute = this->hasGPU && useGPU;
	this->yolov8_Net->opt.use_fp16_arithmetic = false;
	this->yolov8_Net->opt.num_threads = 4;
	this->yolov8_Net->load_param(param);
	this->yolov8_Net->load_model(bin);
}


Yolov8Pose::~Yolov8Pose()
{
	delete this->yolov8_Net;
}


float Yolov8Pose::sigmod(const float in)
{
	return 1.f / (1.f + expf(-1.f * in));
}


float Yolov8Pose::softmax(const float* src, float* dst, int length)
{
	float alpha = -FLT_MAX;
	for (int c = 0; c < length; c++)
	{
		float score = src[c];
		if (score > alpha)
		{
			alpha = score;
		}
	}

	float denominator = 0;
	float dis_sum = 0;
	for (int i = 0; i < length; ++i)
	{
		dst[i] = expf(src[i] - alpha);
		denominator += dst[i];
	}
	for (int i = 0; i < length; ++i)
	{
		dst[i] /= denominator;
		dis_sum += i * dst[i];
	}
	return dis_sum;
}


void Yolov8Pose::generate_proposals(
	int stride,
	const ncnn::Mat& feat_blob,
	const float prob_threshold,
	std::vector<Object>& objects
)
{
	const int reg_max = 16;
	float dst[16];
	const int num_w = feat_blob.w;
	const int num_grid_y = feat_blob.c;
	const int num_grid_x = feat_blob.h;
	const int kps_num = 17;

	const int num_class = num_w - 4 * reg_max;

	const int clsid = 0;

	for (int i = 0; i < num_grid_y; i++)
	{
		for (int j = 0; j < num_grid_x; j++)
		{
			const float* matat = feat_blob.channel(i).row(j);

			float score = matat[0];
            score = sigmod(score);
			if (score < prob_threshold)
			{
				continue;
			}

			float x0 = j + 0.5f - softmax(matat + 1, dst, 16);
			float y0 = i + 0.5f - softmax(matat + (1 + 16), dst, 16);
			float x1 = j + 0.5f + softmax(matat + (1 + 2 * 16), dst, 16);
			float y1 = i + 0.5f + softmax(matat + (1 + 3 * 16), dst, 16);

			x0 *= stride;
			y0 *= stride;
			x1 *= stride;
			y1 *= stride;

			std::vector<float> kps;
			for(int k=0; k<kps_num; k++)
			{
                float kps_x = (matat[1 + 64 + k * 3] * 2.f+ j) * stride;
                float kps_y = (matat[1 + 64 + k * 3 + 1] * 2.f + i) * stride;
                float kps_s = sigmod(matat[1 + 64 + k * 3 + 2]);
				
				kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
			}

			Object obj;
			obj.rect.x = x0;
			obj.rect.y = y0;
			obj.rect.width = x1 - x0;
			obj.rect.height = y1 - y0;
			obj.label = 0;
			obj.prob = score;
			obj.kps = kps;
			objects.push_back(obj);
		}
	}
}


float Yolov8Pose::clamp(float val, float min, float max)
{
	return val > min ? (val < max ? val : max) : min;
}


void Yolov8Pose::non_max_suppression(
	std::vector<Object>& proposals,
	std::vector<Object>& results,
	int orin_h,
	int orin_w,
	float dh,
	float dw,
	float ratio_h,
	float ratio_w,
	float conf_thres,
	float iou_thres
)
{
	results.clear();
	std::vector<cv::Rect> bboxes;
	std::vector<float> scores;
	std::vector<int> labels;
	std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

	for (auto& pro : proposals)
	{
		bboxes.push_back(pro.rect);
		scores.push_back(pro.prob);
		labels.push_back(pro.label);
        kpss.push_back(pro.kps);
	}

	cv::dnn::NMSBoxes(
		bboxes,
		scores,
		conf_thres,
		iou_thres,
		indices
	);

	for (auto i : indices)
	{
		auto& bbox = bboxes[i];
		float x0 = bbox.x;
		float y0 = bbox.y;
		float x1 = bbox.x + bbox.width;
		float y1 = bbox.y + bbox.height;
		float& score = scores[i];
		int& label = labels[i];

		x0 = (x0 - dw) / ratio_w;
		y0 = (y0 - dh) / ratio_h;
		x1 = (x1 - dw) / ratio_w;
		y1 = (y1 - dh) / ratio_h;

		x0 = clamp(x0, 0.f, orin_w);
		y0 = clamp(y0, 0.f, orin_h);
		x1 = clamp(x1, 0.f, orin_w);
		y1 = clamp(y1, 0.f, orin_h);

		Object obj;
		obj.rect.x = x0;
		obj.rect.y = y0;
		obj.rect.width = x1 - x0;
		obj.rect.height = y1 - y0;
		obj.prob = score;
		obj.label = label;
        obj.kps = kpss[i];
		for(int n=0; n<obj.kps.size(); n+=3)
		{
			obj.kps[n] = clamp((obj.kps[n] - dw) / ratio_w, 0.f, orin_w);
			obj.kps[n + 1] = clamp((obj.kps[n + 1] - dh) / ratio_h, 0.f, orin_h);
		}

		results.push_back(obj);
	}
}


int Yolov8Pose::detect_yolov8(const cv::Mat& bgr, std::vector<Object>& objects)
{
	int img_w = bgr.cols;
	int img_h = bgr.rows;

	// letterbox pad to multiple of MAX_STRIDE
	int w = img_w;
	int h = img_h;
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

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

	// pad to target_size rectangle
	// ultralytics/yolo/data/dataloaders/v5augmentations.py letterbox
	int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
	int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

	int top = hpad / 2;
	int bottom = hpad - hpad / 2;
	int left = wpad / 2;
	int right = wpad - wpad / 2;

	ncnn::Mat in_pad;
	ncnn::copy_make_border(in,
		in_pad,
		top,
		bottom,
		left,
		right,
		ncnn::BORDER_CONSTANT,
		114.f);

	const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	in_pad.substract_mean_normalize(0, norm_vals);

	ncnn::Extractor ex = yolov8_Net->create_extractor();

	ex.input("images", in_pad);

	std::vector<Object> proposals;

	// stride 8
	{
		ncnn::Mat out;
		ex.extract("output0", out);

		std::vector<Object> objects8;
		generate_proposals(8, out, prob_threshold, objects8);

		proposals.insert(proposals.end(), objects8.begin(), objects8.end());
	}

	// stride 16
	{
		ncnn::Mat out;
		ex.extract("378", out);

		std::vector<Object> objects16;
		generate_proposals(16, out, prob_threshold, objects16);

		proposals.insert(proposals.end(), objects16.begin(), objects16.end());
	}

	// stride 32
	{
		ncnn::Mat out;
		ex.extract("403", out);

		std::vector<Object> objects32;
		generate_proposals(32, out, prob_threshold, objects32);

		proposals.insert(proposals.end(), objects32.begin(), objects32.end());
	}

	non_max_suppression(proposals, objects,
		img_h, img_w, hpad / 2, wpad / 2,
		scale, scale, prob_threshold, nms_threshold);
	return 0;
}


void Yolov8Pose::draw_objects(
        const cv::Mat &image,
        cv::Mat &res,
        const std::vector<Object> &objs,
        const std::vector<std::vector<unsigned int>> &SKELETON,
        const std::vector<std::vector<unsigned int>> &KPS_COLORS,
        const std::vector<std::vector<unsigned int>> &LIMB_COLORS
) 
{
    res = image.clone();
    const int num_point = 17;
    for (auto &obj: objs) {
        cv::rectangle(
                res,
                obj.rect,
                {0, 0, 255},
                2
        );

        char text[256];
        sprintf(
                text,
                "person %.1f%%",
                obj.prob * 100
        );

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
                text,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
                &baseLine
        );

        int x = (int) obj.rect.x;
        int y = (int) obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(
                res,
                cv::Rect(x, y, label_size.width, label_size.height + baseLine),
                {0, 0, 255},
                -1
        );

        cv::putText(
                res,
                text,
                cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                {255, 255, 255},
                1
        );

        auto &kps = obj.kps;
        for (int k = 0; k < num_point + 2; k++)
		{
            if (k < num_point)
			{
                int kps_x = std::round(kps[k * 3]);
                int kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > 0.5f)
				{
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto &ske = SKELETON[k];
            int pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.5f && pos2_s > 0.5f)
			{
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            }
        }
    }
}


inline cv::Scalar random_get_color(int idx)
{
	idx += 3;
	return cv::Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}

void Yolov8Pose::detect_objects_tracker(
        const cv::Mat &image,
        cv::Mat &res,
        const std::vector<Object> &objs,
		const std::vector<STrack>& output_stracks,
        const std::vector<std::vector<unsigned int>> &SKELETON,
        const std::vector<std::vector<unsigned int>> &KPS_COLORS,
        const std::vector<std::vector<unsigned int>> &LIMB_COLORS
) 
{
    res = image.clone();
    const int num_point = 17;
    for (auto &obj: objs) {
		/*
        cv::rectangle(
                res,
                obj.rect,
                {0, 0, 255},
                2
        );

        char text[256];
        sprintf(
                text,
                "person %.1f%%",
                obj.prob * 100
        );

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
                text,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
                &baseLine
        );

        int x = (int) obj.rect.x;
        int y = (int) obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(
                res,
                cv::Rect(x, y, label_size.width, label_size.height + baseLine),
                {255, 0, 0},
                -1
        );

        cv::putText(
                res,
                text,
                cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                {255, 255, 255},
                1
        );
		*/

        auto &kps = obj.kps;
        for (int k = 0; k < num_point + 2; k++)
		{
            if (k < num_point)
			{
                int kps_x = std::round(kps[k * 3]);
                int kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];

                if (kps_s > 0.0f)   // 0.0f 
				{
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto &ske = SKELETON[k];
            int pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.0f && pos2_s > 0.0f)   // 0.0f
			{
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            }
        }

        bool is_fall = fall_estimate(kps);

        char text_[32];
        if (is_fall)
        {
            sprintf(text_, "STATUS:FALL");
        }
        else
        {
            sprintf(text_, "STATUS:NORMAL");
        }

        int baseLine_ = 0;
        cv::Size label_size_ = cv::getTextSize(
                text_,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
                &baseLine_
        );

        int x_ = (int) obj.rect.x;
        int y_ = (int) obj.rect.y - 15;

        if (y_ > res.rows)
            y_ = res.rows;

        if (is_fall)
        {
            cv::rectangle(
                res,
                cv::Rect(x_, y_, label_size_.width, label_size_.height + baseLine_),
                {0, 0, 255},
                -1
            );
        }
        else
        {
            cv::rectangle(
                res,
                cv::Rect(x_, y_, label_size_.width, label_size_.height + baseLine_),
                {0, 255, 0},
                -1
            );
        }

        cv::putText(
                res,
                text_,
                cv::Point(x_, y_ + label_size_.height),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                {0, 0, 0},
                1
        );
    }

	// tracker
	for (int i = 0; i < output_stracks.size(); i++)
	{
		std::vector<float> tlwh = output_stracks[i].tlwh;
		// bool vertical = tlwh[2] / tlwh[3] > 1.6;
		// if (tlwh[2] * tlwh[3] > 20 && !vertical)
		if (tlwh[2] * tlwh[3] > 20)
		{
			cv::Scalar s = random_get_color(output_stracks[i].track_id);

			char text[256];
			sprintf(
					text,
					"ID: %d",
					output_stracks[i].track_id
			);

			int baseLine = 0;
			cv::Size label_size = cv::getTextSize(
					text,
					cv::FONT_HERSHEY_SIMPLEX,
					0.4,
					1,
					&baseLine
			);

			int x = (int) tlwh[0];
			int y = (int) tlwh[1] + 1;

			if (y > res.rows)
				y = res.rows;


			// 绘制矩形框
			cv::rectangle(
				res,
				cv::Rect(x, y, label_size.width, label_size.height + baseLine),
				s,
				cv::FILLED // 填充矩形框，使其作为背景
			);

			// 绘制文本边框
			cv::rectangle(
				res,
				cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3] + baseLine),
				s,
				2
			);

			// 绘制文本
			cv::putText(
				res,
				text,
				cv::Point(x, y + label_size.height),
				cv::FONT_HERSHEY_SIMPLEX,
				0.4,
				{255, 255, 255},
				1
			);
		}
	}
}


void Yolov8Pose::detect_objects(
        const cv::Mat &image,
        cv::Mat &res,
        const std::vector<Object> &objs,
        const std::vector<std::vector<unsigned int>> &SKELETON,
        const std::vector<std::vector<unsigned int>> &KPS_COLORS,
        const std::vector<std::vector<unsigned int>> &LIMB_COLORS
) 
{
    res = image.clone();
    const int num_point = 17;
    for (auto &obj: objs) {
        cv::rectangle(
                res,
                obj.rect,
                {0, 0, 255},
                2
        );

        char text[256];
        sprintf(
                text,
                "person %.1f%%",
                obj.prob * 100
        );

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(
                text,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
                &baseLine
        );

        int x = (int) obj.rect.x;
        int y = (int) obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(
                res,
                cv::Rect(x, y, label_size.width, label_size.height + baseLine),
                {255, 0, 0},
                -1
        );

        cv::putText(
                res,
                text,
                cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                {255, 255, 255},
                1
        );

        auto &kps = obj.kps;
        for (int k = 0; k < num_point + 2; k++)
		{
            if (k < num_point)
			{
                int kps_x = std::round(kps[k * 3]);
                int kps_y = std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];

                if (kps_s > 0.0f)   // 0.0f 
				{
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[k][0], KPS_COLORS[k][1], KPS_COLORS[k][2]);
                    cv::circle(res, {kps_x, kps_y}, 5, kps_color, -1);
                }
            }
            auto &ske = SKELETON[k];
            int pos1_x = std::round(kps[(ske[0] - 1) * 3]);
            int pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

            int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
            int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

            float pos1_s = kps[(ske[0] - 1) * 3 + 2];
            float pos2_s = kps[(ske[1] - 1) * 3 + 2];

            if (pos1_s > 0.0f && pos2_s > 0.0f)   // 0.0f
			{
                cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
                cv::line(res, {pos1_x, pos1_y}, {pos2_x, pos2_y}, limb_color, 2);
            }
        }

        bool is_fall = fall_estimate(kps);

        char text_[32];
        if (is_fall)
        {
            sprintf(text_, "STATUS:FALL");
        }
        else
        {
            sprintf(text_, "STATUS:NORMAL");
        }

        int baseLine_ = 0;
        cv::Size label_size_ = cv::getTextSize(
                text_,
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                1,
                &baseLine_
        );

        int x_ = (int) obj.rect.x;
        int y_ = (int) obj.rect.y - 15;

        if (y_ > res.rows)
            y_ = res.rows;

        if (is_fall)
        {
            cv::rectangle(
                res,
                cv::Rect(x_, y_, label_size_.width, label_size_.height + baseLine_),
                {0, 0, 255},
                -1
            );
        }
        else
        {
            cv::rectangle(
                res,
                cv::Rect(x_, y_, label_size_.width, label_size_.height + baseLine_),
                {0, 255, 0},
                -1
            );
        }

        cv::putText(
                res,
                text_,
                cv::Point(x_, y_ + label_size_.height),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                {0, 0, 0},
                1
        );
    }
}


// 通过检测到的关键点，根据设定的一些规则进行是否摔倒判定
bool Yolov8Pose::fall_estimate(const std::vector<float>& kps)
{
	// 设置一个判断是否为摔倒的变量
	bool is_fall = false;

	// 1. 先获取哪些用于判断的点坐标
	cv::Point L_shoulder = cv::Point((int)kps[5 * 3], (int)kps[5 * 3 + 1]);  // 左肩
	float L_shoulder_confi = kps[5 * 3 + 2];  
	cv::Point R_shoulder = cv::Point((int)kps[6 * 3], (int)kps[6 * 3 + 1]);  // 右肩
	float R_shoulder_confi = kps[6 * 3 + 2];
	cv::Point C_shoulder = cv::Point((int)(L_shoulder.x + R_shoulder.x) / 2, (int)(L_shoulder.y + R_shoulder.y) / 2);  // 肩部中点

	cv::Point L_hip = cv::Point((int)kps[11 * 3], (int)kps[11 * 3 + 1]);  // 左髋
	float L_hip_confi = kps[11 * 3 + 2]; 
	cv::Point R_hip = cv::Point((int)kps[12 * 3], (int)kps[12 * 3 + 1]);  // 右髋
	float R_hip_confi = kps[12 * 3 + 2]; 
	cv::Point C_hip = cv::Point((int)(L_hip.x + R_hip.x) / 2, (int)(L_hip.y + R_hip.y) / 2);  // 髋部中点

	cv::Point L_knee = cv::Point((int)kps[13 * 3], (int)kps[13 * 3 + 1]);  // 左膝
	float L_knee_confi = kps[13 * 3 + 2]; 
	cv::Point R_knee = cv::Point((int)kps[14 * 3], (int)kps[14 * 3 + 1]);  // 右膝
	float R_knee_confi = kps[14 * 3 + 2]; 
	cv::Point C_knee = cv::Point((int)(L_knee.x + R_knee.x) / 2, (int)(L_knee.y + R_knee.y) / 2);  // 膝部中点

	cv::Point L_ankle = cv::Point((int)kps[15 * 3], (int)kps[15 * 3 + 1]);  // 左踝
	float L_ankle_confi = kps[15 * 3 + 2]; 
	cv::Point R_ankle = cv::Point((int)kps[16 * 3], (int)kps[16 * 3 + 1]);  // 右踝
	float R_ankle_confi = kps[16 * 3 + 2];
	cv::Point C_ankle = cv::Point((int)(L_ankle.x + R_ankle.x) / 2, (int)(L_ankle.y + R_ankle.y) / 2);  // 计算脚踝中点

	// 2. 第一个判定条件： 若肩的纵坐标最小值min(L_shoulder.y, R_shoulder.y)不低于脚踝的中心点的纵坐标C_ankle.y
	// 且p_shoulders、p_ankle关键点置信度大于预设的阈值，则疑似摔倒。
	if (L_shoulder_confi > 0.0f && R_shoulder_confi > 0.0f && L_ankle_confi > 0.0f && R_ankle_confi > 0.0f)
	{
		int shoulder_y_min = std::min(L_shoulder.y, R_shoulder.y);
		if (shoulder_y_min >= C_ankle.y)
		{
			is_fall = true;
			return is_fall;
		}
	}

	// 3. 第二个判断条件：若肩的纵坐标最大值max(L_shoulder.y, R_shoulder.y)大于膝盖纵坐标的最小值min(L_knee.y, R_knee.y)，
	// 且p_shoulders、p_knees关键点置信度大于预设的阈值，则疑似摔倒。
	if (L_shoulder_confi > 0.0f && R_shoulder_confi > 0.0f && L_knee_confi > 0.0f && R_knee_confi > 0.0f)
	{
		int shoulder_y_max = std::max(L_shoulder.y, R_shoulder.y);
		int knee_y_min = std::min(L_knee.y, R_knee.y);
		if (shoulder_y_max > knee_y_min)
		{
			is_fall = true;
			return is_fall;
		}
	}

	// 4, 第三个判断条件：计算关键点最小外接矩形的宽高比。p0～p16在x方向的距离是xmax-xmin，在方向的距离是ymax-ymin，
	// 若(xmax-xmin) / (ymax-ymin)不大于指定的比例阈值，则判定为未摔倒，不再进行后续判定。
	const int num_point = 17;  // 17个关键点

	// 初始化xmin, ymin为最大值，xmax, ymax为最小值
	int xmin = std::numeric_limits<int>::max();
	int ymin = std::numeric_limits<int>::max();
	int xmax = std::numeric_limits<int>::min();
	int ymax = std::numeric_limits<int>::min();
	
	for (int k = 0; k < num_point + 2; k++)
	{
		if (k < num_point)
		{
			int kps_x = std::round(kps[k * 3]);  // 关键点x
			int kps_y = std::round(kps[k * 3 + 1]);  // 关键点y
			float kps_s = kps[k * 3 + 2];  // 可见性

			if (kps_s > 0.0f)
			{
				// 更新xmin, xmax, ymin, ymax
				xmin = std::min(xmin, kps_x);
				xmax = std::max(xmax, kps_x);
				ymin = std::min(ymin, kps_y);
				ymax = std::max(ymax, kps_y);
			}
		}
	}

	// 检查是否存在有效的宽度和高度
	if (xmax > xmin && ymax > ymin)
	{
		float aspect_ratio = static_cast<float>(xmax - xmin) / (ymax - ymin);
		
		// 如果宽高比大于指定阈值，则判定为摔倒
		if (aspect_ratio > 0.90f)
		{
			is_fall = true;
			return is_fall;
		}
	}

	// 5. 第四个判断条件：通过两膝与髋部中心点的连线与地面的夹角判断。首先假定有两点p1＝(x1 ,y1 )，p2＝(x2 ,y2 )，那么两点连接线与地面的角度计算公式为：
	// 												θ = arctan((y2-y1) / (x2-x1)) * 180 / pi
	// 此处左膝与髋部的两点是(C_hip, L_knee)，与地面夹角表示为θ1；右膝与髋部的两点 是(C_hip, R_knee)，与地面夹角表示为θ2，
	// 若min(θ1 ,θ2 )＜th1 或 max(θ1 ,θ2 )＜th2，且p_knees、 p_hips关键点置信度大于预设的阈值，则疑似摔倒
    if (L_knee_confi > 0.0f && R_knee_confi > 0.0f && L_hip_confi > 0.0f && R_hip_confi > 0.0f)
    {
        // 左膝与髋部中心的角度
        float theta1 = std::atan2(L_knee.y - C_hip.y, L_knee.x - C_hip.x) * 180.0f / CV_PI;
        // 右膝与髋部中心的角度
        float theta2 = std::atan2(R_knee.y - C_hip.y, R_knee.x - C_hip.x) * 180.0f / CV_PI;

        float min_theta = std::min(std::abs(theta1), std::abs(theta2));
        float max_theta = std::max(std::abs(theta1), std::abs(theta2));

        /*
        根据人体运动规律，阈值th1 和 th2 应设置为代表正常和摔倒之间的界限角度。
        通常情况下，如果人体处于站立或行走状态，膝盖与髋部的连线与地面之间的角度应接近垂直或有一定的倾斜，而当摔倒时，这个角度通常会明显减小。
        th1: 用于判断两膝与髋部的连线与地面的最小角度。可以设定为 20度。如果min(θ1 ,θ2 )＜th1,即两膝与髋部的连线明显接近平行于地面，则有可能表示摔倒的姿态。
        th2: 用于判断两膝与髋部的连线与地面的最大角度。可以设定为 45度。如果max(θ1 ,θ2 )＜th2,即两膝与髋部的连线即使有倾斜但依然小于正常站立的角度范围，也可能表明摔倒的风险。
        */

        // 设定阈值 th1 和 th2，用于判定是否摔倒
        float th1 = 30.0f;  // 假设的最小角度阈值  // 20, 30 ,25
        float th2 = 70.0f;  // 假设的最大角度阈值  // 35, 40, 45, 50, 60

        // std::cout << "min_theta: " << min_theta  << ", " << "max_theta: " << max_theta << std::endl;

        if ((min_theta) < th1 && (max_theta < th2))
        {
			is_fall = true;
			return is_fall;
        }
    }

	// 第五个判断条件：通过肩、髋部、膝盖夹角，髋部、膝盖、脚踝夹角判断。
	// 首先假定有四点p1＝(x1 ,y1 )，p2＝(x2 ,y2 )，p3＝(x3 ,y3 )，p4＝(x4 ,y4 )，其中，p1 p2组 成的向量为v1＝(x2 -x1 ,y2 -y1 )，
	// p3 p4组成的向量为v2＝(x4 -x3 ,y4 -y3 )。v1 v2的夹角计算公式为：
	// θ = arctan((v1 * v2) / (sqrt(v1 * v1) * sqrt(v2 * v2))) * 180 / pi
	// 此处， v1＝(c_shoulder.x - c_hips.x, c_shoulders.y - c_hips.y) 
	//	v2＝(c_knees.x -c_hips.x, c_knees .y - c_hips.y) 
	//	v3＝(c_hips.x - c_knees.x, c_hips.y - c_knees.y) 
	// 	v4＝(c_foot.x - c_knees.x, c_foot.y - c_knees.y) 
	// v1 v2两个向量的夹角表示为θ3，v3 v4两个向量的夹角表示为θ4。若θ3＞th3或θ4＜ th4，且p_shoulders、p_knees、p_hips、p_foot关键点置信度大于预设的阈值，则疑似摔倒。
	// 第五个判断条件：通过肩、髋部、膝盖夹角，髋部、膝盖、脚踝夹角判断。
	// 如果肩、髋、膝和脚踝关键点的置信度都大于阈值，我们继续进行角度的计算。
	if (L_shoulder_confi > 0.0f && R_shoulder_confi > 0.0f && L_hip_confi > 0.0f && R_hip_confi > 0.0f && L_knee_confi > 0.0f && R_knee_confi > 0.0f &&
		L_ankle_confi > 0.0f && R_ankle_confi > 0.0f)
	{
		// 计算向量 v1 和 v2
		cv::Point2f v1(C_shoulder.x - C_hip.x, C_shoulder.y - C_hip.y);
		cv::Point2f v2(C_knee.x - C_hip.x, C_knee.y - C_hip.y);

		// 计算向量 v3 和 v4
		cv::Point2f v3(C_hip.x - C_knee.x, C_hip.y - C_knee.y);
		cv::Point2f v4(C_ankle.x - C_knee.x, C_ankle.y - C_knee.y);

		// 计算向量 v1 和 v2 的夹角 θ3
		float dot_product1 = v1.x * v2.x + v1.y * v2.y;
		float magnitude1 = std::sqrt(v1.x * v1.x + v1.y * v1.y) * std::sqrt(v2.x * v2.x + v2.y * v2.y);
		float theta3 = std::acos(dot_product1 / magnitude1) * 180.0f / CV_PI;

		// 计算向量 v3 和 v4 的夹角 θ4
		float dot_product2 = v3.x * v4.x + v3.y * v4.y;
		float magnitude2 = std::sqrt(v3.x * v3.x + v3.y * v3.y) * std::sqrt(v4.x * v4.x + v4.y * v4.y);
		float theta4 = std::acos(dot_product2 / magnitude2) * 180.0f / CV_PI;

        /*
        定义: 𝜃3是肩、髋、膝三点形成的向量夹角。通常情况下，站立时肩、髋和膝盖的夹角应该接近 180度（几乎成一条直线）。
        摔倒判断: 当人摔倒或发生意外时，这个角度可能会急剧减少。一个合理的阈值可以设定为 120度 或 130度。
        定义: 𝜃4是髋、膝、脚踝三点形成的向量夹角。站立或正常行走时，这个角度通常在 160度 到 180度 之间（接近直线）。在弯曲或下蹲时，这个角度可能会降低。
        摔倒判断: 如果此角度降低到一个较小的值（例如人体接近折叠或蜷缩的状态），可以判断为摔倒。一个合理的阈值可以设定为 60度 或 70度。
        */

        /*
        th3（肩、髋、膝夹角）被设定为70.0f。这个值是基于假设站立时肩、髋和膝盖的夹角应该接近180度（几乎成一条直线），但在摔倒时这个角度可能会急剧减少。
        th4（髋、膝、脚踝夹角）被设定为60.0f。这个值是基于假设站立或正常行走时，这个角度通常在160度到180度之间，而在摔倒或身体接近折叠状态时，这个角度可能会显著降低。
        */

		// 设定角度阈值 th3 和 th4
		float th3 = 70.0f;  // 假设的阈值，肩、髋和膝的角度  // 120.0f, 130.0f 
		float th4 = 30.0f;   // 假设的阈值，髋、膝和脚踝的角度  // 60.0f, 70.0f 

		// 判断是否符合摔倒条件
		if ((theta3 < th3) && (theta4 < th4))
		{
            // std::cout << "theta3: " << theta3  << ", " << "theta4: " << theta4 << std::endl;
			is_fall = true;
		}
		
        return is_fall;
	}
}


int Yolov8Pose::draw_fps(cv::Mat& image)
{
	// resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = { 0.f };

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f)
        {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--)
        {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f)
        {
            return 0;
        }

        for (int i = 0; i < 10; i++)
        {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = image.cols - label_size.width;

    cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
        cv::Scalar(255, 255, 255), -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}