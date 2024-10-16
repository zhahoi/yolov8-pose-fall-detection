#include "yolov8_pose.h"

#define YOLOV8_PARAM "/home/hit/Project/yolov8-pose-fall-detection/weights/yolov8-pose-human-opt.param"
#define YOLOV8_BIN "/home/hit/Project/yolov8-pose-fall-detection/weights/yolov8-pose-human-opt.bin"

std::unique_ptr<Yolov8Pose> yolov8Pose(new Yolov8Pose(YOLOV8_PARAM, YOLOV8_BIN, false));

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <image|video> <path>" << std::endl;
        return -1;
    }

    std::string inputType = argv[1];
    std::string inputPath = argv[2];

    if (inputType == "image") {
        cv::Mat image = cv::imread(inputPath);
        if (image.empty()) {
            std::cerr << "Could not read the image: " << inputPath << std::endl;
            return -1;
        }

        std::vector<Object> objects;
        yolov8Pose->detect_yolov8(image, objects);
        cv::Mat result;
        yolov8Pose->detect_objects(image, result, objects, SKELETON, KPS_COLORS, LIMB_COLORS);
        cv::imshow("Detection Result", result);
        cv::waitKey(0);

    } else if (inputType == "video") {
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Could not open the video: " << inputPath << std::endl;
            return -1;
        }

        cv::Mat frame;
        while (cap.read(frame)) {
            std::vector<Object> objects;
            yolov8Pose->detect_yolov8(frame, objects);
            cv::Mat result;
            yolov8Pose->detect_objects(frame, result, objects, SKELETON, KPS_COLORS, LIMB_COLORS);
            yolov8Pose->draw_fps(result);
            cv::imshow("Detection Result", result);

            // 按 'q' 键退出
            if (cv::waitKey(30) == 'q') {
                break;
            }
        }

    } else {
        std::cerr << "Invalid input type. Please use 'image' or 'video'." << std::endl;
        return -1;
    }

    return 0;
}
