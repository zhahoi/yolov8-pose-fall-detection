#include "yolov8_pose.h"

#define YOLOV8_PARAM "/home/hit/Project/yolov8-pose-human/weights/yolov8-pose-human-opt.param"
#define YOLOV8_BIN "/home/hit/Project/yolov8-pose-human/weights/yolov8-pose-human-opt.bin"
#define SAVE_PATH "/home/hit/Project/yolov8-pose-human/outputs"

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

        cv::imwrite(SAVE_PATH + std::string("/output.jpg"), result);
        cv::imshow("Detection Result", result);
        cv::waitKey(0);

    } else if (inputType == "video") {
        cv::VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Could not open the video: " << inputPath << std::endl;
            return -1;
        }

        // 获取视频的基本信息
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

        // 定义视频编码器和输出文件名
        cv::VideoWriter outputVideo(SAVE_PATH + std::string("/output.mp4"), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));

        cv::Mat frame;
        while (cap.read(frame)) {
            std::vector<Object> objects;
            yolov8Pose->detect_yolov8(frame, objects);
            cv::Mat result;
            yolov8Pose->detect_objects(frame, result, objects, SKELETON, KPS_COLORS, LIMB_COLORS);
            yolov8Pose->draw_fps(result);

            // 写入视频
            outputVideo.write(result);
            cv::imshow("Detection Result", result);

            // 按 'q' 键退出
            if (cv::waitKey(30) == 'q') {
                break;
            }
        }

        // 释放资源
        cap.release();
        outputVideo.release();
        cv::destroyAllWindows();

    } else {
        std::cerr << "Invalid input type. Please use 'image' or 'video'." << std::endl;
        return -1;
    }

    return 0;
}
