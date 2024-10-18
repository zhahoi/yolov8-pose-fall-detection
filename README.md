# yolov8-pose-fall-detection
使用yolov8-pose进行人体关键点检测，通过计算人体各关键点关系进行人体摔倒检测（ncnn框架实现）



## 起

因为最近接收的项目里有一个模块涉及到人员倒地检测，因此就想自己先尝试实现这个功能。调研了一番功能实现，更倾向于使用关键点检测+摔倒判断这种方式解决。因为自己做指针仪表检测的时候用过关键点检测，对这个业务相对熟悉，因此可以减少后续代码理解和处理的时间。其次，自己也想学习一下新知识，丰富一下自己的技术路线。



## 承 

拟定选用的模型为YOLOV8-POSE，因为用得很频繁。数据集的话，COCO有专门的做人体关键点检测的数据集。本来我想直接找现成别人做好的可以直接拿来用，但是找了一圈发现不是收费就是下载链接失效，因此还是打算自己动手。



因为COCO数据集很大，要是下载整个数据集的话超级大，几十个G对我的网速和电脑内存都是挑战。最后，我在Kagge上找到了剥离出来的只有关键点检测的数据集，数据集地址：[coco-2017-keypoints](https://www.kaggle.com/datasets/asad11914/coco-2017-keypoints)，大概有10个G的样子。下载完的数据集没办法直接使用，需要将数据集格式从COCO转换到YOLO可以使用的格式，参考了一下CSDN博客：[COCO姿态检测标签转YOLO格式：用于YOLOv8关键点检测](https://blog.csdn.net/qq_40387714/article/details/140207491)。



转换完成之后，便可以用来训练模型了。因为我最后是使用ncnn框架部署模型，提前也参考了一些转换注意事项，怕有不支持的算子或者中间有一些坑，我再Gayhub检索到了一个别人已经转换好的，可以正常使用的仓库：[yolov8s-pose-ncnn](https://github.com/Rachel-liuqr/yolov8s-pose-ncnn)，这个仓库附带的CSDN博客也给出了训练python代码时需要修改的一些内容。跟着操作训练出了pt权重，熟练地转onnx再到ncnn，最后在虚拟机上推理发现可以正常使用，因此提取人体关键点部分已经解决。（我把我的python训练工程放在下面链接，可以下载使用：通过网盘分享的文件：ultralytics(zhahoi).zip 链接: https://pan.baidu.com/s/1wRSyj2c30HdaWsKIn-vYUg?pwd=8ux7 提取码: 8ux7）。



人体关键点检测检测比较容易实现，但是摔倒检测对我来说是个盲区。为此，我也在网上检索了别人的方法，发现要不是没有提及要不就是规则太过于简单，非常地不靠谱。为此，我检索了一些专利和论文，最终找到一个比较靠谱的专利。这篇专利描述的检测规则比较详细，且规则条数比较多应该可以覆盖大多数场景。因此，基于该篇专利描述的摔倒检测规则，在chatgpt和我个人的共同努力下，完成了人员摔倒规则的判定编写。写完之后，有一点值得头疼的是，判定规则有很多个阈值需要手动设定，但是专利没有给出具体值，我只能找一些视频推理一个个尝试修改，才有了最终稍微有点稳定且可用的检测算法代码。（参考的专利：【发明公布】[202311633969.8 一种基于人体关键点规则的摔倒检测方法及系统](javascript:;)）



## 转

以下是一些图片和视频的展示结果，有一些结果推理不正确，理论上是因为yolov8-pose识别人体关键点出错（为了减少训练时间，我只用了一万多张图片进行训练，如果使用所有的关键点检测数据进行训练的话，相信结果一定好很多）。

**测试视频**

[测试视频1](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output0.gif)

[测试视频2](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output1.gif)

[测试视频3](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output2.gif)

[测试视频4](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output3.gif)

[测试视频5](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output4.gif)

[测试视频6](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output5.gif)



**测试图片**

[测试图片1](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output.jpg)

[测试图片2](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output0.jpg)

[测试图片3](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output1.jpg)

[测试图片4](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output5.jpg)

[测试图片5](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output6.jpg)

[测试图片6](https://github.com/zhahoi/yolov8-pose-fall-detection/blob/main/outputs/output7.jpg)



## 合

花了几天时间完成了这个仓库的代码，有非常多的不足。尤其是摔倒预测那部分，有些阈值参数还是没设置好，存在一定程度的误报。如果看到这个仓库的你想自己使用的话，需要尽量调试一下这些参数，获得更好的检测效果。

后续有时间希望增加跟踪功能。

创作不易，如果觉得这个仓库还可以的话，麻烦给一个star，这就是对我最大的鼓励。
