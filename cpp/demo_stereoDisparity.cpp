#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/StereoDisparity.h>   // StereoDisparity头文件
#include <vpi/algo/ConvertImageFormat.h>    // 影像类型转换头文件

using namespace cv;
using namespace std;

int main() {
    string img_path1 = "../test-data/chair_stereo_left_grayscale.png";
    string img_path2 = "../test-data/chair_stereo_right_grayscale.png";

    // 第一阶段：初始化
    // -------------------------------------------------------------------
    // step1 利用OpenCV读取影像
    Mat img1 = imread(img_path1, IMREAD_GRAYSCALE);
    Mat img2 = imread(img_path2, IMREAD_GRAYSCALE);

    // step2 基于读取的影像构造VPIImage对象
    VPIImage left, right;
    vpiImageCreateWrapperOpenCVMat(img1, 0, &left);
    vpiImageCreateWrapperOpenCVMat(img2, 0, &right);

    // step3 获取影像大小
    int32_t w, h;
    vpiImageGetSize(left, &w, &h);

    // step4 获取影像格式
    VPIImageFormat type;
    vpiImageGetFormat(left, &type);

    // step5 创建输出影像
    VPIImage disparity;
    vpiImageCreate(w, h, VPI_IMAGE_FORMAT_S16, 0, &disparity);

    VPIImage confidence;
    vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U16, 0, &confidence);

    VPIImage display;
    vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U8, 0, &display);

    // step6 视差估计参数
    VPIStereoDisparityEstimatorParams params;
    vpiInitStereoDisparityEstimatorParams(&params);
    params.windowSize = 5;
    params.maxDisparity = 64;

    // step7 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);

    // step8 创建对应Payload
    VPIPayload stereo;
    vpiCreateStereoDisparityEstimator(VPI_BACKEND_CUDA, 480, 270, VPI_IMAGE_FORMAT_U16, NULL, &stereo);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step9 执行操作
    vpiSubmitStereoDisparityEstimator(stream, VPI_BACKEND_CUDA, stereo, left, right, disparity, confidence, &params);

    VPIConvertImageFormatParams cvtParams;
    vpiInitConvertImageFormatParams(&cvtParams);
    cvtParams.scale = 1.0f / (32 * params.maxDisparity) * 255;

    vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, disparity, display, &cvtParams);

    // step10 等待所有操作执行完成
    vpiStreamSync(stream);

    // step11 创建锁定对象，取出数据
    VPIImageData data;
    vpiImageLockData(display, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);

    // step12 将取出的数据转换为OpenCV格式并保存
    Mat out_mat;
    vpiImageDataExportOpenCVMat(data, &out_mat);
    imwrite("../test-data/out_disparity.jpg", out_mat);

    // step13 解除对对象的锁定
    vpiImageUnlock(display);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step14 销毁相关资源
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(stereo);
    vpiImageDestroy(left);
    vpiImageDestroy(right);
    vpiImageDestroy(display);
    vpiImageDestroy(disparity);
    vpiImageDestroy(confidence);
    // -------------------------------------------------------------------

    return 0;
}