#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/MixChannels.h>   // MixChannels头文件

using namespace cv;
using namespace std;

int main() {
    string img_path = "../test-data/img1.png";

    // 第一阶段：初始化
    // -------------------------------------------------------------------
    // step1 利用OpenCV读取影像
    Mat img1 = imread(img_path, IMREAD_COLOR);

    // step2 基于读取的影像构造VPIImage对象
    VPIImage input;
    vpiImageCreateWrapperOpenCVMat(img1, 0, &input);

    // step3 获取影像大小
    int32_t w, h;
    vpiImageGetSize(input, &w, &h);

    // step4 获取影像格式
    VPIImageFormat type;
    vpiImageGetFormat(input, &type);

    // step5 创建输出
    VPIImage outputs[3];
    for (int i = 0; i < 3; ++i) {
        vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U8, 0, &outputs[i]);
    }

    int mappingIn[3] = {0, 1, 2};
    int mappingOut[3] = {0, 1, 2};

    // step6 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step7 执行操作
    vpiSubmitMixChannels(stream, VPI_BACKEND_CPU, &input, 1, outputs, 3, mappingIn, mappingOut, 3);

    // step8 等待所有操作执行完成
    vpiStreamSync(stream);

    // step9 创建锁定对象，取出数据
    for (int i = 0; i < 3; ++i) {
        VPIImageData outData;
        vpiImageLockData(outputs[i], VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outData);

        // step13 将取出的数据转换为OpenCV格式并保存
        Mat out_mat;
        vpiImageDataExportOpenCVMat(outData, &out_mat);
        imwrite("../test-data/out_band_" + to_string(i + 1) + ".jpg", out_mat);

        vpiImageUnlock(outputs[i]);
    }
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step10 销毁相关资源
    vpiStreamDestroy(stream);
    vpiImageDestroy(input);
    for (int i = 0; i < 3; ++i) {
        vpiImageDestroy(outputs[i]);
    }
    // -------------------------------------------------------------------

    return 0;
}