#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/CannyEdges.h>   // MixChannels头文件

using namespace cv;
using namespace std;

int main() {
    string img_path = "../test-data/img1.png";

    // 第一阶段：初始化
    // -------------------------------------------------------------------
    // step1 利用OpenCV读取影像
    Mat img1 = imread(img_path, IMREAD_GRAYSCALE);

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
    VPIImage output;
    vpiImageCreate(w, h, VPI_IMAGE_FORMAT_U8, 0, &output);

    VPIPayload payload;
    vpiCreateCannyEdgeDetector(VPI_BACKEND_CUDA, w, h, &payload);

    VPICannyEdgeDetectorParams params;
    vpiInitCannyEdgeDetectorParams(&params);

    // step6 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step7 执行操作
    float thresholdStrong = 300;
    float thresholdWeak = 100;
    float edgeValue = 255;
    float nonEdgeValue = 0;
    vpiSubmitCannyEdgeDetector(stream, VPI_BACKEND_CUDA, payload, input, output, thresholdStrong,
                               thresholdWeak, edgeValue, nonEdgeValue, &params);

    // step8 等待所有操作执行完成
    vpiStreamSync(stream);

    // step9 创建锁定对象，取出数据
    VPIImageData outData;
    vpiImageLockData(output, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outData);

    // step10 将取出的数据转换为OpenCV格式并保存
    Mat out_mat;
    vpiImageDataExportOpenCVMat(outData, &out_mat);
    imwrite("../test-data/out_canny.jpg", out_mat);

    vpiImageUnlock(output);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step11 销毁相关资源
    vpiStreamDestroy(stream);
    vpiImageDestroy(input);
    vpiImageDestroy(output);
    // -------------------------------------------------------------------

    return 0;
}