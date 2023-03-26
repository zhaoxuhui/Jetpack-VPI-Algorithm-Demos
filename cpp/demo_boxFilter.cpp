#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/BoxFilter.h> // BoxFilter头文件

using namespace cv;
using namespace std;

int main() {
    string img_path = "../test-data/img1.png";

    // 第一阶段：初始化
    // -------------------------------------------------------------------
    // step1 利用OpenCV读取影像
    Mat img = imread(img_path, IMREAD_GRAYSCALE);

    // step2 基于读取的影像构造VPIImage对象
    VPIImage input;
    vpiImageCreateWrapperOpenCVMat(img, 0, &input);

    // step3 获取影像大小
    int32_t w, h;
    vpiImageGetSize(input, &w, &h);

    // step4 获取影像格式
    VPIImageFormat type;
    vpiImageGetFormat(input, &type);

    // step5 创建输出影像
    VPIImage output;
    vpiImageCreate(w, h, type, 0, &output);

    // step6 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step7 将操作提交到流中执行
    vpiSubmitBoxFilter(stream, VPI_BACKEND_CUDA, input, output, 5, 5, VPI_BORDER_ZERO);

    // step8 等待所有操作执行完成
    vpiStreamSync(stream);

    // step9 创建锁定对象，取出数据
    VPIImageData outData;
    vpiImageLockData(output, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outData);

    // step10 将取出的数据转换为OpenCV格式并保存
    Mat out_mat;
    vpiImageDataExportOpenCVMat(outData, &out_mat);
    imwrite("../test-data/out_boxFilter.jpg", out_mat);

    // step11 解除对对象的锁定
    vpiImageUnlock(output);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step12 销毁相关资源
    vpiStreamDestroy(stream);
    vpiImageDestroy(input);
    vpiImageDestroy(output);
    // -------------------------------------------------------------------

    return 0;
}