#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/GaussianPyramid.h> // GaussianPyramid头文件
#include <vpi/Pyramid.h>    // VPI Pyramid类头文件

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

    // step5 创建金字塔对象
    VPIPyramid output;
    vpiPyramidCreate(w, h, type, 4, 0.5, 0, &output);

    // step6 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step7 将操作提交到流中执行
    vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CPU, input, output, VPI_BORDER_CLAMP);

    // step8 等待所有操作执行完成
    vpiStreamSync(stream);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step9 销毁相关资源
    vpiStreamDestroy(stream);
    vpiImageDestroy(input);
    vpiPyramidDestroy(output);
    // -------------------------------------------------------------------

    return 0;
}