#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <opencv2/imgcodecs.hpp>
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/Histogram.h>   // Histogram头文件
#include <vpi/Array.h>  // VPI Array类头文件

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

    // step5 创建输出keypoint
    VPIArray output;
    vpiArrayCreate(256, VPI_ARRAY_TYPE_U32, 0, &output);

    // step6 创建对应Payload
    VPIPayload payload;
    vpiCreateHistogramEven(VPI_BACKEND_CUDA, VPI_IMAGE_FORMAT_U8, 0, 255, 256, &payload);

    // step7 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step8 执行操作
    vpiSubmitHistogram(stream, VPI_BACKEND_CUDA, payload, input, output, 0);

    // step9 等待所有操作执行完成
    vpiStreamSync(stream);

    // step10 创建锁定对象，取出数据
    VPIArrayData data;
    vpiArrayLockData(output, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &data);

    // step11 解除对对象的锁定
    vpiArrayUnlock(output);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step12 销毁相关资源
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(payload);
    vpiImageDestroy(input);
    vpiArrayDestroy(output);
    // -------------------------------------------------------------------

    return 0;
}