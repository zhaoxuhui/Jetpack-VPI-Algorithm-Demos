#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/FFT.h>   // IFFT头文件

using namespace cv;
using namespace std;

int main() {
    string img_path = "../test-data/out_fft.jpg";

    // 第一阶段：初始化
    // -------------------------------------------------------------------
    // step1 利用OpenCV读取影像
    Mat img = imread(img_path, IMREAD_GRAYSCALE);

    Mat img_norm;
    img.convertTo(img_norm, CV_32FC2);
    img_norm = img_norm / 255;

    // step2 基于读取的影像构造VPIImage对象
    VPIImage input;
    vpiImageCreateWrapperOpenCVMat(img_norm, 0, &input);

    // step3 获取影像大小
    int32_t width, height;
    vpiImageGetSize(input, &width, &height);

    // step4 获取影像格式
    VPIImageFormat type;
    vpiImageGetFormat(input, &type);

    // step5 创建输出影像
    VPIImage output;
    vpiImageCreate(width, height, VPI_IMAGE_FORMAT_F32, 0, &output);

    // step6 创建IFFT对应Payload
    VPIPayload ifft;
    vpiCreateIFFT(VPI_BACKEND_CUDA, width, height, VPI_IMAGE_FORMAT_2F32, VPI_IMAGE_FORMAT_2F32, &ifft);

    // step7 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step8 执行操作
    vpiSubmitIFFT(stream, VPI_BACKEND_CUDA, ifft, input, output, 0);

    // step9 等待所有操作执行完成
    vpiStreamSync(stream);

    // step10 创建锁定对象，取出数据到tmpBuffer
    VPIImageData data;
    vpiImageLockData(output, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);

    // step11 将取出的数据转换为OpenCV格式并保存
    Mat out_mat;
    vpiImageDataExportOpenCVMat(data, &out_mat);
    imwrite("../test-data/out_ifft.jpg", out_mat);

    // step12 解除对对象的锁定
    vpiImageUnlock(output);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step13 销毁相关资源
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(ifft);
    vpiImageDestroy(input);
    vpiImageDestroy(output);
    // -------------------------------------------------------------------

    return 0;
}