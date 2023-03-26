#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/Array.h>  // Array头文件
#include <vpi/algo/TemplateMatching.h>  // TemplateMatching头文件

using namespace cv;
using namespace std;

int main() {
    string img_path = "../test-data/img2.png";
    string templ_path = "../test-data/img2-template.png";

    // 第一阶段：初始化
    // -------------------------------------------------------------------
    // step1 利用OpenCV读取影像
    Mat src_img = imread(img_path, IMREAD_GRAYSCALE);
    Mat templ_img = imread(templ_path, IMREAD_GRAYSCALE);

    // step2 基于读取的影像构造VPIImage对象
    VPIImage input, templ;
    vpiImageCreateWrapperOpenCVMat(src_img, 0, &input);
    vpiImageCreateWrapperOpenCVMat(templ_img, 0, &templ);

    // step3 获取影像大小
    int32_t srcW, srcH;
    vpiImageGetSize(input, &srcW, &srcH);

    int32_t templW, templH;
    vpiImageGetSize(templ, &templW, &templH);

    // step4 创建输出
    VPIImage output;
    vpiImageCreate(srcW - templW + 1, srcH - templH + 1, VPI_IMAGE_FORMAT_F32, 0, &output);

    // step5 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);

    // step6 创建对应payload
    VPIPayload payload;
    vpiCreateTemplateMatching(VPI_BACKEND_CUDA, srcW, srcH, &payload);

    vpiTemplateMatchingSetSourceImage(stream, VPI_BACKEND_CUDA, payload, input);

    vpiTemplateMatchingSetTemplateImage(stream, VPI_BACKEND_CUDA, payload, templ, NULL);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step7 执行操作
    vpiSubmitTemplateMatching(stream, VPI_BACKEND_CUDA, payload, output, VPI_TEMPLATE_MATCHING_NCC);
    vpiStreamSync(stream);

    // step8 创建锁定对象，取出数据
    VPIImageData outData;
    vpiImageLockData(output, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outData);

    // step9 将取出的数据转换为OpenCV格式并保存
    Mat out_mat;
    vpiImageDataExportOpenCVMat(outData, &out_mat);
    out_mat = out_mat * 255;
    imwrite("../test-data/out_templ.jpg", out_mat);

    vpiImageUnlock(output);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step10 销毁相关资源
    vpiStreamDestroy(stream);
    vpiImageDestroy(input);
    vpiImageDestroy(templ);
    vpiImageDestroy(output);
    // -------------------------------------------------------------------

    return 0;
}