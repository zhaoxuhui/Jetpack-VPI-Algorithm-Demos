#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/ORB.h>   // ORB头文件
#include <vpi/Pyramid.h>    // Pyramid头文件
#include <vpi/Array.h>  // Array头文件
#include <vpi/algo/GaussianPyramid.h>

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
    VPIArray corners;
    vpiArrayCreate(1000, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &corners);

    VPIArray descriptors;
    vpiArrayCreate(1000, VPI_ARRAY_TYPE_BRIEF_DESCRIPTOR, 0, &descriptors);

    VPIPayload payload;
    vpiCreateORBFeatureDetector(VPI_BACKEND_CPU, 10000, &payload);

    VPIORBParams params;
    vpiInitORBParams(&params);

    VPIPyramid inputPyr;
    vpiPyramidCreate(w, h, VPI_IMAGE_FORMAT_U8, params.pyramidLevels, 0.5, VPI_BACKEND_CPU, &inputPyr);

    // step6 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step7 创建高斯金字塔
    vpiSubmitGaussianPyramidGenerator(stream, VPI_BACKEND_CPU, input, inputPyr, VPI_BORDER_ZERO);
    vpiStreamSync(stream);

    // step8 提取ORB特征
    vpiSubmitORBFeatureDetector(stream, VPI_BACKEND_CPU, payload, inputPyr, corners, descriptors, &params,
                                VPI_BORDER_ZERO);
    vpiStreamSync(stream);

    // step9 创建锁定对象，取出数据
    VPIArrayData outData;
    vpiArrayLockData(corners, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outData);

    VPIKeypointF32 *kps = (VPIKeypointF32 *) outData.buffer.aos.data;

    int pos_i = kps[0].y;
    int pos_j = kps[0].x;

    cout << pos_i << " " << pos_j << endl;

    vpiArrayUnlock(corners);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step10 销毁相关资源
    vpiStreamDestroy(stream);
    vpiImageDestroy(input);
    vpiArrayDestroy(corners);
    vpiArrayDestroy(descriptors);
    // -------------------------------------------------------------------

    return 0;
}