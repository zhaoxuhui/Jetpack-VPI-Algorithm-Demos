#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/FASTCorners.h>   // FASTCorners头文件
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

    // step5 创建输出
    VPIArray corners;
    vpiArrayCreate(1000, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &corners);

    VPIFASTCornerDetectorParams params;
    vpiInitFASTCornerDetectorParams(&params);
    params.circleRadius = 3;
    params.arcLength = 9;
    params.intensityThreshold = 142;
    params.nonMaxSuppression = 1;

    // step6 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step7 执行操作
    vpiSubmitFASTCornerDetector(stream, VPI_BACKEND_CPU, input, corners, &params, VPI_BORDER_LIMITED);

    // step8 等待所有操作执行完成
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
    // -------------------------------------------------------------------

    return 0;
}