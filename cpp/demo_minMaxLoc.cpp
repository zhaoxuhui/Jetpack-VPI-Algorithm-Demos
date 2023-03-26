#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/MinMaxLoc.h>   // MinMaxLoc头文件
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
    int32_t width, height;
    vpiImageGetSize(input, &width, &height);

    // step4 获取影像格式
    VPIImageFormat type;
    vpiImageGetFormat(input, &type);

    // step5 创建输出keypoint
    VPIArray minCoords;
    vpiArrayCreate(10000, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &minCoords);

    VPIArray maxCoords;
    vpiArrayCreate(10000, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &maxCoords);

    // step6 创建对应Payload
    VPIPayload payload;
    vpiCreateMinMaxLoc(VPI_BACKEND_CPU, width, height, type, &payload);

    // step7 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step8 执行操作
    vpiSubmitMinMaxLoc(stream, VPI_BACKEND_CPU, payload, input, minCoords, maxCoords);

    // step9 等待所有操作执行完成
    vpiStreamSync(stream);

    // step10 创建锁定对象，取出数据
    VPIArrayData minCoordsData, maxCoordsData;
    vpiArrayLockData(minCoords, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &minCoordsData);
    vpiArrayLockData(maxCoords, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &maxCoordsData);

    VPIKeypointF32 *min_coords = (VPIKeypointF32 *) minCoordsData.buffer.aos.data;
    VPIKeypointF32 *max_coords = (VPIKeypointF32 *) maxCoordsData.buffer.aos.data;

    int min_i = min_coords[0].y;
    int min_j = min_coords[0].x;

    int max_i = max_coords[0].y;
    int max_j = max_coords[0].x;

    vpiArrayUnlock(maxCoords);
    vpiArrayUnlock(minCoords);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step11 销毁相关资源
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(payload);
    vpiImageDestroy(input);
    vpiArrayDestroy(minCoords);
    vpiArrayDestroy(maxCoords);
    // -------------------------------------------------------------------

    return 0;
}