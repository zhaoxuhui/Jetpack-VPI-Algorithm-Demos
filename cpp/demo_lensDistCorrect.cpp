#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/LensDistortionModels.h>   // DistortionModel头文件
#include <vpi/algo/Remap.h>   // Remap头文件

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
    int32_t width, height;
    vpiImageGetSize(input, &width, &height);

    // step4 获取影像格式
    VPIImageFormat type;
    vpiImageGetFormat(input, &type);

    // step5 创建输出影像
    VPIImage output;
    vpiImageCreate(width, height, type, 0, &output);

    // step6 构造校正变换关系
    VPIWarpMap map;
    memset(&map, 0, sizeof(map));
    map.grid.numHorizRegions = 1;
    map.grid.numVertRegions = 1;
    map.grid.regionWidth[0] = width;
    map.grid.regionHeight[0] = height;
    map.grid.horizInterval[0] = 1;
    map.grid.vertInterval[0] = 1;
    vpiWarpMapAllocData(&map);

    // step7 构造校正模型
    VPIFisheyeLensDistortionModel fisheye;
    memset(&fisheye, 0, sizeof(fisheye));
    fisheye.mapping = VPI_FISHEYE_EQUIDISTANT;
    fisheye.k1 = -0.126;
    fisheye.k2 = 0.004;
    fisheye.k3 = 0;
    fisheye.k4 = 0;

    float sensorWidth = 22.2; /* APS-C sensor */
    float focalLength = 7.5;
    float f = focalLength * width / sensorWidth;
    const VPICameraIntrinsic K =
            {
                    {f, 0, static_cast<float>(width / 2.0)},
                    {0, f, static_cast<float>(height / 2.0)}
            };
    const VPICameraExtrinsic X =
            {
                    {1, 0, 0, 0},
                    {0, 1, 0, 0},
                    {0, 0, 1, 0}
            };

    vpiWarpMapGenerateFromFisheyeLensDistortionModel(K, X, K, &fisheye, &map);

    // step8 创建校正对应Payload
    VPIPayload warp;
    vpiCreateRemap(VPI_BACKEND_CUDA, &map, &warp);

    // step9 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step10 执行操作
    vpiSubmitRemap(stream, VPI_BACKEND_CUDA, warp, input, output, VPI_INTERP_CATMULL_ROM, VPI_BORDER_ZERO, 0);

    // step11 等待所有操作执行完成
    vpiStreamSync(stream);

    // step12 创建锁定对象，取出数据到tmpBuffer
    VPIImageData data;
    vpiImageLockData(output, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);

    // step13 将取出的数据转换为OpenCV格式并保存
    Mat out_mat;
    vpiImageDataExportOpenCVMat(data, &out_mat);
    imwrite("../test-data/out_dist.jpg", out_mat);

    // step14 解除对对象的锁定
    vpiImageUnlock(output);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step15 销毁相关资源
    vpiStreamDestroy(stream);
    vpiPayloadDestroy(warp);
    vpiWarpMapFreeData(&map);
    vpiImageDestroy(input);
    vpiImageDestroy(output);
    // -------------------------------------------------------------------

    return 0;
}