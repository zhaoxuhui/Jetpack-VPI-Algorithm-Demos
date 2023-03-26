#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/Remap.h> // Remap头文件
#include <vpi/WarpMap.h>    // VPI WarpMap类头文件

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

    // step6 创建稠密的Warp Map
    VPIWarpMap map;
    memset(&map, 0, sizeof(map));
    map.grid.numHorizRegions = 1;
    map.grid.numVertRegions = 1;
    map.grid.regionWidth[0] = w;
    map.grid.regionHeight[0] = h;
    map.grid.horizInterval[0] = 1;
    map.grid.vertInterval[0] = 1;
    vpiWarpMapAllocData(&map);

    // step7 创建小星球特效
    vpiWarpMapGenerateIdentity(&map);
    int i;
    for (i = 0; i < map.numVertPoints; ++i) {
        VPIKeypointF32 *row = (VPIKeypointF32 *) ((uint8_t *) map.keypoints + map.pitchBytes * i);
        int j;
        for (j = 0; j < map.numHorizPoints; ++j) {
            float x = row[j].x - w / 2.0f;
            float y = row[j].y - h / 2.0f;

            const float R = h / 8.0f; /* planet radius */

            const float r = sqrtf(x * x + y * y);

            float theta = M_PI + atan2f(y, x);
            float phi = M_PI / 2 - 2 * atan2f(r, 2 * R);

            row[j].x = fmod((theta + M_PI) / (2 * M_PI) * (w - 1), w - 1);
            row[j].y = (phi + M_PI / 2) / M_PI * (h - 1);
        }
    }

    // step8 创建算法需要的payload
    VPIPayload warp;
    vpiCreateRemap(VPI_BACKEND_CUDA, &map, &warp);

    // step9 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step10 将操作提交到流中执行
    vpiSubmitRemap(stream, VPI_BACKEND_CUDA, warp, input, output, VPI_INTERP_LINEAR, VPI_BORDER_ZERO, 0);

    // step11 等待所有操作执行完成
    vpiStreamSync(stream);

    // step12 创建锁定对象，取出数据
    VPIImageData outData;
    vpiImageLockData(output, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outData);

    // step13 将取出的数据转换为OpenCV格式并保存
    Mat out_mat;
    vpiImageDataExportOpenCVMat(outData, &out_mat);
    imwrite("../test-data/out_remap.jpg", out_mat);

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