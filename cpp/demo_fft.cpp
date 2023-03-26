#include <iostream> // C++标准头文件
#include <opencv2/opencv.hpp>   // OpenCV头文件
#include <vpi/OpenCVInterop.hpp>    // VPI与OpenCV互操作的头文件
#include <vpi/Image.h>  // VPI Image类头文件
#include <vpi/Stream.h> // VPI Stream类头文件
#include <vpi/algo/ConvertImageFormat.h> // ConvertImageFormat头文件
#include <vpi/algo/FFT.h>   // FFT头文件

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

    // step5 创建Float32类型的输入影像
    VPIImage inputF32;
    vpiImageCreate(width, height, VPI_IMAGE_FORMAT_F32, 0, &inputF32);

    // step6 创建用于可视化的谱段的影像
    float *tmpBuffer = (float *) malloc(width * 2 * height * sizeof(float));

    // step7 创建输出谱段影像
    VPIImage spectrum;
    vpiImageCreate(width / 2 + 1, height, VPI_IMAGE_FORMAT_2F32, 0, &spectrum);

    // step8 创建FFT对应Payload
    VPIPayload fft;
    vpiCreateFFT(VPI_BACKEND_CUDA, width, height, VPI_IMAGE_FORMAT_F32, VPI_IMAGE_FORMAT_2F32, &fft);

    // step9 创建执行流
    VPIStream stream;
    vpiStreamCreate(0, &stream);
    // -------------------------------------------------------------------

    // 第二阶段：处理并保存结果
    // -------------------------------------------------------------------
    // step10 将输入的Uint8类型影像转化为Float32类型
    vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, input, inputF32, NULL);

    // step11 执行傅立叶变换
    vpiSubmitFFT(stream, VPI_BACKEND_CUDA, fft, inputF32, spectrum, 0);

    // step12 等待所有操作执行完成
    vpiStreamSync(stream);

    // step13 创建锁定对象，取出数据到tmpBuffer
    VPIImageData data;
    vpiImageLockData(spectrum, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &data);

    // step13.1 可视化谱段处理
    // Fills the right half of the complex data with the missing values. The left half is copied directly from VPI's output.
    /* make width/height even*/
    width = width & -2;
    height = height & -2;

    /* center pixel coordinate */
    int cx = width / 2;
    int cy = height / 2;

    /* Image data is in host-accessible pitch-linear layout. */
    assert(data.bufferType == VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR);
    VPIImageBufferPitchLinear *dataPitch = &data.buffer.pitch;

    int i, j;
    for (i = 0; i < (int) height; ++i) {
        for (j = 0; j < (int) width; ++j) {
            float re, im;
            if (j < cx) {
                const float *pix =
                        (const float *) ((const char *) dataPitch->planes[0].data +
                                         i * dataPitch->planes[0].pitchBytes) +
                        j * 2;
                re = pix[0];
                im = pix[1];
            } else {
                const float *pix = (const float *) ((const char *) dataPitch->planes[0].data +
                                                    ((height - i) % height) * dataPitch->planes[0].pitchBytes) +
                                   ((width - j) % width) * 2;
                /* complex conjugate */
                re = pix[0];
                im = -pix[1];
            }

            tmpBuffer[i * (width * 2) + j * 2] = re;
            tmpBuffer[i * (width * 2) + j * 2 + 1] = im;
        }
    }

    // step13.2 Convert the complex frequencies into normalized log-magnitude
    float min = FLT_MAX, max = -FLT_MAX;

    for (i = 0; i < (int) height; ++i) {
        for (j = 0; j < (int) width; ++j) {
            float re, im;
            re = tmpBuffer[i * (width * 2) + j * 2];
            im = tmpBuffer[i * (width * 2) + j * 2 + 1];

            float mag = logf(sqrtf(re * re + im * im) + 1);
            tmpBuffer[i * width + j] = mag;

            min = mag < min ? mag : min;
            max = mag > max ? mag : max;
        }
    }

    for (i = 0; i < (int) height; ++i) {
        for (j = 0; j < (int) width; ++j) {
            tmpBuffer[i * width + j] = (tmpBuffer[i * width + j] - min) / (max - min);
        }
    }

    // step13.3 Shift the spectrum so that DC is at center
    for (i = 0; i < (int) height; ++i) {
        for (j = 0; j < (int) cx; ++j) {
            float a = tmpBuffer[i * width + j];

            /* quadrant 0? */
            if (i < cy) {
                /* swap it with quadrant 3 */
                tmpBuffer[i * width + j] = tmpBuffer[(i + cy) * width + (j + cx)];
                tmpBuffer[(i + cy) * width + (j + cx)] = a;
            }
                /* quadrant 2? */
            else {
                /* swap it with quadrant 1*/
                tmpBuffer[i * width + j] = tmpBuffer[(i - cy) * width + (j + cx)];
                tmpBuffer[(i - cy) * width + (j + cx)] = a;
            }
        }
    }

    // step14 将取出的数据转换为OpenCV格式并保存
    // tmpBuffer波动范围在0到1之间
    Mat out_mat(height, width, CV_32F, tmpBuffer);
    out_mat = out_mat * 255;
    imwrite("../test-data/out_fft.jpg", out_mat);

    // step15 解除对对象的锁定
    vpiImageUnlock(spectrum);
    // -------------------------------------------------------------------

    // 第三阶段：清理资源
    // -------------------------------------------------------------------
    // step16 销毁相关资源
    vpiStreamDestroy(stream);
    vpiImageDestroy(input);
    vpiImageDestroy(inputF32);
    vpiImageDestroy(spectrum);
    free(tmpBuffer);
    // -------------------------------------------------------------------

    return 0;
}