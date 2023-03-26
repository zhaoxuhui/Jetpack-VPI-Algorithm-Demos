import cv2
import vpi  # 导入VPI包

if __name__ == '__main__':
    img_path1 = "../test-data/chair_stereo_left_grayscale.png"
    img_path2 = "../test-data/chair_stereo_right_grayscale.png"

    # step1 利用OpenCV读取影像
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    left = vpi.asimage(img1)
    right = vpi.asimage(img2)

    # step3 执行操作
    with vpi.Backend.CUDA:
        output = vpi.stereodisp(left, right, window=5, maxdisp=64) \
            .convert(vpi.Format.U8, scale=1.0 / (32 * 64) * 255)

    # step4 结果输出
    with output.rlock_cpu() as outData:
        cv2.imwrite("../test-data/out_disparity.jpg", outData)