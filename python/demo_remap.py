import cv2
import numpy as np
import vpi  # 导入VPI包

if __name__ == '__main__':
    img_path = "../test-data/img1.png"

    # step1 利用OpenCV读取影像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(img)

    # step3 创建warp map
    warp = vpi.WarpMap(vpi.WarpGrid(input.size))
    wx, wy = np.asarray(warp).transpose(2, 1, 0)

    x = wx - input.width / 2
    y = wy - input.height / 2

    R = input.height / 8  # planet radius
    r = np.sqrt(x * x + y * y)

    theta = np.pi + np.arctan2(y, x)
    phi = np.pi / 2 - 2 * np.arctan2(r, 2 * R)

    wx[:] = np.fmod((theta + np.pi) / (2 * np.pi) * (input.width - 1), input.width - 1)
    wy[:] = (phi + np.pi / 2) / np.pi * (input.height - 1)

    # step4 执行操作
    with vpi.Backend.CUDA:
        output = input.remap(warp)

    # step5 结果输出
    with output.rlock_cpu() as outData:
        cv2.imwrite("../test-data/out_remap.jpg", outData)