import cv2
import vpi  # 导入VPI包
import numpy as np

if __name__ == '__main__':
    img_path = "../test-data/img1.png"

    # step1 利用OpenCV读取影像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(img)

    # step3 构造变换关系
    grid = vpi.WarpGrid(input.size)
    sensorWidth = 22.2  # APS-C sensor
    focalLength = 7.5
    f = focalLength * input.width / sensorWidth

    K = [[f, 0, input.width / 2],
         [0, f, input.height / 2]]
    X = np.eye(3, 4)

    warp = vpi.WarpMap.fisheye_correction(grid, K=K, X=X,
                                          mapping=vpi.FisheyeMapping.EQUIDISTANT,
                                          coeffs=[-0.126, 0.004])

    # step4 执行操作
    with vpi.Backend.CUDA:
        output = input.remap(warp, interp=vpi.Interp.CATMULL_ROM, border=vpi.Border.ZERO)

    # step5 结果输出
    with output.rlock_cpu() as outData:
        cv2.imwrite("../test-data/out_dist.jpg", outData)