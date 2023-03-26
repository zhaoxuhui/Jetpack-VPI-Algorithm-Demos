import cv2
import vpi  # 导入VPI包

if __name__ == '__main__':
    img_path = "../test-data/img1.png"

    # step1 利用OpenCV读取影像
    img1 = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(img1)

    outputs = [vpi.Image(input.size, vpi.Format.U8),
               vpi.Image(input.size, vpi.Format.U8),
               vpi.Image(input.size, vpi.Format.U8)]

    # step3 执行操作
    with vpi.Backend.CPU:
        vpi.mixchannels([input], outputs, [0, 1, 2], [0, 1, 2])

    # step4 结果输出
    for i in range(len(outputs)):
        output = outputs[i]
        with output.rlock_cpu() as outData:
            cv2.imwrite("../test-data/out_band_" + str(i + 1) + ".jpg", outData)