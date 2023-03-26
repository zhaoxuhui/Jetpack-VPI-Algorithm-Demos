import cv2
import vpi  # 导入VPI包
import numpy as np

if __name__ == '__main__':
    img_path = "../test-data/img2.png"
    templ_path = "../test-data/img2-template.png"

    # step1 利用OpenCV读取影像
    src_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    templ_img = cv2.imread(templ_path, cv2.IMREAD_GRAYSCALE)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(src_img)
    templ = vpi.asimage(templ_img)

    # step3 执行操作
    with vpi.Backend.CUDA:
        output = vpi.templateMatching(input, templ)

    # step4 结果输出
    with output.rlock_cpu() as outData:
        outData = outData * 255
        cv2.imwrite("../test-data/out_templ.jpg", outData)
