import cv2
import vpi  # 导入VPI包

if __name__ == '__main__':
    img_path = "../test-data/img1.png"

    # step1 利用OpenCV读取影像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(img)

    # step4 执行操作
    with vpi.Backend.CUDA:
        output = input.rescale((input.width * 2 // 3, input.height * 3 // 2), interp=vpi.Interp.LINEAR,
                               border=vpi.Border.ZERO)

    # step5 结果输出
    with output.rlock_cpu() as outData:
        cv2.imwrite("../test-data/out_rescale.jpg", outData)