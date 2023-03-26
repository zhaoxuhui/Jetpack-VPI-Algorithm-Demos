import cv2
import vpi  # 导入VPI包

if __name__ == '__main__':
    img_path = "../test-data/img1.png"

    # step1 利用OpenCV读取影像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(img)

    # step3 执行操作
    gaussian = vpi.Pyramid(input.size, input.format, 4)
    with vpi.Backend.CUDA:
        output = input.laplacian_pyramid(4, out_gaussian=gaussian)
