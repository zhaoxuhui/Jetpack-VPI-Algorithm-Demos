import cv2
import vpi  # 导入VPI包

if __name__ == '__main__':
    img_path = "../test-data/img1.png"

    # step1 利用OpenCV读取影像
    img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(img1)

    # step3 执行操作
    with vpi.Backend.CPU:
        min_coords, max_coords = input.minmaxloc(min_capacity=10000, max_capacity=10000)

    # step4 输出结果
    with input.rlock_cpu() as in_data, min_coords.rlock_cpu() as min_data, max_coords.rlock_cpu() as max_data:
        min_loc = tuple(min_data[0].astype(int)[::-1])
        max_loc = tuple(max_data[0].astype(int)[::-1])

        min_value = in_data[min_loc]
        max_value = in_data[max_loc]

        print(min_loc, min_value)
        print(max_loc, max_value)
