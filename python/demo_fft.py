import cv2
import vpi  # 导入VPI包
import numpy as np

if __name__ == '__main__':
    img_path = "../test-data/img1.png"

    # step1 利用OpenCV读取影像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # step2 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(img)

    # step4 执行操作
    with vpi.Backend.CUDA:
        output = input.convert(vpi.Format.F32).fft()

    # step5 变换后的后处理，以方便可视化
    hfreq = output.cpu().view(dtype=np.complex64).squeeze(2)

    if input.width % 2 == 0:
        wpad = input.width // 2 - 1
        padmode = 'reflect'
    else:
        wpad = input.width // 2
        padmode = 'symmetric'

    freq = np.pad(hfreq, ((0, 0), (0, wpad)), mode=padmode)
    freq[:, hfreq.shape[1]:] = np.conj(freq[:, hfreq.shape[1]:])
    freq[1:, hfreq.shape[1]:] = freq[1:, hfreq.shape[1]:][::-1]

    lmag = np.log(1 + np.absolute(freq))
    spectrum = np.fft.fftshift(lmag)

    # step6 结果输出
    max_v = np.max(spectrum)
    min_v = np.min(spectrum)
    scale = 255 / (max_v - min_v)
    spectrum = scale * (spectrum - min_v)
    cv2.imwrite("../test-data/out_fft.jpg", spectrum)
