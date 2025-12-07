import cv2
import numpy as np
from paddleocr import PaddleOCR
import time

def init_cam():
    camera = cv2.VideoCapture(4, cv2.CAP_V4L2)#可能要改成1？
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    if not camera.isOpened():
        print("无法打开摄像头")
        exit()

    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 设置手动曝光，因为自动曝光默认过曝并且收敛太慢
    camera.set(cv2.CAP_PROP_EXPOSURE, -100)  # 曝光值，通常负值降低曝光
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
    camera.set(cv2.CAP_PROP_FPS, 15)
    return camera

def ocr_preprocess(image):
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # image = cv2.convertScaleAbs(image, alpha=1, beta=-100)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转换为BW
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0) #增加对比度
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 4) #二值化
    image = cv2.medianBlur(image, 3) #去噪
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) #转换回“彩色”图像来防止paddle报错。。。
    cv2.imwrite("/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/image.jpg", image)
    # image = cv2.resize(image, (2000, 1500))
    return image

def ocr_inf(camera, ocr):
    image = camera.read()[1]
    '''
    image_path = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/test images/WIN_20251110_20_03_44_Pro.jpg" #或许可以换成别的？
    image = cv2.imread(image_path) #读取图片
    '''
    image = ocr_preprocess(image)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    result = ocr.predict(image)
    for res in result:
        res.print()
        # res.save_to_img("/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/result.jpg")
        res.save_to_json("/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/result.json")

if __name__ == "__main__":
    camera = init_cam()
    # time.sleep(5)
    # image = camera.read()[1]
    # image = ocr_preprocess(image)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ocr = PaddleOCR(use_textline_orientation=True, use_doc_unwarping=False, lang="ch")
    ocr_inf(camera, ocr)
    # ocr_inf(None, ocr)