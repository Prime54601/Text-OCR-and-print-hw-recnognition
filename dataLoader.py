import os
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

current_dir = Path(__file__).parent

def pad_to_square(image, fill_value=255):
    """
    将图像填充为正方形，保持原始比例
    """
    h, w = image.shape
    max_dim = max(h, w)
    
    # 创建正方形画布
    padded_image = np.full((max_dim, max_dim), fill_value, dtype=image.dtype)
    
    # 计算居中位置
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    
    # 将原图放在中心
    padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return padded_image

def load_hw_data(data_dir_hw = f"{current_dir}/HWDB1.1tst_gnt", trn_count_hw = 10000, val_count_hw = 1000):
    hwTrain = []
    hwVal = []
    cnt = 0
    maxcnt = trn_count_hw + val_count_hw
    for file_name in os.listdir(data_dir_hw):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(data_dir_hw, file_name)
            with open(file_path, 'rb') as f:
                header_size = 10
                while True:
                    header = np.fromfile(f, dtype='uint8', count=header_size)
                    if not header.size:
                        break

                    sample_size = header[0].astype(np.uint32) + (header[1].astype(np.uint32) << 8) + (header[2].astype(np.uint32) << 16) + (header[3].astype(np.uint32) << 24)
                    tagcode = header[5].astype(np.uint16) + (header[4].astype(np.uint16) << 8)
                    width = header[6].astype(np.uint16) + (header[7].astype(np.uint16) << 8)
                    height = header[8].astype(np.uint16) + (header[9].astype(np.uint16) << 8)

                    pixel_count = width.astype(np.uint16) * height.astype(np.uint16)

                    if header_size + pixel_count != sample_size:
                        break
                    image = np.fromfile(f, dtype='uint8', count=pixel_count).reshape((height, width))
                    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
                    # cv2.imshow("binary_image", binary_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # raiseerror
                    if cnt < trn_count_hw:
                        binary_image = pad_to_square(binary_image)
                        hwTrain.append((binary_image, 1))
                    else:
                        hwVal.append((binary_image, 1))
                    cnt += 1
                    if cnt >= maxcnt:
                        break
    return hwTrain, hwVal

#==================printed chars===============

# ================= 配置区域 =================
# 字体文件路径 (请修改为你电脑上的实际路径)
FONT_PATH = "/usr/local/share/fonts/h/HarmonyOS_Sans_SC_Medium.ttf"  

# 图片参数
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
BG_COLOR = 255  # 白色背景

# 字体参数
FONT_SIZE = 80
TEXT_COLOR = 0      # 黑色文字

# 生成数量
NUM_IMAGES = 1
# ===========================================

def generate_images_numpy(trn_count = 10000, val_count = 1000):
    """
    生成指定数量的汉字图片，并以 NumPy 数组形式返回。
    
    Args:
        num (int): 生成图片的数量
        font_path (str): 字体文件的路径
        
    Returns:
        tuple: (images_array, labels_list)
            - images_array: shape 为 (num, height, width, 3) 的 uint8 数组
            - labels_list: 包含对应汉字的列表
    """
    
    # 检查字体文件
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"找不到字体文件: {FONT_PATH}，请检查路径配置。")

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception as e:
        raise Exception(f"字体加载失败: {e}")
    
    num = trn_count + val_count

    #print(f"正在生成 {num} 张图片数据 (内存中)...")

    trn_list = []
    val_list = []

    for i in range(num):
        char = chr(random.randint(0x4E00, 0x9FA5))
        
        # 1. 创建图片
        image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), BG_COLOR)
        draw = ImageDraw.Draw(image)

        # 2. 计算居中位置
        bbox = draw.textbbox((0, 0), char, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (IMAGE_WIDTH - text_w) / 2 - bbox[0]
        y = (IMAGE_HEIGHT - text_h) / 2 - bbox[1]

        # 3. 绘制文字
        draw.text((x, y), char, font=font, fill=TEXT_COLOR)

        # 4. 转换为 Numpy 数组
        # PIL image 转 numpy 默认 shape 是 (H, W, C)
        img_array = np.array(image)
        
        if i < trn_count:
            trn_list.append((img_array, 0))
        else:
            val_list.append((img_array, 0))

    # 5. 将列表堆叠成一个大的 numpy 数组
    # 最终 shape: (num, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    #final_array = np.array(image_list, dtype=np.uint8)
    
    return trn_list, val_list

if __name__ == '__main__': #1 for handwritten, 0 for printed
    trn, val = load_hw_data(trn_count_hw = 70)
    trn1, val1 = generate_images_numpy(trn_count = 30, val_count = 10)
    print(trn[0][0])
    # trn[0] is image, trn[1] is tag
    # print(trn[0][0])
    # print(trn[0][1])
