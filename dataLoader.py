import os
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

current_dir = Path(__file__).parent

IMAGE_DIM = 128

def pad_to_square(image, fill_value=255):
    """
    将图像填充为正方形，保持原始比例
    """
    h, w = image.shape
    max_dim = max(h, w)
    resize_factor = (IMAGE_DIM - 15) / max_dim
    image = cv2.resize(image, (int(resize_factor * w), int(resize_factor * h)))
    h, w = image.shape
    
    # 创建正方形画布
    padded_image = np.full((IMAGE_DIM, IMAGE_DIM), fill_value, dtype=image.dtype)
    
    # 计算居中位置
    y_offset = (IMAGE_DIM - h) // 2
    x_offset = (IMAGE_DIM - w) // 2
    
    # 将原图放在中心
    padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    return padded_image

HEADER_SIZE = 10

def load_hw_data(data_dir_hw = f"{current_dir}/HWDB1.1tst_gnt", trn_count = 10000, val_count = 1000):
    hwTrain = []
    hwVal = []
    cnt = 0
    maxcnt = trn_count + val_count
    for file_name in os.listdir(data_dir_hw):
        if file_name.endswith('.gnt'):
            file_path = os.path.join(data_dir_hw, file_name)
            with open(file_path, 'rb') as f:
                # 添加文件内样本计数器
                file_sample_count = 0
                # 限制每个文件的最大迭代次数，防止死循环
                max_iterations_per_file = 100000
                iterations = 0
                
                # header_size = 10
                while iterations < max_iterations_per_file and cnt < maxcnt:
                    iterations += 1
                    seqlen = random.randint(1, 7)
                    join_image = np.full((IMAGE_DIM, IMAGE_DIM * seqlen), 255, dtype="uint8")
                    j = 0
                    ptr = 0
                    reach_file_end = False
                    while j < seqlen and cnt < maxcnt:
                        header = np.fromfile(f, dtype='uint8', count=HEADER_SIZE)
                        if not header.size:
                            reach_file_end = True
                            break
                        discard = random.randint(0, 3)
                        if discard != 0:
                            width = header[6].astype(np.uint16) + (header[7].astype(np.uint16) << 8)
                            height = header[8].astype(np.uint16) + (header[9].astype(np.uint16) << 8)
                            np.fromfile(f, dtype='uint8', count=width*height)
                            continue
                        sample_size = header[0].astype(np.uint32) + (header[1].astype(np.uint32) << 8) + (header[2].astype(np.uint32) << 16) + (header[3].astype(np.uint32) << 24)
                        # tagcode = header[5].astype(np.uint32) + (header[4].astype(np.uint32) << 8)
                        width = header[6].astype(np.uint16) + (header[7].astype(np.uint16) << 8)
                        height = header[8].astype(np.uint16) + (header[9].astype(np.uint16) << 8)

                        pixel_count = width.astype(np.uint16) * height.astype(np.uint16)

                        if HEADER_SIZE + pixel_count != sample_size:
                            reach_file_end = True
                            break
                        image = np.fromfile(f, dtype='uint8', count=pixel_count).reshape((height, width))
                        # cv2.imshow("image", image)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        # err
                        image = pad_to_square(image)
                        join_image[:, ptr:ptr + IMAGE_DIM] = image
                        ptr += IMAGE_DIM
                        j += 1

                    if reach_file_end:
                        # 如果到达文件末尾，跳出当前文件处理循环
                        break

                    noise = np.random.normal(0, 0.65, join_image.shape)
                    noisy_image = join_image + noise
                    # 确保像素值在有效范围内
                    join_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                    binary_image = cv2.adaptiveThreshold(join_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
                    binary_image = cv2.resize(binary_image, (128, 128))
                    # cv2.imshow("binary_image", binary_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # raiseerror
                    if cnt < trn_count:   
                        hwTrain.append((binary_image, 1))
                    else:
                        hwVal.append((binary_image, 1))
                    cnt += 1
                    file_sample_count += 1
                    if cnt >= maxcnt:
                        return hwTrain, hwVal
                # 如果单个文件处理了太多样本仍未达到目标数量，跳出该文件
                if iterations >= max_iterations_per_file:
                    print(f"Warning: Max iterations reached for file {file_name}, moving to next file")
    return hwTrain, hwVal

#==================printed chars===============

# ================= 配置区域 =================
# 字体文件路径 (请修改为你电脑上的实际路径)
font_paths  = ["/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/fonts/SimHei.ttf",
               "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/fonts/SimSun.ttf"]

# 图片参数
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

# 字体参数
FONT_SIZE_MIN = 70  # 最小字体大小
FONT_SIZE_MAX = 90  # 最大字体大小
# ===========================================

def generate_images_numpy(trn_count = 10000, val_count = 1000):
    
    # 检查字体文件
    for path in font_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到字体文件: {path}，请检查路径配置。")
    
    num = trn_count + val_count

    trn_list = []
    val_list = []

    for i in range(num):
        index = random.randint(0, len(font_paths) - 1)
        path = font_paths[index]
        
        # 随机选择字体粗细（通过调整字体大小实现）
        font_size = random.randint(FONT_SIZE_MIN, FONT_SIZE_MAX)
        try:
            font = ImageFont.truetype(path, font_size)
        except Exception as e:
            raise Exception(f"字体加载失败: {e}")
        seqlen = random.randint(1, 7)
        char_list = []
        for j in range(seqlen): #加载随机长度的字符组
            char = chr(random.randint(0x4E00, 0x9FA5))
            char_list.append(char)
        
        # 1. 创建图片
        image_width = int((seqlen * (font_size + 15) - 10) * 1.5)
        x_offset = random.randint(-20, 20) #随机偏移
        y_offset = random.randint(-20, 20)
        image = Image.new('L', (image_width, int(font_size * 1.5)), 255) #空白画布，背景颜色为255
        draw = ImageDraw.Draw(image)

        # 2. 计算居中位置
        text_w = []
        text_h = 0
        bbox = draw.textbbox((0, 0), char_list[0], font=font)
        init_width = bbox[0]
        init_height = bbox[1]
        for char in char_list:
            bbox = draw.textbbox((0, 0), char, font=font)
            text_w.append(bbox[2] - bbox[0])  #一个字的宽度
            text_h = max(text_h, bbox[3]-bbox[1])
        x = (image_width - sum(text_w) - 15 * seqlen + 15) / 2 - init_width + x_offset #计算字符串左边的边界
        y = (int(font_size * 1.5) - text_h) / 2 - init_height + y_offset
        # 3. 绘制文字
        for char in char_list:
            draw.text((x, y), char, font=font, fill=0) #字的颜色为0
            x += text_w[char_list.index(char)] + 15

        # 4. 转换为 Numpy 数组
        # PIL image 转 numpy 默认 shape 是 (H, W, C)
        img_array = np.array(image)

        # 添加透视变换效果，模拟倾斜视角
        rows, cols = img_array.shape
        # 定义原始点和目标点，创建透视变换矩阵
        # 随机生成一些偏移量来模拟不同角度的拍摄效果
        shift_range = 12  # 控制变形程度
        pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        pts2 = np.float32([
            [random.uniform(0, shift_range), random.uniform(0, shift_range)],
            [cols - random.uniform(0, shift_range), random.uniform(0, shift_range)],
            [random.uniform(0, shift_range), rows - random.uniform(0, shift_range)],
            [cols - random.uniform(0, shift_range), rows - random.uniform(0, shift_range)]
        ])
        
        # 计算透视变换矩阵并应用
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img_array = cv2.warpPerspective(img_array, M, (cols, rows), flags=cv2.INTER_LINEAR, borderValue=255)

        # 添加噪声
        noise = np.random.normal(0, 0.65, img_array.shape)
        noisy_image = img_array + noise
        img_array = np.clip(noisy_image, 0, 255).astype(np.uint8) # 确保像素值在有效范围内
        binary_image = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2) #二值化以匹配实际推理用图片
        resized_image = cv2.resize(binary_image, (IMAGE_WIDTH, IMAGE_HEIGHT)) #缩放以适应模型输入

        # cv2.imshow("binary_image", resized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # raiseerror
        
        if i < trn_count:
            trn_list.append((resized_image, 0))
        else:
            val_list.append((resized_image, 0))
    
    return trn_list, val_list

if __name__ == '__main__':
    trn, val = load_hw_data(trn_count = 30, val_count = 10)
    trn1, val1 = generate_images_numpy(trn_count = 30, val_count = 10)
    # [0]是图片，[1]是标签
    # 标签1为手写，0为印刷