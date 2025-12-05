import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

def get_random_chinese_char():
    """生成一个随机的中文字符"""
    val = random.randint(0x4E00, 0x9FA5)
    return chr(val)

def generate_images_numpy(num, font_path):
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
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"找不到字体文件: {font_path}，请检查路径配置。")

    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except Exception as e:
        raise Exception(f"字体加载失败: {e}")

    print(f"正在生成 {num} 张图片数据 (内存中)...")

    image_list = []
    label_list = []

    for i in range(num):
        char = get_random_chinese_char()
        
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
        
        image_list.append(img_array)
        label_list.append(char)

    # 5. 将列表堆叠成一个大的 numpy 数组
    # 最终 shape: (num, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    final_array = np.array(image_list, dtype=np.uint8)
    
    return final_array, label_list

if __name__ == "__main__":
    try:
        # 调用函数
        images_data, labels = generate_images_numpy(NUM_IMAGES, FONT_PATH)
        
        # 打印结果信息
        print("\n生成完毕！")
        print(f"数组形状 (Shape): {images_data.shape}")  # 预期输出: (10, 128, 128, 3)
        print(f"数组类型 (Dtype): {images_data.dtype}")  # 预期输出: uint8
        print(f"标签前5个: {labels[:5]}")
        print(images_data[0])
        
        # 如果需要，你可以在这里直接将数据用于模型训练
        # 例如: model.fit(images_data, labels, ...)
        
    except Exception as e:
        print(f"运行出错: {e}")