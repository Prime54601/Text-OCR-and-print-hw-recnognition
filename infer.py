import torch
from PIL import Image
import torchvision.transforms as transforms
from extractData import extract_rec_data
from trainmodel import BinaryCNN
from dataLoader import *
import torchvision.transforms.functional as TF

json_dir = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/result.json"
image_dir = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/image.jpg"


def getimage(box, image_dir):
    # 这里应该返回一个PIL图像或者numpy数组
    image = Image.open(image_dir)
    image = image.crop(box)
    image = image.convert('L')
    # image.show()
    return image
    
def preprocess_image(image):
    """
    预处理图像以适应模型输入要求
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    # return transform(image).unsqueeze(0)  # 添加批次维度
    return transform(image)

def model_infer(box):
    model = BinaryCNN()
    model.load_state_dict(torch.load("/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/models/handwritten_vs_printed_cnn.pth"))
    model.eval()

    image = getimage(box, image_dir)
    image_cv = np.array(image)
    cv2.imshow("image", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    processed_image = preprocess_image(image)
    processed_image = processed_image.unsqueeze(0)

    with torch.no_grad():
        logits = model(processed_image)
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class # 1为手写，0为印刷


if __name__ == "__main__":
    json_dir = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/result.json"
    image_dir = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/image.jpg"
    box_list, text_list = extract_rec_data(json_dir)

    # 加载模型
    model = BinaryCNN()
    model.load_state_dict(torch.load("/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/models/handwritten_vs_printed_cnn.pth"))
    model.eval()

    # 获取图像
    image = getimage(box_list[4], image_dir)

    # image_cv = np.array(image)
    # cv2.imshow("image", image_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # err

    # 预处理图像
    processed_image = preprocess_image(image)

    image_pil = TF.to_pil_image(processed_image)
    image_cv = np.array(image_pil)
    # print(image_cv)
    # err
    cv2.imshow("image", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # raiseerror
    processed_image = processed_image.unsqueeze(0)

    # 执行推理
    with torch.no_grad():
        logits = model(processed_image)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1)

    # 输出详细结果
    print("推理完成:")
    print(f"原始logits: {logits}")
    print(f"概率分布: {probabilities}")
    print(f"预测类别: {predicted_class.item()}")

    # 类别映射
    class_names = ["印刷体", "手写体"]
    print(f"最终预测: {class_names[predicted_class.item()]}")