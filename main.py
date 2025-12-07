from openpyxl import Workbook
import os

from ocr_infer import *
from extractData import extract_rec_data
from infer import *

#========================================初=====始=====化===============================================
json_path = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/result.json"
cam = init_cam()
ocr = PaddleOCR(use_textline_orientation=True, use_doc_unwarping=False, lang="ch")

file = 'result.xlsx'
path= './大一上 工程学导论/AI组学习资料/test projects/output'
path_to_file = os.path.join(path, file)
wb = Workbook()
ws = wb.active
ws.title = '表单识别结果'
is_hand_written = False
first_time = True
wb_head = []
wb_body = []
#======================================================================================================


if __name__ == "__main__":
    while 1:
        ocr_inf(cam, ocr) #ocr识别并将结果存储在json文件中
        rec_boxes, rec_texts = extract_rec_data(json_path)
        wb_body = []
        # print("boxes: ", rec_boxes)
        if rec_boxes == []: #若没有找到文字，代表本批表单已经识别完，退出循环
            print("No more data to append. Aborting!")
            break
        for box, text in zip(rec_boxes, rec_texts):
            # print(box)
            # for i in range(len(box)):
            #     box[i] *= 2
            # print(box)
            print(box)
            image = getimage(box, "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/image.jpg")
            image = np.array(image)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if model_infer(box) == 1: #判断是否为手写文字
                is_hand_written = True
            else:
                is_hand_written = False

            print(text, is_hand_written) #调试用，正式使用时记得注释
            if(is_hand_written == False): #不是手写文字，存入表头
                wb_head.append(text)
            if(is_hand_written == True): #手写文字存入表身
                wb_body.append(text)

        if(first_time == True): #表头只导入一次
            ws.append(wb_head)
            first_time = False
        ws.append(wb_body)
        break #测试用，只处理一张图片

    # os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
    print(path_to_file)
    wb.save(path_to_file)