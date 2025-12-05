from openpyxl import Workbook
import os

from ocr_infer import *
from extractData import extract_rec_data

#========================================初=====始=====化===============================================
json_path = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/result.json"
# cam = init_cam()
# ocr = PaddleOCR(use_textline_orientation=True, use_doc_unwarping=False, lang="ch")

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

while 1:
    # ocr_inf(cam, ocr)
    rec_boxes, rec_texts = extract_rec_data(json_path)
    wb_body = []
    # print("boxes: ", rec_boxes)
    if rec_boxes == []: #若没有找到文字，代表本批表单已经识别完，退出循环
        print("No more data to append. Aborting!")
        break
    for box, text in zip(rec_boxes, rec_texts):
        if(is_hand_written == False): #不是手写文字，存入表头
            wb_head.append(text)
        if(is_hand_written == True): #手写文字存入表身
            wb_body.append(text)

    if(first_time == True): #表头只导入一次
        ws.append(wb_head)
        first_time = False
    ws.append(wb_body)
    break

wb.save(path_to_file)