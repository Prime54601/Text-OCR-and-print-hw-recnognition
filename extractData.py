import json

def extract_rec_data(json_file_path):
    """
    从JSON文件中提取rec_boxes和rec_texts数据
    
    Args:
        json_file_path (str): JSON文件路径
    
    Returns:
        tuple: (rec_boxes_list, rec_texts_list)
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 提取rec_boxes和rec_texts
    rec_boxes_list = data.get('rec_boxes', [])
    rec_texts_list = data.get('rec_texts', [])
    
    return rec_boxes_list, rec_texts_list

# 示例用法
if __name__ == "__main__":
    # 文件路径
    json_file_path = "/mnt/data/Class Projects/大一上 工程学导论/AI组学习资料/test projects/output/result.json"
    
    # 提取数据
    rec_boxes, rec_texts = extract_rec_data(json_file_path)
    
    # 打印结果
    print("rec_boxes:")
    for i, box in enumerate(rec_boxes):
        print(f"  Box {i}: {box}")
    
    print("\nrec_texts:")
    for i, text in enumerate(rec_texts):
        print(f"  Text {i}: '{text}'")
    
    # 验证数据一致性
    print(f"\n总共找到 {len(rec_boxes)} 个边界框和 {len(rec_texts)} 个文本项")