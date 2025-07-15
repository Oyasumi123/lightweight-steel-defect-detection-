import os
import glob
import xml.etree.ElementTree as ET


# 获取类别名称和数量
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    print("Classes loaded:", class_names)
    return class_names, len(class_names)


# 转换坐标
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 转换XML到YOLO格式
def convert_xml_to_yolo(xml_root_path, txt_save_path, classes_path):
    print("XML root path:", xml_root_path)
    print("TXT save path:", txt_save_path)
    print("Classes path:", classes_path)

    if not os.path.exists(txt_save_path):
        os.makedirs(txt_save_path)
        print("Directory created:", txt_save_path)

    xml_paths = glob.glob(os.path.join(xml_root_path, '*.xml'))
    print("XML files found:", xml_paths)

    classes, _ = get_classes(classes_path)

    for xml_id in xml_paths:
        print("Processing file:", xml_id)
        txt_id = os.path.join(txt_save_path, os.path.basename(xml_id)[:-4] + '.txt')
        txt = open(txt_id, 'w')
        xml = open(xml_id, encoding='utf-8')
        tree = ET.parse(xml)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') is not None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            print("Class found:", cls)
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('xmax').text)),
                 int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('ymax').text)))
            box = convert((w, h), b)
            txt.write(str(cls_id) + ' ' + ' '.join([str(a) for a in box]) + '\n')
        txt.close()
        print("TXT file created:", txt_id)


if __name__ == '__main__':
    # 修改为你的XML文件路径
    xml_root_path = r"F:\model\liyolonet\liyolo\my-data\GC10-DET\lable"
    # 修改为你的TXT文件保存路径
    txt_save_path = r"F:\model\liyolonet\liyolo\my-data\GC10-DET\labels"
    # 修改为你的类别文件路径
    classes_path = r"F:\model\liyolonet\liyolo\my-data\GC10-DET\labels.txt"

    convert_xml_to_yolo(xml_root_path, txt_save_path, classes_path)


# if __name__ == "__main__":
#     # 把forklift_pallet的voc的xml标签文件转化为yolo的txt标签文件
#     # 1、需要转化的类别
#     classes = ['6_siban']#6_siban 4_shuiban 8_yahen 1_chongkong 5_youban 7_yiwu 10_yaozhed 2_hanfeng 9_zhehen 3_yueyawan
#     # 2、voc格式的xml标签文件路径
#     xml_files1 = r'my-data/GC10-DET/lable'
#     # 3、转化为yolo格式的txt标签文件存储路径
#     save_txt_files1 = r'my-data/GC10-DET/lab1'

