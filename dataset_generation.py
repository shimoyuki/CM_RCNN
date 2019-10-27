"""
CM R-CNN
Tools for imitate fake images and extract json file of them.

Written by Lycoris
"""

import os
from skimage import data, exposure
from matplotlib import pyplot as plt
import cv2
import json
import shutil
import random
# from multiprocessing import Pool
import numpy as np
import tools

# Define the input and output img
mask_dir = '/mnt/sda1/hxc/data/mask/'
org_dir = '/mnt/sda1/hxc/data/org/'
back_dir = '/mnt/sda1/hxc/data/background/'
imit_dir = '/mnt/sda1/hxc/data/imitation/'
train_dir = '/mnt/sda1/hxc/data/train/'
val_dir = '/mnt/sda1/hxc/data/val/'

# image_list = os.listdir(org_dir)
# image_list = [os.path.join(org_dir, _) for _ in image_list]


# 导出带标记类型的单张图片的标记json文件
# @name 掩模图文件绝对路径
# @org_data 原图像的包含文件名信息的json标记文件，via格式
# @type 目标区域的标签名
def extract_edge_with_type(name, org_data, type):
    file_name = name.split('/')[-1]
    if 'jpg' not in file_name:
        return
    print('Dealing img ' + file_name + '……')
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    _, thres = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('Find %d contours.' % len(contours))

    for key in org_data.keys():
        if key.find(file_name) == 0:
            json_body = org_data[key]["regions"]
            offset = len(json_body)

            for i in range(0, len(contours)):
                size = cv2.contourArea(contours[i])
                print('Contour size: %d.' % size)
                if size < 50:
                    continue
                epsilon = 0.01 * cv2.arcLength(contours[i], True)
                approx = cv2.approxPolyDP(contours[i], epsilon, True)
                x_coord = approx[:, 0][:, 0].tolist()
                x_coord.append(x_coord[0])
                y_coord = approx[:, 0][:, 1].tolist()
                y_coord.append(y_coord[0])

                # Put into json part
                shape_attributes = {"name": "polygon"}
                shape_attributes["all_points_x"] = x_coord
                shape_attributes["all_points_y"] = y_coord
                
                region_attributes = {"class_name": type}
                sub_body = {"shape_attributes": shape_attributes, "region_attributes": region_attributes}
                json_body[str(i + offset)] = sub_body
    return


# 使用正负样本合成数据集
def imitate(num):
    back_names = random.sample(os.listdir(back_dir), num)
    for back_name in back_names:
        print('Using negative sample: ' + back_name + '……')
        org_name = random.sample(os.listdir(org_dir), 1)[0]
        print('Using positive sample: ' + org_name + '……')

        img_mask = cv2.imread(mask_dir + org_name, cv2.IMREAD_GRAYSCALE)#掩模图
        img_org = cv2.imread(org_dir + org_name, cv2.IMREAD_COLOR)#实例图
        img_back = cv2.imread(back_dir + back_name, cv2.IMREAD_COLOR)#背景图
        if img_back is None or img_org is None or img_mask is None:
            print("Bad image data.")
            continue
        img_back = cv2.resize(img_back, (img_mask.shape[1], img_mask.shape[0]))
        #取实例轮廓
#         _, thres = cv2.threshold(img_mask, 50, 255, cv2.THRESH_BINARY)
#         _, contours, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for i in range(img_org.shape[0]):
#             for j in range(img_org.shape[1]):
#                 for k in range(len(contours)):
#                     if cv2.pointPolygonTest(contours[k], (i,j), False) >= 0:
#                         img_back[i][j] = img_org[i][j]
#                         #         imit_name = 'imit_' + org_name.split('.')[0] + '_' + back_name.split('.')[0] + '.jpg'
#         imit_name = 'imit_' + org_name.split('.')[0] + '_' + back_name.split('.')[0] + '.jpg'
#         print('Generating imitate sample: ' + imit_name + '……')
#         if not os.path.exists(imit_dir + 'org'):
#             os.makedirs(imit_dir + 'org')
#         cv2.imwrite(imit_dir + 'org/' + imit_name, img_back)
        #轮廓内部填充为白色
#         cv2.drawContours(img_back, contours, -1, (255, 255, 255), -1)
        
        #取轮廓的近似多边形
#         for i in range(len(contours)):
#             cnt = contours[i]
#             epsilon = 0.01 * cv2.arcLength(cnt,True)
#             approx = cv2.approxPolyDP(cnt, epsilon, True)
#             cv2.polylines(img_org, [approx], True, (255, 255, 255), 1)
#             plt.figure()
#             plt.imshow(img_org)
#             plt.show()

        #根据掩模图用实例替换背景图对应区域
        for i in range(img_mask.shape[0]):
            for j in range(img_mask.shape[1]):
                if img_mask[i][j] == 255:
                    if (img_back[i][j] < np.array([10, 10, 10])).all():
                        img_mask[i][j] = 0
                    else:
                        img_back[i][j] = img_org[i][j]
  
        imit_name = 'imit_' + back_name.split('.')[0] + '.jpg'
        print('Generating imitate sample: ' + imit_name + '……')
        if not os.path.exists(imit_dir + 'org'):
             os.makedirs(imit_dir + 'org')
        if not os.path.exists(imit_dir + 'mask'):
             os.makedirs(imit_dir + 'mask')
        cv2.imwrite(imit_dir + 'org/' + imit_name, img_back)
        cv2.imwrite(imit_dir + 'mask/' + imit_name, img_mask)
        tools.move_file(back_dir + back_name, back_dir + 'imitated/' + back_name)
    return


# 删除无标记的正样本图
def delete_extra():
    with open(back_dir + 'trainLabels.csv') as file:
        for line in file:
            print(line)
            info = line.split(',')
            if int(info[1]) != 0:
                print('delete file ' + info[0] + '.jpeg')
                os.remove(back_dir + info[0] + '.jpeg')
    return


# 按概率分离数据集
def split_dataset(rate):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    files = os.listdir(org_dir)
    for name in files:
        tools.copy_file(org_dir + name, train_dir + name)
    file_num = len(files)
    pick_num = int(file_num * rate)
    print(file_num)
    print(pick_num)
    sample = random.sample(files, pick_num)
    for name in sample:
        tools.move_file(train_dir + name, val_dir + name)
    return


if __name__ == '__main__':
    # delete_extra()

    # imitate(2000)

    # split_dataset(0.1)

    mask_json_path = os.path.join(mask_dir, 'via_region_data.json')
    train_json_path = os.path.join(train_dir, 'via_region_data.json')
    val_json_path = os.path.join(val_dir, 'via_region_data.json')

    image_list = os.listdir(train_dir)
    image_list = [os.path.join(mask_dir, _) for _ in image_list]
    with open(train_json_path, mode='r', encoding='UTF-8') as train_json_file:
        train_data = json.load(train_json_file)
    for name in image_list:
        extract_edge_with_type(name, train_data, "hemorrhage")
    with open(train_json_path, mode='w', encoding='UTF-8') as train_json_file:
        json.dump(train_data, train_json_file, ensure_ascii=False)

    # image_list = os.listdir(val_dir)
    # image_list = [os.path.join(mask_dir, _) for _ in image_list]
    # with open(val_json_path, mode='r', encoding='UTF-8') as val_json_file:
    #     val_data = json.load(val_json_file)
    # for name in image_list:
    #     extract_edge_with_type(name, val_data, "hemorrhage")
    # with open(val_json_path, mode='w', encoding='UTF-8') as val_json_file:
    #     json.dump(val_data, val_json_file, ensure_ascii=False)
