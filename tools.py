"""
CM R-CNN
Tools for file operation and presentation of image's infomation

Written by Lycoris
"""

import os
from skimage import data, exposure
from matplotlib import pyplot as plt
import cv2
import numpy as np
import json
import shutil
# from multiprocessing import Pool
import numpy as np

# Define the input and output img
split_dir = '/mnt/sda1/hxc/data/split/'
entire_dir = '/mnt/sda1/hxc/data/entire/'
mask_dir = '/mnt/sda1/hxc/data/mask/'
org_dir = '/mnt/sda1/hxc/data/org/'
equal_dir = '/mnt/sda1/hxc/data/equalYCrCb/'


# 图像归一化
def normalize(img, min_bound, max_bound):
    img.astype(np.float16)
    img = (img - min_bound) / (max_bound - min_bound)
    img[img > 1] = 1.
    img[img < 0] = 0.
    return img


# 标准分数归一化
def z_score(img):
    for i in range(img.shape[2]):
        mu = np.average(img[:, :, i])
        sigma = np.std(img[:, :, i])
        print(mu)
        print(sigma)
        img[:, :, i] = abs(img[:, :, i] - mu) / sigma
    return img


# 显示灰度图像信息
def print_image(file_name):
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    print(type(img))  # type() 函数如果只有第一个参数则返回对象的类型   在这里函数显示图片类型为 numpy类型的数组
    print(img.shape)
    # 图像矩阵的shape属性表示图像的大小，shape会返回tuple元组，
    # 第一个元素表示矩阵行数，第二个元组表示矩阵列数，第三个元素是3，表示像素值由光的三原色组成
    print(img.size)  # 返回图像的大小，size的具体值为shape三个元素的乘积
    print(img.dtype)  # 数组元素的类型通过dtype属性获得
    hist = cv2.calcHist([img], [0], None, [255], [1, 255])
    plt.figure()  # 新建一个图像
    plt.title('Grayscale Histogram')  # 图像的标题
    plt.xlabel('Bins')  # X轴标签
    plt.ylabel('# of Pixels')  # Y轴标签

    plt.plot(hist)  # 画图
    plt.xlim([0, 256])  # 设置x坐标轴范围
    plt.show()  # 显示图像
    # pixel_data = np.array(img)
    # for i in range(0, img.shape[0]):
    #     for j in range(0, img.shape[1]):
    #         if pixel_data[i, j] > 0:
    #             print(i, j, pixel_data[i, j])
    return


# 显示rgb图像信息
def print_rgb_image(file_name):
    print(file_name)
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(type(img))  # type() 函数如果只有第一个参数则返回对象的类型   在这里函数显示图片类型为 numpy类型的数组
    print(img.shape)
    # 图像矩阵的shape属性表示图像的大小，shape会返回tuple元组，
    # 第一个元素表示矩阵行数，第二个元组表示矩阵列数，第三个元素是3，表示像素值由光的三原色组成
    print(img.size)  # 返回图像的大小，size的具体值为shape三个元素的乘积
    print(img.dtype)  # 数组元素的类型通过dtype属性获得

    # img_equalize = equalize_rgb(file_name)
    img_equalize = equalize_ycrcb(file_name)

    plt.figure()

    plt.subplot(321)
    plt.title('Original Image')
    plt.imshow(img)

    plt.subplot(322)
    plt.title('Original Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    hist = cv2.calcHist([img[:, :, 0]], [0], None, [256], [0, 256])
    plt.plot(hist, color='r')
    hist = cv2.calcHist([img[:, :, 1]], [0], None, [256], [0, 256])
    plt.plot(hist, color='g')
    hist = cv2.calcHist([img[:, :, 2]], [0], None, [256], [0, 256])
    plt.plot(hist, color='b')
    plt.xlim([0, 256])  # 设置x坐标轴范围

    plt.subplot(323)
    plt.title('Equalize Image')
    plt.imshow(img_equalize)

    plt.subplot(324)
    plt.title('Equalize Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    hist = cv2.calcHist([img_equalize[:, :, 0]], [0], None, [256], [0, 256])
    plt.plot(hist, color='r')
    hist = cv2.calcHist([img_equalize[:, :, 1]], [0], None, [256], [0, 256])
    plt.plot(hist, color='g')
    hist = cv2.calcHist([img_equalize[:, :, 2]], [0], None, [256], [0, 256])
    plt.plot(hist, color='b')
    plt.xlim([0, 256])  # 设置x坐标轴范围

    plt.subplot(325)
    img_normalize = (normalize(img, -50., 255.) * 255).astype(np.uint8)
    plt.title('Normalize Image')
    plt.imshow(img_normalize)

    plt.subplot(326)
    plt.title('Normalize Grayscale Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    hist = cv2.calcHist([img_normalize[:, :, 0]], [0], None, [256], [0, 256])
    plt.plot(hist, color='r')
    hist = cv2.calcHist([img_normalize[:, :, 1]], [0], None, [256], [0, 256])
    plt.plot(hist, color='g')
    hist = cv2.calcHist([img_normalize[:, :, 2]], [0], None, [256], [0, 256])
    plt.plot(hist, color='b')
    plt.xlim([0, 256])  # 设置x坐标轴范围

    plt.show()  # 显示图像
    return


# 转换颜色空间bgr->rgb
def equalize_rgb(file_name):
    if 'jpg' not in file_name:
        return
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_equalize = cv2.equalizeHist(img[:, :, 0])
    g_equalize = cv2.equalizeHist(img[:, :, 1])
    b_equalize = cv2.equalizeHist(img[:, :, 2])
    img_equalize = cv2.merge([r_equalize, g_equalize, b_equalize])
    cv2.imwrite(os.path.join(equal_dir, file_name.split('/')[-1]), img_equalize)
    return img_equalize


# 转换颜色空间bgr->ycrcb
def equalize_ycrcb(file_name):
    if 'jpg' not in file_name:
        return
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_equalize = cv2.equalizeHist(img[:, :, 0])
    img_equalize = cv2.merge([y_equalize, img[:, :, 1], img[:, :, 2]])
    img_equalize = cv2.cvtColor(img_equalize, cv2.COLOR_YCrCb2RGB)
    cv2.imwrite(os.path.join(equal_dir, file_name.split('/')[-1]), img_equalize)
    return img_equalize


# 移动文件
def move_file(src_file, dst_file):
    if not os.path.isfile(src_file):
        print("%s is not exist!" % (src_file))
    else:
        dstpath = os.path.split(dst_file)[0]
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.move(src_file, dst_file)
        print("move %s -> %s" % (src_file, dst_file))
    return


# 复制文件
def copy_file(src_file, dst_file):
    if not os.path.isfile(src_file):
        print("%s is not exist!" % (src_file))
    else:
        dstpath = os.path.split(dst_file)[0]
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copyfile(src_file, dst_file)
        print("copy %s -> %s" % (src_file, dst_file))
    return


# 整理数据集图像
def change_imagefile():
    image_file_names = os.listdir(entire_dir)
    i = 1
    for file_name in image_file_names:
        # move original images
        org_src = os.path.join(entire_dir, file_name)
        org_dst = os.path.join(org_dir, str(i) + '.jpg')

        # move mask images
        mask_src = os.path.join(split_dir, file_name.split('.')[0] + '_hemorrhage_binary.png')
        mask_dst = os.path.join(mask_dir, str(i) + '.jpg')

        move_file(org_src, org_dst)
        move_file(mask_src, mask_dst)

        # move different type images
#         move_file(os.path.join(split_dir, 'mark1/' + file_name.split('.')[0] + '_mark1.png'),
#                   os.path.join(mask_dir, 'mark1/' + str(i) + '.jpg'))
#         move_file(os.path.join(split_dir, 'mark2/' + file_name.split('.')[0] + '_mark2.png'),
#                   os.path.join(mask_dir, 'mark2/' + str(i) + '.jpg'))
#         move_file(os.path.join(split_dir, 'mark3/' + file_name.split('.')[0] + '_mark3.png'),
#                   os.path.join(mask_dir, 'mark3/' + str(i) + '.jpg'))
#         move_file(os.path.join(split_dir, 'mark4/' + file_name.split('.')[0] + '_mark4.png'),
#                   os.path.join(mask_dir, 'mark4/' + str(i) + '.jpg'))
        i += 1
    return


# 计算数据集平均像素值
def print_mean_pixel(path):
    file_file_names = os.listdir(path)
    file_file_names = [os.path.join(path, _) for _ in file_file_names]
    r_channel = 0
    g_channel = 0
    b_channel = 0
    img_num = len(file_file_names)
    for file_name in file_file_names:
        if 'jpg' not in file_name:
            continue
        print('Dealing img ' + file_name + '……')
        img = cv2.imread(file_name, cv2.IMREAD_COLOR)
        shape = img.shape[0] * img.shape[1]
        r_channel += np.sum(img[:, :, 2]) / shape
        g_channel += np.sum(img[:, :, 1]) / shape
        b_channel += np.sum(img[:, :, 0]) / shape
        print(r_channel, g_channel, b_channel)

    r_mean = r_channel / img_num
    g_mean = g_channel / img_num
    b_mean = b_channel / img_num
    print("Total %d images; R_mean is %f, G_mean is %f, B_mean is %f" % (img_num, r_mean, g_mean, b_mean))
    return


if __name__ == '__main__':
    change_imagefile()
    
    # image_list = os.listdir(org_dir)
    # image_list = [os.path.join(org_dir, _) for _ in image_list]
    # print_rgb_image('/mnt/sda1/hxc/data/val/840.jpg')
    # print(image_list)
    # for file_name in image_list:
    #     equalize_ycrcb(file_name)
    # print_mean_pixel('/mnt/sda1/hxc/data/org/')
