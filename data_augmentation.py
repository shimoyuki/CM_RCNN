"""
CM R-CNN
Tools for data augmentation.

Written by Lycoris
"""

import Augmentor
import glob
import os
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_path = '/mnt/sda1/hxc/data/org'
mask_path = '/mnt/sda1/hxc/data/mask'
img_type = 'jpg'
train_tmp_path = '/mnt/sda1/hxc/data/tmp/org'
mask_tmp_path = '/mnt/sda1/hxc/data/tmp/mask'


# 拷贝图像及掩膜到单独文件夹以符合Augmentor格式
def initial(train_path, mask_path):
    masks = glob.glob(mask_path + '/*.' + img_type)

    for i in range(len(masks)):
        train_img_tmp_path = train_tmp_path + '/' + str(i)
        mask_img_tmp_path = mask_tmp_path + '/' + str(i)
        if not os.path.lexists(train_img_tmp_path):
            os.makedirs(train_img_tmp_path)
            img = load_img(train_path + '/' + masks[i].split("\\")[-1])
            x_t = img_to_array(img)
            img_tmp = array_to_img(x_t)
            img_tmp.save(train_img_tmp_path + '/' + masks[i].split("\\")[-1])

        if not os.path.lexists(mask_img_tmp_path):
            os.makedirs(mask_img_tmp_path)
            mask = load_img(mask_path + '/' + masks[i].split("\\")[-1])
            x_l = img_to_array(mask)
            mask_tmp = array_to_img(x_l)
            mask_tmp.save(mask_img_tmp_path + '/' + masks[i].split("\\")[-1])

        print("%s folder has been created!" % str(i))

    return i + 1


# 数据增广操作
# @num 原始图像数量
def augment(num):
    sum = 0
    for i in range(num):
        p = Augmentor.Pipeline(train_tmp_path + '/' + str(i))
        p.ground_truth(mask_tmp_path + '/' + str(i))
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)  # 旋转,左右旋转的数值不能超过25
        p.flip_left_right(probability=0.5)  # 按概率左右翻转，其它函数：p.flip_random()……
        p.zoom_random(probability=0.5, percentage_area=0.99)  # 随即将一定比例面积的图形放大至全图
        p.flip_top_bottom(probability=0.5)  # 按概率随即上下翻转
        # p.skew_tilt(probability=0.5,magnitude=1)#上下左右方向的垂直型变，参数magnitude为型变的程度（0，1），其它函数：skew_left_right(),skew()……
        # p.skew_corner(probability=0.5,magnitude=1)#向四个角形变
        p.random_distortion(probability=0.5, grid_width=10, grid_height=10, magnitude=10)  # 小块变形
        p.shear(probability=0.5, max_shear_left=5, max_shear_right=5)  # 错切变换，使图像向某一侧倾斜,参数范围是0-25
        # p.crop_by_size(probability=0.5,width=100,height=100,centre=True)#截取，其它函数：p.crop_centre(),p.crop_random()……
        # p.random_erasing(probability=0.5,rectangle_area=0.5)#随机擦除
        count = random.randint(1, 3)
        print("\nNo.%s data is being augmented and %s data will be created" % (i, count))
        sum = sum + count
        p.sample(count)
        print("Done")
    print("%s pairs of data has been created totally" % sum)
    return


# 拷贝原数据到临时文件夹
def copy(train_path, mask_path):
    masks = glob.glob(mask_path + '/*.' + img_type)

    for i in range(len(masks)):
        train_img_tmp_path = train_tmp_path
        mask_img_tmp_path = mask_tmp_path
        if not os.path.lexists(train_img_tmp_path):
            os.makedirs(train_img_tmp_path)
        img = load_img(train_path + '/' + masks[i].split("\\")[-1])
        x_t = img_to_array(img)
        img_tmp = array_to_img(x_t)
        img_tmp.save(train_img_tmp_path + '/' + masks[i].split("\\")[-1])

        if not os.path.lexists(mask_img_tmp_path):
            os.makedirs(mask_img_tmp_path)
        mask = load_img(mask_path + '/' + masks[i].split("\\")[-1])
        x_l = img_to_array(mask)
        mask_tmp = array_to_img(x_l)
        mask_tmp.save(mask_img_tmp_path + '/' + masks[i].split("\\")[-1])

        print("%s images has been copied!" % str(i))
    return


if __name__ == '__main__':
    # copy(train_path, mask_path)
    a = initial(train_path, mask_path)
    augment(a)
