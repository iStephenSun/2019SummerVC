import os
import cv2
import random
import shutil
import numpy as np
import pickle
import HogFeature


def get_image_array(path):
    array_of_img = []
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename, cv2.IMREAD_GRAYSCALE)
        img = np.array(img)
        array_of_img.append(img)
    return array_of_img


def label_pos(image_data, image_data_label):
    size = len(image_data)

    for i in range(size):
        image_data_label.append(1)


def label_neg(image_data, image_data_label):
    size = len(image_data)

    for i in range(size):
        image_data_label.append(0)


def save_as_pkl(data, path):
    file = open(path, 'wb')
    pickle.dump(data, file)
    file.close()


def get_image_list_hog(image, feature):
    for i in range(len(image)):
        roi = image[i]
        h = HogFeature.image_hog(roi)
        feature.append(h)


def training_data_set(source1, source2, train_pkl):
    pos_image = get_image_array(source1)
    neg_image = get_image_array(source2)
    train_image = pos_image + neg_image

    train_label = []
    train_feature = []
    get_image_list_hog(train_image, train_feature)

    label_pos(pos_image, train_label)
    label_neg(neg_image, train_label)

    train_data = (train_feature, train_label)
    save_as_pkl(train_data, train_pkl)


def select_test_img(source1, source2, target, train_pkl, test_pkl):
    print("测试图片目录", target)
    rate = 0.3

    image1 = os.listdir(source1)
    number1 = len(image1)
    picknumber1 = int(number1 * rate)
    sample1 = random.sample(image1, picknumber1)

    image2 = os.listdir(source2)
    number2 = len(image2)
    picknumber2 = int(number2 * rate)
    sample2 = random.sample(image2, picknumber2)

    test_feature = []
    test_label = []
    cnt = 0
    for name in sample1:
        cnt = cnt + 1
        shutil.move(source1 + name, target + str(cnt) + ".jpg")
        test_label.append(1)

    for name in sample2:
        cnt = cnt + 1
        shutil.move(source2 + name, target + str(cnt) + ".jpg")
        test_label.append(0)

    print("Begin generate pkl")

    test_image = get_image_array(target)
    get_image_list_hog(test_image, test_feature)
    test_data = (test_feature, test_label)
    save_as_pkl(test_data, test_pkl)

    training_data_set(source1, source2, train_pkl)

    print("Data Set Ready")

    return
