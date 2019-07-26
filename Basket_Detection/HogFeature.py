import cv2
import os
import numpy as np
import math
from HOGDescriptor import HOGDescriptor

win_size = (80, 80)
block_size = (16, 16)
over_lap = (8, 8)
cell_size = (8, 8)
bin = 9

def load_images(directory_name):
    array_of_img = []
    for filename in os.listdir(directory_name):
        # print(filename) #just for test
        # img is used to store the image data
        img = cv2.imread(directory_name + "/" + filename, cv2.IMREAD_GRAYSCALE)
        array_of_img.append(img)
        # print(img)
    return array_of_img


def image_hog(image):
    # hog = HOGDescriptor(image)
    # h = HOGDescriptor.HOGExtract(hog)
    hog = cv2.HOGDescriptor(win_size, block_size, over_lap, cell_size, bin)
    h = hog.compute(image)
    return h


def compute_hog(path):
    # 初始化
    features = []
    imgs = load_images(path)

    for i in range(len(imgs)):
        roi = imgs[i]
        h = image_hog(roi)
        features.append(h)

    features = np.array(features)

    return features


def Euclidean(center, test):
    # 计算每个维度上差值平方根之和
    return math.sqrt(((center-test)**2).sum())


def test_distance(center, test_path, save_path):
    test_image = load_images(test_path)

    distance = []

    for i in range(len(test_image)):
        test_result = image_hog(test_image[i])
        distance.append(Euclidean(center, test_result))

    distance = np.array(distance)

    f = open(save_path, 'a')
    for d in distance:
        f.write(str(d) + '\n')
    f.close()

    return
