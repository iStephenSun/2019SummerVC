import pickle
import numpy as np


def load_pkl(path):
    f = open(path, 'rb')
    (feature, label) = pickle.load(f)
    f.close()
    return feature, label


def load_distance(path):
    dis = []
    f = open(path, 'r')  # 以读方式打开文件
    for line in f.readlines():  # 依次读取每行
        line = line.strip()  # 去掉每行头尾空白
        if not len(line) or line.startswith('#'):  # 判断是否是空行或注释行
            continue  # 是的话，跳过不处理
        dis.append(float(line))  # 保存
    dis = np.array(dis)
    return dis


def get_threshold(pos_dis, neg_dis):
    threshold_min = min(pos_dis)
    threshold_max = max(neg_dis)
    if threshold_max > 4:
        threshold_max = 4
    # print(threshold_min, threshold_max)
    step = (float(threshold_max) - float(threshold_min))/50
    # print(step)
    threshold_values = np.arange(threshold_min, threshold_max, step)
    return threshold_values
