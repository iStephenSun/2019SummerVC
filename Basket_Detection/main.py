import numpy as np

import HogFeature
import LabelFunc
import PictureCombine
import PrepareDataSet
import ReadFile
import Classifier
import ROC


if __name__ == "__main__":
    print('-------------------')
    print("Read the configure file")
    fp = open('configure.txt', 'r')
    strConfig = fp.readlines()
    label_vFn = ''
    label_vAnnFile = ''
    crop_vFn = ''
    crop_vAnnFile = ''
    outDIRPos = ''
    outDIRNeg = ''
    task = ''
    crop_size = ''
    crop_row = ''
    crop_column = ''
    crop_pos_save = ''
    crop_neg_save = ''
    test_path = ''
    train_path = ''
    train_pkl = ''
    test_pkl = ''
    center_pos_path = ''
    center_neg_path = ''
    pos_distance_path = ''
    neg_distance_path = ''
    test_distance_path = ''

    for line in strConfig:
        line = line.strip('')
        if line.find('gl_task') >= 0:
            task = line.split('=')[1].strip()
        if line.find('label_fn_video') >= 0:
            label_vFn = line.split('=')[1].strip()
        if line.find('label_fn_annotation') >= 0:
            label_vAnnFile = line.split('=')[1].strip()
        if line.find('crop_fn_video') >= 0:
            crop_vFn = line.split('=')[1].strip()
        if line.find('crop_fn_annotation') >= 0:
            crop_vAnnFile = line.split('=')[1].strip()
        if line.find('crop_dir_pos') >= 0:
            outDIRPos = line.split('=')[1].strip()
        if line.find('crop_dir_neg') >= 0:
            outDIRNeg = line.split('=')[1].strip()
        if line.find('IMAGE_SIZE') >= 0:
            crop_size = line.split('=')[1].strip()
        if line.find('IMAGE_ROW') >= 0:
            crop_row = line.split('=')[1].strip()
        if line.find('IMAGE_COLUMN') >= 0:
            crop_column = line.split('=')[1].strip()
        if line.find('POS_SAVE_PATH') >= 0:
            crop_pos_save = line.split('=')[1].strip()
        if line.find('NEG_SAVE_PATH') >= 0:
            crop_neg_save = line.split('=')[1].strip()
        if line.find('testing_fn') >= 0:
            test_path = line.split('=')[1].strip()
        if line.find('train_pkl') >= 0:
            train_pkl = line.split('=')[1].strip()
        if line.find('test_pkl') >= 0:
            test_pkl = line.split('=')[1].strip()
        if line.find('training_list') >= 0:
            train_path = line.split('=')[1].strip()
        if line.find('pos_train_fn') >= 0:
            center_pos_path = line.split('=')[1].strip()
        if line.find('neg_train_fn') >= 0:
            center_neg_path = line.split('=')[1].strip()
        if line.find('test_distance_dir') >= 0:
            test_distance_path = line.split('=')[1].strip()
        if line.find('pos_distance_dir') >= 0:
            pos_distance_path = line.split('=')[1].strip()
        if line.find('neg_distance_dir') >= 0:
            neg_distance_path = line.split('=')[1].strip()

    print('-------------------')
    print("Carry the task", task)

    hoopPos = np.zeros((2, 2), np.int)
    # hoopPos = LabelFunc.getHoopPosition(label_vFn)
    # hoopPos = np.array([[865, 79], [965, 179]]) # ball1
    # hoopPos = np.array([[505, 272], [555, 322]]) # ball3

    if task == 'label':
        LabelFunc.labelGoalFrames(label_vFn, hoopPos, label_vAnnFile)
    elif task == 'crop':
        LabelFunc.cropHoop(crop_vFn, hoopPos, crop_vAnnFile, outDIRPos, outDIRNeg)
    elif task == 'compose':
        PictureCombine.image_compose(outDIRPos, int(crop_size), int(crop_row), int(crop_column), crop_pos_save)
        PictureCombine.image_compose(outDIRNeg, int(crop_size), int(crop_row), int(crop_column), crop_neg_save)
        PrepareDataSet.select_test_img(crop_pos_save, crop_neg_save, test_path, train_pkl, test_pkl)
    elif task == 'training':
        hog_center = np.mean(HogFeature.compute_hog(train_path), 0)
        HogFeature.test_distance(hog_center, center_pos_path, pos_distance_path)
        HogFeature.test_distance(hog_center, center_neg_path, neg_distance_path)
        HogFeature.test_distance(hog_center, test_path, test_distance_path)
    elif task == 'ROC':
        pos_distance = ReadFile.load_distance(pos_distance_path)
        neg_distance = ReadFile.load_distance(neg_distance_path)
        threshold_values = ReadFile.get_threshold(pos_distance, neg_distance)
        test_data = ReadFile.load_distance(test_distance_path)
        test_img, test_label = ReadFile.load_pkl(test_pkl)
        test_label = np.array(test_label, dtype=int)
        ROC.plotROC(test_data, test_label, threshold_values)
    elif task == 'Nearest':
        train_data, train_label = ReadFile.load_pkl(train_pkl)
        test_data, test_label = ReadFile.load_pkl(test_pkl)
        print("pkl files are load")
        Classifier.Classify(train_data, train_label, test_data, test_label)