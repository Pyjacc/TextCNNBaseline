
from tensorflow.keras.models import load_model
import numpy as np
import os
from pprint import pprint
import tensorflow.keras.backend as K
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.metrics import categorical_accuracy
from textCNN.data_loader import preprocess,get_data
from textCNN.config import *

def micro_f1(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    macro_f1 = K.mean(2 * precision * recall / (precision + recall + K.epsilon()))

    """Micro_F1 metric.
    """
    precision = K.sum(true_positives) / (K.sum(predicted_positives)+K.epsilon())
    recall = K.sum(true_positives) / (K.sum(possible_positives)+K.epsilon())
    micro_f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return micro_f1


def macro_f1(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    macro_f1 = K.mean(2 * precision * recall / (precision + recall + K.epsilon()))

    """Micro_F1 metric.
    """
    precision = K.sum(true_positives) / K.sum(predicted_positives)
    recall = K.sum(true_positives) / K.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return macro_f1

def f1_np(y_true, y_pred):
    """F1 metric.
    Computes the micro_f1 and macro_f1, metrics for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)), axis=0)
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)), axis=0)
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)), axis=0)

    """Macro_F1 metric.
    """
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (possible_positives + 1e-8)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall + 1e-8))

    """Micro_F1 metric.
    """
    precision = np.sum(true_positives) / np.sum(predicted_positives)
    recall = np.sum(true_positives) / np.sum(possible_positives)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return micro_f1, macro_f1       # 建议使用macro_f1


def test(model, x_test, y_test):
    # batch_size=1：一个一个进行预测，verbose：verbosity mode日志显示，0 or 1
    # https://blog.csdn.net/C_chuxin/article/details/84573398
    y_pred = model.predict(x=x_test, batch_size=1, verbose=1)
    # print(y_pred)
    metrics = [f1_np]   #将函数f1_np的地址放入list中，后面的使用中类似回调
    result = {}
    for func in metrics:
        # print(func.__name__)  # f1_np
        result[func.__name__] = func(y_test, y_pred)    #调用f1_np函数
    pprint(result)      #打印结果中'f1_np': (nan, 0.0)维度值为（micro_f1, macro_f1）



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN test project.')
    parser.add_argument('--results_dir', default=results_dir,type=str, help='The results dir including log, model, vocabulary and some images.')
    args = parser.parse_args()
    print('Parameters:', args)
    x_test, y_test = get_data(test_x_path, test_y_path)
    print("Loading model...")

    # custom_objects（字定义的评价指标）要与train中metrics=[micro_f1,macro_f1]对应
    model = load_model(os.path.join(args.results_dir, 'TextCNN.h5'), custom_objects={"micro_f1":micro_f1,"macro_f1":macro_f1})
    test(model, x_test, y_test)