import os
import argparse
from tensorflow import keras
import tensorflow as tf
from pprint import pprint
import time
import tensorflow.keras.backend as K
from textCNN import data_loader
from textCNN.textcnn_model import TextCNN
from textCNN.config import *


# 评价函数（代码可能有问题）
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

# 评价函数
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

def parserArgument():
    parser = argparse.ArgumentParser(description='This is the TextCNN train project.')
    # 样本最大长度，超过padding_size就截取，不足就进行padding
    parser.add_argument('-p', '--padding_size', default=PADDINGSIZE, type=int, help='Padding size of sentences.(default=128)')
    # 词向量大小
    parser.add_argument('-e', '--embed_size', default=50, type=int, help='Word embedding size.(default=512)')
    # 卷积核
    parser.add_argument('-f', '--filter_sizes', default='3,4,5', help='Convolution kernel sizes.(default=3,4,5)')
    # 每层卷积层神经元个数
    parser.add_argument('-n', '--num_filters', default=32, type=int,
                        help='Number of each convolution kernel.(default=128)')
    parser.add_argument('-d', '--dropout_rate', default=0.2, type=float,
                        help='Dropout rate in softmax layer.(default=0.5)')
    # 标签类数
    parser.add_argument('-c', '--num_classes', default=88, type=int, help='Number of target classes.(default=18)')
    # 做正则的系数（正则化通常用在损失函数中）
    parser.add_argument('-l', '--regularizers_lambda', default=0.01, type=float,
                        help='L2 regulation parameter.(default=0.01)')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float, help='Learning rate.(default=0.005)')
    parser.add_argument('-vocab_size', default=VOCABSIZE, type=int, help='Limit vocab size.(default=50000)')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help='Number of epochs.(default=10)')
    # 验证数据集比例（5%）
    parser.add_argument('--fraction_validation', default=0.05, type=float,
                        help='The fraction of validation.(default=0.05)')
    # 测试集比例
    parser.add_argument('-t', '--test_sample_percentage', default=0.1, type=float,
                        help='The fraction of test data.(default=0.1)')
    parser.add_argument('--results_dir', default=results_dir, type=str,
                        help='The results dir including log, model, vocabulary and some images.(default=./results/)')
    args = parser.parse_args()
    print('Parameters:', args, '\n')

    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    return args

def train(x_train, y_train, vocab_size, feature_size, save_path):
    '''
    :param x_train:
    :param y_train:
    :param vocab_size:
    :param feature_size: 样本最大长度
    :param save_path:
    :return:
    '''
    # 1. 创建模型
    model = TextCNN(vocab_size,feature_size,args.embed_size,
                    args.num_classes,args.num_filters,args.filter_sizes,
                    args.regularizers_lambda,args.dropout_rate)
    model.summary()     #训练过程中，打印模型结构
    # 2. 定义超参，优化器，损失函数，等
    model.compile(tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss= 'binary_crossentropy',metrics=[micro_f1,macro_f1])
    # 3. 训练，steps_per_epoch：每个epoch跑多少遍
    history = model.fit(x=x_train,y=y_train,batch_size=args.batch_size,
                        epochs=args.epochs,steps_per_epoch=VOCABSIZE//args.batch_size,
                        validation_split=args.fraction_validation,shuffle=True)
    # 4. 保存
    keras.models.save_model(model,save_path)        #保存模型
    pprint(history.history)

if __name__ == '__main__':
    args = parserArgument()

    x_train, y_train = data_loader.get_data(train_x_path, train_y_path)
    train(x_train, y_train, args.vocab_size, args.padding_size, os.path.join(args.results_dir, 'TextCNN.h5'))