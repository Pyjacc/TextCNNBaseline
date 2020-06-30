
import pandas as pd
from tensorflow.keras import preprocessing
import numpy as np
import jieba
from sklearn.preprocessing import MultiLabelBinarizer
import os
from pathlib import Path
import json
import csv
import re
from textCNN.config import *


def get_data(file_x_path, file_y_path, origin_data=origin_data_patch):
    if not os.path.exists(file_x_path):
        preprocess(origin_data)

    x_train = np.load(file_x_path)
    y_train = np.load(file_y_path)

    return x_train, y_train


def preprocess(data_file, save_dir='../datasets/', vocab_size=VOCABSIZE, padding_size=PADDINGSIZE):

    data_file = Path(data_file)
    # header：要不要索引，label：标签列，item：表示文本列
    df = pd.read_csv(data_file,header=None,names=["label","item"],dtype=str)    #转为pandas格式文件

    df["item"] = df.item.apply(lambda x: list(jieba.cut(x)))    #对item列的每一行进行分词，并赋给item列

    corpus = df.item.tolist()       #构建词的语料
    # text_processer类似vocab对象
    text_preprocesser = preprocessing.text.Tokenizer(num_words=vocab_size)      #词表（空壳子）
    # fit_on_texts：统计词频，并根据词频排序
    text_preprocesser.fit_on_texts(corpus)
    x = text_preprocesser.texts_to_sequences(corpus)

    word_dict = text_preprocesser.word_index        #词典，key：词，value：词频
    with open(save_dir + 'vocab.txt','w',encoding='utf-8') as f:
        for k,v in word_dict.items():
            f.write(f'{k}\t{str(v)}\n')     #保存字典

    #进行pad操作，padding：指定从左边还是右边进行pad操作，truncating：截断
    x = preprocessing.sequence.pad_sequences(x,maxlen=padding_size,padding='post',truncating='post')

    y = df.label.apply(lambda x:set(x.split())).tolist()
    mlb = MultiLabelBinarizer() #创建对象
    y = mlb.fit_transform(y)    #将y转为多标签方式

    # 保存所有的标签
    with open(f'{save_dir} labels_{data_file.stem}.txt','w',encoding='utf-8') as f:
        for label in mlb.classes_:
            f.write(f'{label}\n')

    index = list(range(len(x)))
    np.random.seed(0)
    np.random.shuffle(index)

    x = x[index]
    y = y[index]

    split = int(len(x) * 0.9)       #90%作为训练集
    # 保存为.npy格式，提高读取效率。
    # 注：从txt或其他文件读取1000 * 1000的数据，直接读取大约需要1s，而转成.npy后，读取只需要约0.01s，
    # 速度提升100倍。如果需要多次读取相同的数据文件，这是一个有用的技巧，数据量越大，速度提升越明显！
    # 采用np.load()的形势加载.npy文件
    np.save(f'{save_dir}{data_file.stem}_train_x.npy',x[:split])    #标签对应的文本
    np.save(f'{save_dir}{data_file.stem}_test_x.npy', x[split:])
    np.save(f'{save_dir}{data_file.stem}_train_y.npy', y[:split])   #真实标签
    np.save(f'{save_dir}{data_file.stem}_test_y.npy', y[split:])


if __name__ == '__main__':
    preprocess(origin_data_patch)
