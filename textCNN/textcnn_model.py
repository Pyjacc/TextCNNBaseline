from tensorflow import keras

def TextCNN(vocab_size, feature_size, embed_size, num_classes, num_filters,
            filter_sizes, regularizers_lambda, dropout_rate):
    '''
    :param vocab_size: 词典大小
    :param feature_size: 样本最大长度
    :param embed_size: 词向量大小
    :param num_classes: 标签数量
    :param num_filters: 神经元数量
    :param filter_sizes: 卷积核数量
    :param regularizers_lambda: 正则化系数
    :param dropout_rate: dropout系数
    :return:
    '''
    inputs = keras.Input(shape=(feature_size,), name='input_data')  #shape：样本的长度
    embed_initer = keras.initializers.RandomUniform(minval=-1, maxval=1)    # 创建随机初始化对象
    # embedding层
    # embed.shape:(None, 128, 50).None位置为batch_size的维度，层与层之间传播时，batch_size的维度为None
    embed = keras.layers.Embedding(vocab_size,
                                   embed_size,
                                   embeddings_initializer=embed_initer,
                                   input_length=feature_size,
                                   name='embedding')(inputs)

    # 增加1维，为了做卷积操作（卷积操作为3维）
    # embed.shape:(None, 128, 50, 1)
    embed = keras.layers.Reshape((feature_size, embed_size, 1), name='add_channel')(embed)  #增加了1维通道

    pool_outputs = []       #保存池化结果
    # map：将filter_sizes中的每个string对象转为int
    # filter_sizes.split(','):['3', '4', '5']
    for filter_size in list(map(int, filter_sizes.split(','))):
        filter_shape = (filter_size, embed_size)
        # 进行卷积操作
        conv = keras.layers.Conv2D(num_filters,     #神经元数量
                                   filter_shape,    # 卷积核
                                   strides=(1, 1),  #步长
                                   padding='valid', #边沿进行padding（padding方式）
                                   data_format='channels_last',
                                   activation='relu',#激活函数
                                   kernel_initializer='glorot_normal',
                                   bias_initializer=keras.initializers.constant(0.1),#偏置
                                   name='convolution_{:d}'.format(filter_size))(embed)

        max_pool_shape = (feature_size - filter_size + 1, 1)    #池化后的维度大小与卷积操核大小有关
        # 池化
        pool = keras.layers.MaxPool2D(pool_size=max_pool_shape,
                                      strides=(1, 1),   #步长
                                      padding='valid',  #padding方式（多维层面padding）
                                      data_format='channels_last',
                                      name='max_pooling_{:d}'.format(filter_size))(conv)
        pool_outputs.append(pool)

    pool_outputs = keras.layers.concatenate(pool_outputs, axis=-1, name='concatenate')  #拼接
    # api：https://keras.io/zh/layers/core/
    pool_outputs = keras.layers.Flatten(data_format='channels_last', name='flatten')(pool_outputs)  #展开，便于做全连接
    pool_outputs = keras.layers.Dropout(dropout_rate, name='dropout')(pool_outputs)
    # 全连接层，activation可用sigmod
    outputs = keras.layers.Dense(num_classes, activation='softmax',
                                 kernel_initializer='glorot_normal',
                                 bias_initializer=keras.initializers.constant(0.1),
                                 kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 bias_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 name='dense')(pool_outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


