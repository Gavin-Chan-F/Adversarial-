import numpy as np
from tensorflow.keras.layers import Input, Embedding, Dropout, Dense, Concatenate, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Model



class TextCNN(Model):

    # 模型参数初始化，layer初始化
    def __init__(self, maxlen, max_features,embedding_dims, class_num=1, last_activation='sigmoid'):
        super(TextCNN, self).__init__()
        self.maxlen = maxlen    #单句最大词数
        self.max_features = max_features    #最大特征数，即词表维度
        self.embedding_dims = embedding_dims    #初始化向量维度
        self.class_num = class_num  # 分类数
        self.last_activation = last_activation  # 激活函数，单分类sigmoid，多分类softmax
        self.embedding = Embedding(self.max_features,self.embedding_dims,input_length=self.maxlen)
        self.Conv1D3K = Conv1D(filters=128, kernel_size=3, activation='relu')
        self.Conv1D4K = Conv1D(filters=128, kernel_size=4, activation='relu')
        self.Conv1D5K = Conv1D(filters=128, kernel_size=5, activation='relu')
        self.GlobalMaxPooling1D3K = GlobalMaxPooling1D()
        self.GlobalMaxPooling1D4K = GlobalMaxPooling1D()
        self.GlobalMaxPooling1D5K = GlobalMaxPooling1D()
        self.Concatenate = Concatenate()
        self.Dense = Dense(self.class_num,activation=self.last_activation)
    def call(self, x):
        # input = Input((self.maxlen,))
        embedding = self.embedding(x)
        convs = []
        c3 = self.Conv1D3K(embedding)
        c3 = self.GlobalMaxPooling1D3K(c3)
        convs.append(c3)
        c4 = self.Conv1D4K(embedding)
        c4 = self.GlobalMaxPooling1D4K(c4)
        convs.append(c4)
        c5 = self.Conv1D5K(embedding)
        c5 = self.GlobalMaxPooling1D5K(c5)
        convs.append(c5)
        conv_f = self.Concatenate(convs)
        output = self.Dense(conv_f)
        return output




