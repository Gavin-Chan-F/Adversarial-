import functools
import time
import numpy as np
from model.adversarial import fgsm, pgd, free_LB
from tensorflow.keras.callbacks import EarlyStopping
from data_manager import stopwordList, data_process
from model.TextCNN import TextCNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from tensorflow.keras.models import load_model

maxlen = 100 # 最大句长（词数）
max_features = 10000 # 词表维度
test_size=0.3 # 测试集比例
embedding_dims = 128 # 词向量维度
batch_size = 64 # batch size
epochs = 20 # 最大训练epoch
model_save_path = "./save_data/" # 模型存储路径


def train():
    stopWords = stopwordList('./raw_data/stopWord.txt') # 停用词
    # 读取训练数据
    f = open("./raw_data/all_data_clean.txt", "r", encoding='utf-8')
    lines = f.readlines()
    f.close()
    texts = []
    labels = []
    for line in lines:
        # print(line)
        try:
            label, text = line.strip().split("\t")
            texts.append(text)
            labels.append(label)
        except:
            pass
    # 数据预处理
    x_train,x_test,y_train,y_test = data_process(texts,labels,stopWords,maxlen,max_features,test_size,2022)
    g = open('./save_data/report.txt', 'w', encoding='utf-8')
    le = joblib.load(model_save_path+"le.model")
    class_names = le.classes_
    # 模型训练：baseline及3种对抗策略
    for train_method in ['normal', 'fgsm', 'pgd', 'freeLB']:
        model = TextCNN(maxlen, max_features, embedding_dims, class_num=8, last_activation='softmax')
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        if train_method == 'fgsm':
            train_step = fgsm(model, Adam(), categorical_crossentropy)
            model.train_step = functools.partial(train_step, model)
            model.train_function = None
        elif train_method == 'pgd':
            train_step = pgd(model, Adam(), categorical_crossentropy, alpha=0.1, esp=1.0, k=3)
            model.train_step = functools.partial(train_step, model)
            model.train_function = None
        elif train_method == 'freeLb':
            train_step = free_LB(model, Adam(), categorical_crossentropy, alpha=0.1, esp=1.0, k=3)
            model.train_step = functools.partial(train_step, model)
            model.train_function = None
        else:
            pass
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
        time_start = time.time()
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=[early_stopping],
                  validation_data=(x_test, y_test))
        model.save_weights(model_save_path + "/" + train_method + "/" + train_method, save_format="tf")
        result = model.predict(x_test)
        y_true = np.argmax(y_test, axis=-1)
        y_pred = np.argmax(result, axis=-1)
        # print(y_test[0])
        # print(y_pred[0])
        report = classification_report(y_true, y_pred, target_names=class_names)
        time_end = time.time()
        time_use = time_end - time_start
        g.write("train methord: "+train_method + "\n")
        g.write("training time: " + str(time_use) + '\n')
        g.write(report)
        g.write("\n\n\n")
    g.close()


if __name__ == '__main__':
    train()

