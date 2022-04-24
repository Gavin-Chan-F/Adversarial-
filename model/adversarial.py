import numpy as np
import tensorflow as tf
from tensorflow.keras import backend, losses


def fgsm(model, optimizer, loss_func, esp=0.1):
    """
    :param model: 模型TextCNN
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param esp: 扰动学习率
    :return: train_step
    """
    @tf.function
    def train_step(self, data):
        x,y = data
        with tf.GradientTape() as tape:
            # print(x)
            # print(y)
            y_predict = model(x,training=True)
            loss = loss_func(y, y_predict)
        # print(model.trainable_variables)
        embedding = model.trainable_variables[0]
        gradients_bak = tape.gradient(loss, model.trainable_variables)
        embedding_gradient = tf.zeros_like(embedding) + gradients_bak[0]
        model.trainable_variables[0].assign_add(esp*embedding_gradient)
        with tf.GradientTape() as tape1:
            y_predict1 = model(x,training=True)
            loss1 = loss_func(y, y_predict1)
        gradients = tape1.gradient(loss1,model.trainable_variables)
        model.trainable_variables[0].assign_sub(esp*embedding_gradient)
        optimizer.apply_gradients(zip(gradients+gradients_bak, model.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
    return train_step




def pgd(model, optimizer, loss_func, alpha=0.1, esp=1.0, k=3):
    """
    :param model: 模型TextCNN
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param alpha: 扰动学习率
    :param esp: 扰动半径
    :param k: 迭代次数
    :return: train_step
    """
    @tf.function
    def train_step(self, data):
        x,y = data
        for i in range(k):
            with tf.GradientTape() as tape:
                y_predict = model(x, training=True)
                loss = loss_func(y, y_predict)
            embedding = model.trainable_variables[0]
            if i == 0:
                embedding_bak = model.trainable_variables[0]
                # r = tf.zeros_like(embedding_bak)
            gradients = tape.gradient(loss, model.trainable_variables)
            embedding_gradient = tf.zeros_like(embedding) + gradients[0]
            if i == 0:
                gradient_bak = gradients
            norm = tf.norm(embedding_gradient)
            if norm <= esp:
                model.trainable_variables[0].assign_add(alpha * embedding_gradient)
            else:
                model.trainable_variables[0].assign_add(alpha * embedding_gradient / norm)
        with tf.GradientTape() as tape1:
            y_predict1 = model(x, training=True)
            loss1 = loss_func(y, y_predict1)
        gradients = tape1.gradient(loss1, model.trainable_variables)
        model.trainable_variables[0] = embedding_bak
        optimizer.apply_gradients(zip(gradients+gradient_bak, model.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
    return train_step




def free_LB(model, optimizer, loss_func, init_mag=2e-2, alpha=0.1, k=3):
    """
    :param model: 模型TextCNN
    :param optimizer: 优化器
    :param loss_func: 损失函数
    :param init_mag: 扰动均匀分布初始化参数
    :param alpha: 扰动学习率
    :param k: 迭代次数
    :return: train_step
    """
    @tf.function
    def train_step(self, data):
        x,y = data
        for i in range(k):
            with tf.GradientTape() as tape:
                if i == 0:
                    model.trainable_variables[0] += tf.random.uniform(model.trainable_variables[0].shape(), -init_mag, init_mag)
                y_predict = model(x, training=True)
                loss = loss_func(y, y_predict)
            embedding = model.trainable_variables[0]
            # if i == 0:
            #     embedding_bak = model.trainable_variables[0]
            gradients_step = tape.gradient(loss, model.trainable_variables)
            embedding_gradient = tf.zeros_like(embedding) + gradients_step[0]
            if i == 0:
                gradients = gradients_step
            else:
                gradients += gradients_step
            model.trainable_variables[0].assign_add(alpha * embedding_gradient)
        with tf.GradientTape() as tape1:
            y_predict1 = model(x, training=True)
            loss1 = loss_func(y, y_predict1)
        gradients_last_step = tape1.gradient(loss1, model.trainable_variables)
        gradients += gradients_last_step
        # model.trainable_variables[0] = embedding_bak
        gradients_div = [gradient/(tf.ones_like(gradient) * (k+1)) for gradient in gradients]
        # gradients = tf.divide(gradients, k+1)
        optimizer.apply_gradients(zip(gradients_div, model.trainable_variables))
        return {m.name: m.result() for m in self.metrics}
    return train_step