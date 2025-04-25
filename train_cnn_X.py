# -*- coding: utf-8 -*-集中式的cnn模型训练
import tensorflow as tf
import matplotlib.pyplot as plt
from time import *
import os

# 指定要创建的文件夹路径
folder_path = 'models'

# 使用os.makedirs()函数创建文件夹
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',  #标签以独热编码形式返回
        seed=123,
        image_size=(img_height, img_width),   #把图片调整为指定的高度和宽度
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names


# 构建CNN模型
def model_load(IMG_SHAPE, class_num):
    # 搭建模型
    model = tf.keras.models.Sequential([
        # 对模型做归一化的处理，将0-255之间的数字统一处理到0到1之间
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),
        # 卷积层，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # 添加池化层，池化的kernel大小是2*2
        tf.keras.layers.MaxPooling2D(2, 2),
        # Add another convolution
        # 卷积层，输出为64个通道，卷积核大小为3*3，激活函数为relu
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        # 池化层，最大池化，对2*2的区域进行池化操作
        tf.keras.layers.MaxPooling2D(2, 2),
        # 将二维的输出转化为一维
        tf.keras.layers.Flatten(),
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),
        # 通过softmax函数将模型输出为类名长度的神经元上，激活函数采用softmax对应概率值
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数，学习率默认0.01
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # 返回模型
    return model


# 展示训练过程的曲线
def show_loss_acc2(history,epochs):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 取整数的epoch对应的数据点
    epoches = range(1, len(acc) + 1)

    # 绘制准确率图像
    plt.figure()
    plt.plot(epoches, val_acc,marker='o')
    plt.ylabel('Accuracy')
   #plt.ylim([min(plt.ylim()), 1])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(range(1, epochs + 1))
    plt.grid(True)
    plt.savefig('X_results/cnn_acc_10_3.png', dpi=100)
    
    # 绘制损失图像
    plt.figure()
    plt.plot(epoches, val_loss,marker='o')
    plt.ylabel('Cross Entropy')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(1, epochs + 1))
    plt.grid(True)
    plt.savefig('X_results/cnn_loss_10_3.png', dpi=100)
   
    # 绘制准确率图像
    plt.figure()
    plt.plot(epoches, acc,marker='o')
    plt.ylabel('Accuracy')
   #plt.ylim([min(plt.ylim()), 1])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(range(1, epochs + 1))
    plt.grid(True)
    plt.savefig('X_results/cnn_acc_t10_3.png', dpi=100)
    
    # 绘制损失图像
    plt.figure()
    plt.plot(epoches, loss,marker='o')
    plt.ylabel('Cross Entropy')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(1, epochs + 1))
    plt.grid(True)
    plt.savefig('X_results/cnn_loss_t10_3.png', dpi=100)

def show_loss_acc(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('X_results/cnn_10_2.png', dpi=100)

def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # todo 加载数据集， 修改为你的数据集的路径（高，宽，batchsize）
    train_ds, val_ds, class_names = data_load("./X_datasets/train",
                                              "./X_datasets/test", 224, 224, 16)
    print(val_ds)
    print(class_names)
    # 加载模型
    model = model_load(IMG_SHAPE=(224, 224, 3),class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo 保存模型， 修改为你要保存的模型的名称
    model.save("X_models/normal_cnn.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  
    # 绘制模型训练过程图
    show_loss_acc(history)
    show_loss_acc2(history,epochs)


if __name__ == '__main__':
    train(epochs=10)
