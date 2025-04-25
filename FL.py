#标准的联邦学习
# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt
from time import *
import numpy as np
import os
#import split_data
# 指定要创建的文件夹路径
folder_path = 'X_models'

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
    # 加载验证集
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名   
    return train_ds, test_ds,class_names

def client_average_data_load(train_ds, test_ds, client_num):
    # 获取训练数据的大小
    train_ds_size = len(train_ds)
    # 将训练数据平均分为 client_num 份
    client_train_data = []
    for i in range(client_num):
        # 计算每个客户端的起始和结束索引
        start_index = (i * train_ds_size) // client_num
        end_index = ((i + 1) * train_ds_size) // client_num
        # 从原始训练集中切片得到每个客户端的训练数据
        client_train_data.append(train_ds.shard(client_num, i))
    # 测试数据对每个客户端都是相同的
    client_test_data = [test_ds] * client_num
    return client_train_data, client_test_data


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
    #model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数，学习率默认0.01
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # 返回模型
    return model


# 展示训练过程的曲线
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
    plt.title('X_Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('X_results/test_cnn.png', dpi=100)



if __name__ == '__main__':
    global_epochs=10
    num_clients=5
    client_epochs=2
    log_dir = "./X_models/"
    #获取全部数据集及种类
    train_ds, test_ds, class_names=data_load("./X_datasets/train","./X_datasets/test", 224, 224, 16)
    print(class_names)
    #客户端数据集
    client_train_data = [None] * num_clients
    client_test_data = [None] * num_clients
    client_class_names = [None] * num_clients
    client_train_data,client_test_data=client_average_data_load(train_ds, test_ds,num_clients)
    #client_train_data,client_test_data=split_data.get_client_data(partition=1)
    #client_train_data[0],client_test_data[0],client_class_names[0] = data_load(clients_data_dirs, clients_test_data_dirs, 224, 224, 16)
    #client_train_data[1],client_test_data[1],client_class_names[1] = data_load("./face_datasets/train/huangtingqiao", "./face_datasets/test/huangtingqiao", 224, 224, 16)


    #初始全局模型
    global_model = model_load(IMG_SHAPE=(224, 224, 3), class_num=len(class_names))
    begin_time = time()  
     
    global_test_accuracy = []  # 保存全局模型在测试集上的准确率
    global_test_loss = []  # 保存全局模型在测试集上的损失
    tglobal_test_accuracy = []  # 保存全局模型在测试集上的准确率
    tglobal_test_loss = []  # 保存全局模型在测试集上的损失
    evaluation_results_file = "X_result/global_iid_10r.txt"
    #全局训练 
    for e in range(global_epochs):     
        print("-------------------Global Epoch{}------------------ ".format(e))
        global_name = f'global_{e}'
        client_models = []  
        #客户端训练
        for i in range(num_clients): 
            print("---client {} is training---".format(i))    
            client_name = f'client_{i}'  

            client_train_ds= client_train_data[i]  # 载入客户端的训练数据
            client_test_ds = client_test_data[i]  # 载入客户端的验证数据          
            #初始客户端模型
            client_model = model_load(IMG_SHAPE=(224, 224, 3), class_num=len(class_names))
            #client_model = model_load(IMG_SHAPE=(224, 224, 3), class_num=len(class_names))
            client_model.set_weights(global_model.get_weights())

            history=client_model.fit(client_train_ds, validation_data=client_test_ds,epochs=client_epochs)
            client_model.save(log_dir + f'{global_name}_{client_name}_easy2.h5')
       
            client_models.append(client_model)  # 将客户端模型添加到列表中 
            show_loss_acc(history)
        # 聚合更新到全局模型
        global_weights = global_model.get_weights()
        # 计算各客户端权重的平均值
        averaged_weights = [sum(client_model.get_weights()[i] for client_model in client_models) / len(client_models) for i in range(len(global_weights))]

        # 设置全局模型的权重为平均值
        global_model.set_weights(averaged_weights)
        global_model.save(os.path.join(folder_path, "federated_model_iid_10r.h5"))
        end_time = time()
        run_time = end_time - begin_time
        print(f'Epoch {e+ 1}, Time: {run_time:.2f} seconds')

        #全局模型在测试集上的准确率
        global_loss, global_accuracy = global_model.evaluate(test_ds)
        global_test_accuracy.append(global_accuracy)
        global_test_loss.append(global_loss)
        print(f'Global Model - Test Loss: {global_loss}, Test Accuracy: {global_accuracy}')
        with open(evaluation_results_file, "a") as file:
            file.write(f"Epoch {e+1} - Test Loss: {global_loss}, Test Accuracy: {global_accuracy}\n")

                #全局模型在测试集上的准确率
        tglobal_loss, tglobal_accuracy = global_model.evaluate(train_ds)
        tglobal_test_accuracy.append(tglobal_accuracy)
        tglobal_test_loss.append(tglobal_loss)
        print(f'Global Model - Train Loss: {tglobal_loss}, Train Accuracy: {tglobal_accuracy}')
        with open(evaluation_results_file, "a") as file:
            file.write(f"Epoch {e+1} - Train Loss: {tglobal_loss}, Train Accuracy: {tglobal_accuracy}\n")  

    plt.figure()
    plt.plot(range(1, global_epochs + 1), global_test_accuracy, marker='o')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.xticks(range(1, global_epochs + 1))
    plt.grid(True)
    plt.savefig('X_result/global_acc_iid_10r.png', dpi=100)
    #plt.show()

    # 绘制全局模型在测试集上的损失随 epoch 变化的图像
    plt.figure()
    plt.plot(range(1, global_epochs + 1), global_test_loss, marker='o')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.xticks(range(1, global_epochs + 1))
    plt.grid(True)
    plt.savefig('X_result/global_loss_iid_10r.png', dpi=100)
    #plt.show()

    plt.figure()
    plt.plot(range(1, global_epochs + 1), tglobal_test_accuracy, marker='o')
    plt.title('Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.xticks(range(1, global_epochs + 1))
    plt.grid(True)
    plt.savefig('X_result/global_acc_iid_10tr.png', dpi=100)
    #plt.show()

    # 绘制全局模型在测试集上的损失随 epoch 变化的图像
    plt.figure()
    plt.plot(range(1, global_epochs + 1), tglobal_test_loss, marker='o')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.xticks(range(1, global_epochs + 1))
    plt.grid(True)
    plt.savefig('X_result/global_loss_iid_8tr.png', dpi=100)


