# @Brief   : 模型测试代码，测试会生成热力图，
# 热力图会保存在result目录下，
# results的存放的都是集中式的训练，没有s的result存放的是联邦学习的结果

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# 数据加载，分别从训练的数据集的文件夹和测试的文件夹中加载训练集和验证集
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
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


# 测试mobilenet准确率
def test_mobilenet():
    # todo 加载数据, 修改为你自己的数据集的路径
    train_ds, test_ds, class_names = data_load("../X_datasets/train",
                                               "../X_datasets/test", 224, 224, 16)
    # todo 加载模型，修改为你的模型名称
    model = tf.keras.models.load_model("X_models/normal_cnn.h5")
    # model.summary()
    # 测试
    loss, accuracy = model.evaluate(test_ds)
    # 输出结果
    print('Mobilenet test accuracy :', accuracy)

    test_real_labels = []
    test_pre_labels = []
    for test_batch_images, test_batch_labels in test_ds:
        test_batch_labels = test_batch_labels.numpy()
        test_batch_pres = model.predict(test_batch_images)
        # print(test_batch_pres)

        test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
        test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
        # print(test_batch_labels_max)
        # print(test_batch_pres_max)
        # 将推理对应的标签取出
        for i in test_batch_labels_max:
            test_real_labels.append(i)

        for i in test_batch_pres_max:
            test_pre_labels.append(i)
        # break

    # print(test_real_labels)
    # print(test_pre_labels)
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    # print(heat_maps_sum)
    print()
    heat_maps_float = heat_maps / heat_maps_sum
    print(heat_maps_float)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name="results/heatmap_mobilenet.png")


# 测试cnn模型准确率
def test_cnn():
    # todo 加载数据集， 修改为你的数据集的路径
    train_ds, test_ds, class_names = data_load("./X_datasets/train",
                                              "./X_datasets/test", 224, 224, 16)
    # todo 加载模型，修改为你的模型名称
    model = tf.keras.models.load_model("X_models/normal_cnn.h5")
    # model.summary()
    # 测试
    loss, accuracy = model.evaluate(test_ds)
    # 输出结果
    print('CNN test accuracy :', accuracy)

    # 对模型分开进行推理
    test_real_labels = []
    test_pre_labels = []
    for test_batch_images, test_batch_labels in test_ds:
        test_batch_labels = test_batch_labels.numpy()
        test_batch_pres = model.predict(test_batch_images)
        # print(test_batch_pres)

        test_batch_labels_max = np.argmax(test_batch_labels, axis=1)
        test_batch_pres_max = np.argmax(test_batch_pres, axis=1)
        # print(test_batch_labels_max)
        # print(test_batch_pres_max)
        # 将推理对应的标签取出
        for i in test_batch_labels_max:
            test_real_labels.append(i)

        for i in test_batch_pres_max:
            test_pre_labels.append(i)
        # break

    # print(test_real_labels)
    # print(test_pre_labels)
    class_names_length = len(class_names)
    heat_maps = np.zeros((class_names_length, class_names_length))
    for test_real_label, test_pre_label in zip(test_real_labels, test_pre_labels):
        heat_maps[test_real_label][test_pre_label] = heat_maps[test_real_label][test_pre_label] + 1

    print(heat_maps)
    heat_maps_sum = np.sum(heat_maps, axis=1).reshape(-1, 1)
    # print(heat_maps_sum)
    print()
    heat_maps_float = heat_maps / heat_maps_sum
    print(heat_maps_float)
    # title, x_labels, y_labels, harvest
    show_heatmaps(title="heatmap", x_labels=class_names, y_labels=class_names, harvest=heat_maps_float,
                  save_name="X_results/heatmap_D.png")


def show_heatmaps(title, x_labels, y_labels, harvest, save_name):
    # 这里是创建一个画布
    fig, ax = plt.subplots()
    # cmap https://blog.csdn.net/ztf312/article/details/102474190
    im = ax.imshow(harvest, cmap="OrRd")
    # 这里是修改标签
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)

    # 因为x轴的标签太长了，需要旋转一下，更加好看
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 添加每个热力块的具体数值
    # Loop over data dimensions and create text annotations.
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            text = ax.text(j, i, round(harvest[i, j], 2),
                           ha="center", va="center", color="black")
    ax.set_xlabel("Predict label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(save_name, dpi=100)
    # plt.show()


if __name__ == '__main__':
    #test_mobilenet()
    test_cnn()
