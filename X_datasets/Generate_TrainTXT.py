import os

Name_label = []  # 姓名标签
path = './X_datasets/train/'  # 数据集文件路径

# 检查数据集文件夹是否存在
if not os.path.exists(path):
    print(f"Error: 数据集文件夹 {path} 不存在！")
    exit()

dir = os.listdir(path)  # 列出所有人
label = 0  # 设置计数器

# 数据写入
with open('./X_datasets/train.txt', 'w') as f:
    for name in dir:
        if name == '.DS_Store':
            continue  # 跳过 .DS_Store 文件
        Name_label.append(name)
        print(Name_label[label])
        after_generate = os.listdir(path + '/' + name)
        for image in after_generate:
            if image.endswith(".jpg"):
                f.write(image + ";" + str(label) + "\n")
        label += 1
