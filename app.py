#后端，前端网页在templates里
# -*- coding: utf-8 -*-
import os
from PIL import Image
import numpy as np
import json
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import cv2
app = Flask(__name__)
CORS(app)  # 解决跨域问题


weights_path = "./X_models/federated_model_iid_10r.h5"
class_json_path = "./static/js/class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."


# create model
model = tf.keras.models.load_model(weights_path)
# load model weights


# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)


# def load_image(image):
#   img = Image.open(image)
#   img = img.resize((128, 128))
#   img = np.array(img)
#   img = img / 255
#   img = img.reshape((1,) + img.shape + (1,))  # reshape img to size(1, 128, 128, 1)
#   return img


def get_prediction():
    try:
        img_init = cv2.imread("test.png")  # 打开图片
        h, w, c = img_init.shape
        scale = 400 / h
        img = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)  # 将图片的大小统一调整到400的高，方便界面显示
        #img = cv2.resize(img_init, (0, 0))  # 将图片的大小统一调整到400的高，方便界面显示
        img = cv2.resize(img, (224, 224))  # 将图片大小调整到224*224用于模型推理
        outputs = model.predict(img.reshape(1, 224, 224, 3)).reshape(-1,)  # 将图片输入模型得到结果
        # result_index = int(np.argmax(outputs))
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p)) for index, p in enumerate(outputs)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["file"]
    image.save("test.png")
    img_bytes = image.read()
    info = get_prediction()
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)