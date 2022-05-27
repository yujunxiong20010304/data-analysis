# 导入包
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# 加载数据
fashion_mnist = keras.datasets.fashion_mnist
# 训练集和测试集的划分
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)
# 数据预处理,将每个像素点压缩到0到1之间
train_images = train_images / 255.0
test_images = test_images / 255.0


# 搭建神经网络

def create_model():
    # 创建神经网络模型
    model = tf.keras.models.Sequential([
        # 输入 隐藏 输出
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10)
    ])
    # 编译模型
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()]
                  )
    # 返回模型
    return model


# 构建模型
new_model = create_model()
# 训练模型
new_model.fit(train_images, train_labels, epochs=30)
# 保存模型
new_model.save("model/my_mode13.h5")

create_model()
'''# 打印图象
plt.figure()
plt.xticks([])
plt.yticks([])
plt.imshow(train_images[100])
plt.show()'''
