
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 類別數量(MNIST 共 10 個數字)
category = 10

# 載入 MNIST 資料集（包括訓練和測試資料）
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 標準化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# One-hot encoding
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2 = tf.keras.utils.to_categorical(y_test, category)

print(y_train[0])
print("↧")
print(y_train2[0])

# CNN
model = tf.keras.Sequential([
    # 第一層卷積層 (Conv2D)，使用 ReLU 激活函數
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),padding="same", activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)), # 使用 2x2 最大池化層 (MaxPooling)
    # 第二層卷積層 (Conv2D)，增加特徵提取能力
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),padding="same", activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)), # 第二層最大池化層
    tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),padding="same", activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)), # 第二層最大池化層
    # 將 2D 特徵轉為 1D 特徵向量 (Flatten)
    tf.keras.layers.Flatten(),
    # 使用 Dropout 防止過擬合
    tf.keras.layers.Dropout(rate=0.5),
    # 全連接層 (Dense Layer)
    tf.keras.layers.Dense(256, activation='relu'),
    # 最後一層輸出層
    tf.keras.layers.Dense(units=category, activation=tf.nn.softmax)
])
model.summary()

# 設定損失函數、optimizer
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# 訓練模型
history = model.fit(x_train, y_train2, batch_size=128, epochs=30)

# 測量模型在測試集上的表現
score = model.evaluate(x_test, y_test2, batch_size=128)
print("score:", score)

# 預測測試集前 4 筆的結果
predict = model.predict(x_test)
print("Ans:", np.argmax(predict[0]), np.argmax(predict[1]), np.argmax(predict[2]), np.argmax(predict[3]))
print("y_test", y_test[:4])

# 繪製訓練準確率與損失折線圖
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.title('Model Accuracy and Loss')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
