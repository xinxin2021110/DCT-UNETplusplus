import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization, Activation, Conv2DTranspose, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def vgg16_fcn(input_size=(256, 256, 3)):
    vgg16_base = VGG16(include_top=False, weights='imagenet', input_shape=input_size)

    for layer in vgg16_base.layers:
        layer.trainable = False

    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(vgg16_base.output)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    x = UpSampling2D(size=(32, 32))(x)

    model = Model(inputs=vgg16_base.input, outputs=x)
    return model

# 使用修改后的VGG16-FCN模型
input_shape = (256, 256, 3)
fcn_model = vgg16_fcn(input_shape)

fcn_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# 训练FCN模型
fcn_model.fit(images, masks, batch_size=4, epochs=100, validation_split=0.3)

predictions = fcn_model.predict(images)

# 可视化结果
plot_images(images, masks, predictions)

# 计算评估指标
iou, dice, accuracy, precision, recall, f1, oa, kappa, preds_binary = compute_metrics(fcn_model, images, masks)
print("IoU:", iou)
print("Dice coefficient:", dice)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Overall Accuracy (OA):", oa)
print("Kappa coefficient:", kappa)
