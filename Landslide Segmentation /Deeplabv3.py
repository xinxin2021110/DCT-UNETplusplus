import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import DepthwiseConv2D
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def load_data(data_path):
    image_dir = os.path.join(data_path, 'image')
    mask_dir = os.path.join(data_path, 'mask')
    image_names = [name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))]
    mask_names = [name for name in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, name))]
    images = []
    masks = []
    for img_name, mask_name in zip(image_names, mask_names):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
        mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(256, 256), color_mode='grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        mask = tf.keras.preprocessing.image.img_to_array(mask)
        mask = mask / 255.0  # 将掩模图像转换为二值图像
        images.append(img)
        masks.append(mask)
    return np.array(images), np.array(masks)

def plot_images(images, masks, preds):
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(3, 3, i+4)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title('Mask Image')
        plt.axis('off')
        
        plt.subplot(3, 3, i+7)
        plt.imshow(preds[i].squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
    plt.show()

def prepare_dataset(data_path):
    images, masks = load_data(data_path)
    images = images.astype('float32') / 255.0  # 归一化
    return images, masks

def deeplabv3_model(input_size=(256, 256, 3)):
    base_model = MobileNetV2(input_shape=input_size, include_top=False, weights='imagenet')
    
    # Encoder部分
    inputs = Input(input_size)
    x = base_model(inputs)
    
    # Decoder部分
    x = Conv2D(256, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(4, 4))(x)  # 使用双线性上采样将特征图放大4倍
    
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(4, 4))(x)  # 使用双线性上采样将特征图放大4倍
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(4, 4))(x)  # 使用双线性上采样将特征图放大4倍
    
    x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)  # 输出掩模图像
    
    model = Model(inputs=inputs, outputs=x)
    return model

def compute_metrics(model, images, masks):
    preds = model.predict(images)
    preds_binary = (preds > 0.5).astype(np.float32)
    
    # 计算IoU
    intersection = np.logical_and(masks, preds_binary)
    union = np.logical_or(masks, preds_binary)
    iou = np.sum(intersection) / np.sum(union)
    
    # 计算Dice系数
    dice = 2 * np.sum(intersection) / (np.sum(masks) + np.sum(preds_binary))
    
    # 计算准确率
    accuracy = accuracy_score(masks.flatten(), preds_binary.flatten())
    
    # 计算精确率
    precision = precision_score(masks.flatten(), preds_binary.flatten())
    
    # 计算召回率
    recall = recall_score(masks.flatten(), preds_binary.flatten())
    
    # 计算F1分数
    f1 = f1_score(masks.flatten(), preds_binary.flatten())
    
    # 总体准确率（OA）
    oa = (intersection.sum() / masks.size) * 10
    
    # Kappa系数
    kappa = cohen_kappa_score(masks.flatten(), preds_binary.flatten())
    
    return iou, dice, accuracy, precision, recall, f1, oa, kappa, preds_binary

data_path = 'Bijie_landslide_dataset/Bijie-landslide-dataset/landslide'
images, masks = prepare_dataset(data_path)

input_shape = (256, 256, 3)

# 使用修改后的DeepLabv3+模型
deeplabv3_input = Input(shape=input_shape, name='deeplabv3_input')
deeplabv3_output = deeplabv3_model(input_shape)(deeplabv3_input)

combined_output = deeplabv3_output[:, :256, :256, :]  # 裁剪输出以匹配输入尺寸
combined_model = Model(inputs=deeplabv3_input, outputs=combined_output)

combined_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# 训练组合模型
combined_model.fit(images, masks, batch_size=4, epochs=100, validation_split=0.3)

predictions = combined_model.predict(images)

# 可视化结果
plot_images(images, masks, predictions)

# 计算评估指标
iou, dice, accuracy, precision, recall, f1, oa, kappa, preds_binary = compute_metrics(combined_model, images, masks)

print("IoU:", iou)
print("Dice coefficient:", dice)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Overall Accuracy (OA):", oa)
print("Kappa coefficient:", kappa)
