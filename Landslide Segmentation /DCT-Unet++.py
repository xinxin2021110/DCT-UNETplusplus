import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, LayerNormalization, Dense, MultiHeadAttention, GlobalAveragePooling2D, Add, Reshape, Conv2DTranspose, Activation, GlobalMaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_gaussian, create_pairwise_bilateral

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

def transformer_block(inputs, num_heads, key_dim):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    ffn_output = tf.keras.Sequential([
        Dense(4 * key_dim, activation='relu'),
        Dense(inputs.shape[-1]),  # 保持与输入的最后一个维度一致
    ])(x)
    x = Add()([x, ffn_output])
    return x

def cbam_block(input_tensor, reduction_ratio=16):
    channel = input_tensor.shape[-1]
    # Channel Attention Module
    shared_layer_one = Dense(channel // reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_tensor)    
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    
    max_pool = GlobalMaxPooling2D()(input_tensor)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    cbam_feature = tf.keras.layers.multiply([input_tensor, cbam_feature])
    
    # Spatial Attention Module
    avg_pool = tf.reduce_mean(cbam_feature, axis=3, keepdims=True)
    max_pool = tf.reduce_max(cbam_feature, axis=3, keepdims=True)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool], axis=3)
    cbam_feature = Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    
    cbam_feature = tf.keras.layers.multiply([cbam_feature, input_tensor])
    return cbam_feature

# 新增边界增强损失函数
def boundary_loss(y_true, y_pred):
    # 计算边界增强损失
    boundary_true = tf.image.sobel_edges(y_true)
    boundary_pred = tf.image.sobel_edges(y_pred)
    boundary_loss = tf.reduce_mean(tf.square(boundary_true - boundary_pred))
    return boundary_loss

def dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def iou_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def unetplusplus_transformer_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=1)(inputs)
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=1)(conv1_1)
    conv1_1 = cbam_block(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv2_1 = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=2)(pool1)
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=2)(conv2_1)
    conv2_1 = cbam_block(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1 = Conv2D(256, 3, activation='relu', padding='same', dilation_rate=3)(pool2)
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same', dilation_rate=3)(conv3_1)
    conv3_1 = cbam_block(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_1)

    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', dilation_rate=4)(pool3)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same', dilation_rate=4)(conv4_1)
    conv4_1 = cbam_block(conv4_1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_1)

    conv5_1 = Conv2D(1024, 3, activation='relu', padding='same', dilation_rate=5)(pool4)
    conv5_1 = Conv2D(1024, 3, activation='relu', padding='same', dilation_rate=5)(conv5_1)
    conv5_1 = cbam_block(conv5_1)

    # Decoder
    up4_1 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5_1)
    up4_1 = concatenate([up4_1, conv4_1], axis=3)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same')(up4_1)
    conv4_1 = Conv2D(512, 3, activation='relu', padding='same')(conv4_1)
    conv4_1 = cbam_block(conv4_1)

    up3_1 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4_1)
    up3_1 = concatenate([up3_1, conv3_1], axis=3)
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same')(up3_1)
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same')(conv3_1)
    conv3_1 = cbam_block(conv3_1)

    up2_1 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3_1)
    up2_1 = concatenate([up2_1, conv2_1], axis=3)
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same')(up2_1)
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same')(conv2_1)
    conv2_1 = cbam_block(conv2_1)

    up1_1 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv2_1)
    up1_1 = concatenate([up1_1, conv1_1], axis=3)
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same')(up1_1)
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same')(conv1_1)
    conv1_1 = cbam_block(conv1_1)

    # Second Level Nested Skip Connections
    up4_2 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5_1)
    up4_2 = concatenate([up4_2, conv4_1], axis=3)
    conv4_2 = Conv2D(512, 3, activation='relu', padding='same')(up4_2)
    conv4_2 = Conv2D(512, 3, activation='relu', padding='same')(conv4_2)
    conv4_2 = cbam_block(conv4_2)

    up3_2 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4_2)
    up3_2 = concatenate([up3_2, conv3_1], axis=3)
    conv3_2 = Conv2D(256, 3, activation='relu', padding='same')(up3_2)
    conv3_2 = Conv2D(256, 3, activation='relu', padding='same')(conv3_2)
    conv3_2 = cbam_block(conv3_2)

    up2_2 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3_2)
    up2_2 = concatenate([up2_2, conv2_1], axis=3)
    conv2_2 = Conv2D(128, 3, activation='relu', padding='same')(up2_2)
    conv2_2 = Conv2D(128, 3, activation='relu', padding='same')(conv2_2)
    conv2_2 = cbam_block(conv2_2)

    up1_2 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv2_2)
    up1_2 = concatenate([up1_2, conv1_1], axis=3)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='same')(up1_2)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='same')(conv1_2)
    conv1_2 = cbam_block(conv1_2)

    # Third Level Nested Skip Connections
    up3_3 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4_2)
    up3_3 = concatenate([up3_3, conv3_2, conv3_1], axis=3)
    conv3_3 = Conv2D(256, 3, activation='relu', padding='same')(up3_3)
    conv3_3 = Conv2D(256, 3, activation='relu', padding='same')(conv3_3)
    conv3_3 = cbam_block(conv3_3)

    up2_3 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3_3)
    up2_3 = concatenate([up2_3, conv2_2, conv2_1], axis=3)
    conv2_3 = Conv2D(128, 3, activation='relu', padding='same')(up2_3)
    conv2_3 = Conv2D(128, 3, activation='relu', padding='same')(conv2_3)
    conv2_3 = cbam_block(conv2_3)

    up1_3 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv2_3)
    up1_3 = concatenate([up1_3, conv1_2, conv1_1], axis=3)
    conv1_3 = Conv2D(64, 3, activation='relu', padding='same')(up1_3)
    conv1_3 = Conv2D(64, 3, activation='relu', padding='same')(conv1_3)
    conv1_3 = cbam_block(conv1_3)

    # Transformer Block
    transformer_input = Reshape((conv5_1.shape[1] * conv5_1.shape[2], conv5_1.shape[3]))(conv5_1)
    transformer_output = transformer_block(transformer_input, num_heads=8, key_dim=128)
    transformer_output = Reshape((conv5_1.shape[1], conv5_1.shape[2], conv5_1.shape[3]))(transformer_output)

    # Segmentation Output
    segmentation_output = Conv2D(1, 1, activation='sigmoid', name='seg_output')(conv1_3)
    
    # Edge Detection Head
    edge_head = Conv2D(64, 3, activation='relu', padding='same')(conv1_3)
    edge_head = Conv2D(1, 1, activation='sigmoid', name='edge_output')(edge_head)

    model = Model(inputs=inputs, outputs=[segmentation_output, edge_head])

    # 修改损失函数为复合损失函数
    def composite_loss(y_true, y_pred):
        cross_entropy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice_loss = 1 - dice_coefficient(y_true, y_pred)
        iou_loss = 1 - iou_coefficient(y_true, y_pred)
        return cross_entropy_loss + dice_loss + iou_loss

    # 结合边界增强损失
    combined_loss = lambda y_true, y_pred: composite_loss(y_true, y_pred) + boundary_loss(y_true, y_pred)

    model.compile(optimizer=Adam(lr=1e-4), 
                  loss={'seg_output': combined_loss, 'edge_output': 'binary_crossentropy'}, 
                  metrics={'seg_output': ['accuracy'], 'edge_output': ['accuracy']})
    return model

def apply_crf(image, prediction):
    # 将预测结果转换为概率分布
    prediction = prediction.squeeze()  # 将 (256, 256, 1) 转换为 (256, 256)
    softmax = np.zeros((2, prediction.shape[0], prediction.shape[1]), dtype=np.float32)
    softmax[0, :, :] = 1 - prediction
    softmax[1, :, :] = prediction

    # 创建CRF模型
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)

    # 将unary能量添加到CRF模型中
    unary = unary_from_softmax(softmax)
    d.setUnaryEnergy(unary)

    # 增加对比度和位置的一对一对特征
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    d.addPairwiseEnergy(feats, compat=3)

    # 增加颜色特征
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13), img=image, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)

    # 执行推理
    Q = d.inference(5)

    # 获得最终分割结果
    map = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    return map

def compute_metrics(model, images, masks):
    preds = model.predict(images)
    seg_preds = preds[0]
    edge_preds = preds[1]
    seg_preds_binary = (seg_preds > 0.5).astype(np.float32)

    # 应用CRF后处理
    crf_preds = np.zeros(seg_preds_binary.shape[:3], dtype=np.float32)
    for i in range(images.shape[0]):
        crf_preds[i] = apply_crf(images[i].astype(np.uint8), seg_preds_binary[i])

    # 确保 masks 和 crf_preds 的形状匹配
    masks = masks.squeeze()  # 将 (770, 256, 256, 1) 转换为 (770, 256, 256)

    # 计算评估指标
    intersection = np.logical_and(masks, crf_preds)
    union = np.logical_or(masks, crf_preds)
    iou = np.sum(intersection) / np.sum(union)

    dice = 2 * np.sum(intersection) / (np.sum(masks) + np.sum(crf_preds))

    accuracy = accuracy_score(masks.flatten(), crf_preds.flatten())

    precision = precision_score(masks.flatten(), crf_preds.flatten())

    recall = recall_score(masks.flatten(), crf_preds.flatten())

    f1 = f1_score(masks.flatten(), crf_preds.flatten())

    oa = (intersection.sum() / masks.size) * 100

    kappa = cohen_kappa_score(masks.flatten(), crf_preds.flatten())

    return iou, dice, accuracy, precision, recall, f1, oa, kappa, crf_preds, edge_preds

# 加载数据
data_path = 'Bijie_landslide_dataset/Bijie-landslide-dataset/landslide'
images, masks = prepare_dataset(data_path)

# 创建并编译UNet++-Transformer多任务模型
input_shape = (256, 256, 3)
multitask_model = unetplusplus_transformer_model(input_shape)

# 训练模型
multitask_model.fit(images, {'seg_output': masks, 'edge_output': masks}, batch_size=4, epochs=100, validation_split=0.3)

# 可视化结果
predictions = multitask_model.predict(images)
plot_images(images, masks, predictions[0])

# 计算评估指标
iou, dice, accuracy, precision, recall, f1, oa, kappa, crf_preds, edge_preds = compute_metrics(multitask_model, images, masks)

print("IoU:", iou)
print("Dice coefficient:", dice)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Overall Accuracy (OA):", oa)
print("Kappa coefficient:", kappa)

# 可视化结果
plot_images(images, masks, crf_preds)
