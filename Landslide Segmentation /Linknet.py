import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation, Concatenate, UpSampling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def linknet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Decoder
    up5 = UpSampling2D(size=(2, 2))(pool4)
    up5 = Conv2D(512, 2, activation='relu', padding='same')(up5)
    merge5 = Concatenate()([conv4, up5])
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(merge5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(256, 2, activation='relu', padding='same')(up6)
    merge6 = Concatenate()([conv3, up6])
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(128, 2, activation='relu', padding='same')(up7)
    merge7 = Concatenate()([conv2, up7])
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(64, 2, activation='relu', padding='same')(up8)
    merge8 = Concatenate()([conv1, up8])
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv8)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare data
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
        mask = mask / 255.0  # Convert mask image to binary image
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

# Prepare dataset
def prepare_dataset(data_path):
    images, masks = load_data(data_path)
    images = images.astype('float32') / 255.0  # Normalization
    return images, masks

# Compute metrics
def compute_metrics(model, images, masks):
    preds = model.predict(images)
    preds_binary = (preds > 0.5).astype(np.float32)
    
    # Calculate IoU
    intersection = np.logical_and(masks, preds_binary)
    union = np.logical_or(masks, preds_binary)
    iou = np.sum(intersection) / np.sum(union)
    
    # Calculate Dice coefficient
    dice = 2 * np.sum(intersection) / (np.sum(masks) + np.sum(preds_binary))
    
    # Calculate accuracy
    accuracy = accuracy_score(masks.flatten(), preds_binary.flatten())
    
    # Calculate precision
    precision = precision_score(masks.flatten(), preds_binary.flatten())
    
    # Calculate recall
    recall = recall_score(masks.flatten(), preds_binary.flatten())
    
    # Calculate F1 score
    f1 = f1_score(masks.flatten(), preds_binary.flatten())
    
    # Overall Accuracy (OA)
    oa = (intersection.sum() / masks.size) * 10
    
    # Kappa coefficient
    kappa = cohen_kappa_score(masks.flatten(), preds_binary.flatten())
    
    return iou, dice, accuracy, precision, recall, f1, oa, kappa, preds_binary

# Set data path
data_path = 'Bijie_landslide_dataset/Bijie-landslide-dataset/landslide'

# Load and prepare dataset
images, masks = prepare_dataset(data_path)

# Define input shape
input_shape = (256, 256, 3)

# Use modified LinkNet model
linknet_input = Input(shape=input_shape, name='linknet_input')
linknet_output = linknet_model(input_shape)(linknet_input)

combined_output = linknet_output
combined_model = Model(inputs=linknet_input, outputs=combined_output)

combined_model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train combined model
combined_model.fit(images, masks, batch_size=4, epochs=100, validation_split=0.3)

# Make predictions
predictions = combined_model.predict(images)

# Visualize results
plot_images(images, masks, predictions)

# Compute evaluation metrics
iou, dice, accuracy, precision, recall, f1, oa, kappa, preds_binary = compute_metrics(combined_model, images, masks)

print("IoU:", iou)
print("Dice coefficient:", dice)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Overall Accuracy (OA):", oa)
print("Kappa coefficient:", kappa)
