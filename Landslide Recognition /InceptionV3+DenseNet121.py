import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, DenseNet121
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 数据集路径
data_dir = "Bijie_landslide_dataset/Bijie-landslide-dataset"

# 定义数据生成器
datagen = ImageDataGenerator(rescale=1./255)

# 加载数据集
batch_size = 32
image_height = 224
image_width = 224

# 加载landslide数据
landslide_dir = os.path.join(data_dir, "landslide")
landslide_dem_dir = os.path.join(landslide_dir, "dem")
landslide_image_dir = os.path.join(landslide_dir, "image")
landslide_mask_dir = os.path.join(landslide_dir, "mask")

# 加载non-landslide数据
non_landslide_dir = os.path.join(data_dir, "non-landslide")
non_landslide_dem_dir = os.path.join(non_landslide_dir, "dem")
non_landslide_image_dir = os.path.join(non_landslide_dir, "image")

# 读取并预处理数据函数
def read_and_preprocess_data(image_dir, dem_dir=None):
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if not filename.startswith('.')]
    images = [tf.keras.preprocessing.image.load_img(img_path, target_size=(image_height, image_width)) for img_path in image_paths]
    images = [tf.keras.preprocessing.image.img_to_array(img) for img in images]
    images = np.array(images)

    if dem_dir:
        dem_paths = [os.path.join(dem_dir, filename) for filename in os.listdir(dem_dir) if not filename.startswith('.')]
        dem_images = [tf.keras.preprocessing.image.load_img(dem_path, target_size=(image_height, image_width)) for dem_path in dem_paths]
        dem_images = [tf.keras.preprocessing.image.img_to_array(dem) for dem in dem_images]
        dem_images = np.array(dem_images)
        return images, dem_images
    else:
        return images

# 读取并预处理landslide数据
landslide_images, landslide_dem_images = read_and_preprocess_data(landslide_image_dir, landslide_dem_dir)
landslide_labels = np.ones(len(landslide_images))

# 读取并预处理non-landslide数据
non_landslide_images = read_and_preprocess_data(non_landslide_image_dir)
non_landslide_labels = np.zeros(len(non_landslide_images))

# 合并数据集
images = np.concatenate((landslide_images, non_landslide_images), axis=0)
dem_images = np.concatenate((landslide_dem_images, np.zeros_like(non_landslide_images)), axis=0)  # 对于non-landslide数据，DEM图像全为0
labels = np.concatenate((landslide_labels, non_landslide_labels), axis=0)

# 切分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 构建图像处理模型（使用InceptionV3）
input_image = Input(shape=(image_height, image_width, 3))
base_model_image = InceptionV3(include_top=False, weights='imagenet', input_tensor=input_image)
image_features = GlobalAveragePooling2D()(base_model_image.output)
image_output = Dense(256, activation='relu')(image_features)

# 构建DEM处理模型（使用DenseNet121）
input_dem = Input(shape=(image_height, image_width, 3))
base_model_dem = DenseNet121(include_top=False, weights='imagenet', input_tensor=input_dem)
dem_features = GlobalAveragePooling2D()(base_model_dem.output)
dem_output = Dense(256, activation='relu')(dem_features)

# 将两个模型输出连接
merged = Concatenate()([image_output, dem_output])

# 最终的分类层
output = Dense(1, activation='sigmoid')(merged)

# 定义模型
model = Model(inputs=[input_image, input_dem], outputs=output)

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit([X_train, dem_images[:len(X_train)]], y_train, batch_size=batch_size, epochs=50, validation_split=0.2)

# 评估模型
test_loss, test_acc = model.evaluate([X_test, dem_images[len(X_train):]], y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# 模型预测
y_pred = model.predict([X_test, dem_images[len(X_train):]])
y_pred_classes = np.round(y_pred)

# 输出性能报告结果
report = classification_report(y_test, y_pred_classes, target_names=['non-landslide', 'landslide'])
print("Classification Report:\n", report)

# 输出其他性能指标
accuracy = np.mean(y_test == y_pred_classes)
precision = np.mean(y_pred_classes[y_test == 1] == 1)
recall = np.mean(y_pred_classes[y_test == 1] == y_test[y_test == 1])
f1_score = 2 * precision * recall / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
