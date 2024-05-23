import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout, Conv2D, SeparableConv2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Reshape, Conv2D, Multiply, LayerNormalization, MultiHeadAttention, Add, Flatten
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint

data_path = "Bijie_landslide_dataset/Bijie-landslide-dataset"
img_size = (224, 224)
batch_size = 8
num_classes = 2


def spatial_attention_module(input_feature):
    # 使用一个1x1的卷积层来生成注意力图
    attention = Conv2D(1, (1, 1), activation='sigmoid')(input_feature)
    # 将注意力图乘以原特征图来增强重要特征
    enhanced_feature = Multiply()([input_feature, attention])
    return enhanced_feature

def build_image_model(input_shape):
    img_input = Input(shape=input_shape, name='image_input')
    img_base_model = EfficientNetB7(include_top=False, input_tensor=img_input, weights='imagenet')
    img_features = img_base_model.output
    img_features = spatial_attention_module(img_features)
    img_features = GlobalAveragePooling2D()(img_features)
    return Model(inputs=img_input, outputs=img_features, name='ImageModel')

def load_images(folder, filenames, color_mode='rgb'):
    images = []
    for filename in filenames:
        if filename.startswith('.'):  # 过滤掉隐藏文件
            continue
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=img_size, color_mode=color_mode)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)


def transformer_block(x, num_heads=4):
    # Layer Normalization
    x_norm = LayerNormalization()(x)
    # Multi-head self-attention
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=64)(x_norm, x_norm)
    # Skip connection
    x = Add()([x, attn_output])
    # Another layer normalization
    x_norm = LayerNormalization()(x)
    # Feed-forward network
    ff_output = Dense(units=512, activation='relu')(x_norm)
    # Final skip connection
    output = Add()([x, ff_output])
    return output

def prepare_dataset(data_path):
    landslide_path = os.path.join(data_path, "landslide")
    non_landslide_path = os.path.join(data_path, "non-landslide")
    
    landslide_filenames = [filename for filename in os.listdir(os.path.join(landslide_path, "image")) if not filename.startswith('.')]
    non_landslide_filenames = [filename for filename in os.listdir(os.path.join(non_landslide_path, "image")) if not filename.startswith('.')]

    landslide_images = load_images(os.path.join(landslide_path, "image"), landslide_filenames)
    landslide_dems = load_images(os.path.join(landslide_path, "dem"), landslide_filenames, color_mode='grayscale')
    non_landslide_images = load_images(os.path.join(non_landslide_path, "image"), non_landslide_filenames)
    non_landslide_dems = load_images(os.path.join(non_landslide_path, "dem"), non_landslide_filenames, color_mode='grayscale')

    landslide_labels = np.ones(len(landslide_filenames), dtype=int)
    non_landslide_labels = np.zeros(len(non_landslide_filenames), dtype=int)

    images = np.concatenate([landslide_images, non_landslide_images], axis=0)
    dems = np.concatenate([landslide_dems, non_landslide_dems], axis=0)
    labels = np.concatenate([landslide_labels, non_landslide_labels], axis=0)

    return images, dems, labels


images, dems, labels = prepare_dataset(data_path)
labels = to_categorical(labels, num_classes=num_classes)

x_img_train, x_img_val, x_dem_train, x_dem_val, y_train, y_val = train_test_split(images, dems, labels, test_size=0.2, random_state=42)

def build_dem_cnn_model(input_shape):
    inputs = Input(shape=input_shape, name='dem_input')
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, -1))(x)
    x = transformer_block(x)
    x = Flatten()(x)
    return Model(inputs=inputs, outputs=x, name='Advanced_DEM_CNN')


def build_vae_feature_merger(input_dim):
    input_layer = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(input_layer)
    z_mean = Dense(64)(x)
    z_log_var = Dense(64)(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])
    encoder = Model(input_layer, [z_mean, z_log_var, z], name='vae_merger')
    return encoder



# Build the dual input model
def build_dual_input_model(input_shape):
    img_model = build_image_model(input_shape)
    img_features = img_model.output

    dem_input_shape = (img_size[0], img_size[1], 1)  
    dem_model = build_dem_cnn_model(dem_input_shape)
    dem_features = dem_model.output

    merged_features = Concatenate()([img_features, dem_features])
    vae_encoder = build_vae_feature_merger(merged_features.shape[1])
    z_mean, z_log_var, z = vae_encoder(merged_features)

    x = Dense(256, activation='relu')(z)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[img_model.input, dem_model.input], outputs=output, name='DualInputModel')
    return model

# Compile the model
model = build_dual_input_model(input_shape=(224, 224, 3))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define the checkpoint filepath
checkpoint_filepath = 'best_model.h5'

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Train the model with the ModelCheckpoint callback
history = model.fit(
    [x_img_train, x_dem_train],
    y_train,
    validation_data=([x_img_val, x_dem_val], y_val),
    batch_size=batch_size,
    epochs=20,
    callbacks=[checkpoint_callback]  # Pass the ModelCheckpoint callback here
)

# Load the best weights
model.load_weights(checkpoint_filepath)

# Predict using the best weights
y_pred_probs = model.predict([x_img_val, x_dem_val])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=['non-landslide', 'landslide']))
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
