import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Configuration
IMG_SIZE = 128
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 10 

#  Local Dataset Load and Preprocess
DATA_DIR = "oxford_pets" 
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "annotations", "trimaps")

print("Scanning local directory for images and masks...")
image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

valid_image_paths = []
valid_mask_paths = []

# Ensure we only use images that have a corresponding mask
for img_path in image_paths:
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(MASKS_DIR, base_name + ".png")
    if os.path.exists(mask_path):
        valid_image_paths.append(img_path)
        valid_mask_paths.append(mask_path)

print(f"Found {len(valid_image_paths)} valid image/mask pairs.")

# Split 80/20 for train/test
split_idx = int(len(valid_image_paths) * 0.8)
train_img_paths = valid_image_paths[:split_idx]
train_msk_paths = valid_mask_paths[:split_idx]
test_img_paths = valid_image_paths[split_idx:]
test_msk_paths = valid_mask_paths[split_idx:]

def process_path(image_path, mask_path):
    # Load and process image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    
    # Load and process mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method='nearest')
    mask -= 1 # Shift Oxford Pets masks from 1,2,3 to 0,1,2
    
    return img, mask

def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask

# Build tf.data pipelines
train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_msk_paths))
train_dataset = train_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_batches = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_img_paths, test_msk_paths))
test_dataset = test_dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
test_batches = test_dataset.batch(BATCH_SIZE)

#  Build U-Net Architecture
def double_conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def build_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = double_conv_block(inputs, 64)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = double_conv_block(p1, 128)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = double_conv_block(p2, 256)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = double_conv_block(p3, 512)
    p4 = layers.MaxPooling2D(2)(c4)

    # Bottleneck
    c5 = double_conv_block(p4, 1024)

    # Decoder with Skip Connections
    u6 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = double_conv_block(u6, 512)

    u7 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = double_conv_block(u7, 256)

    u8 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = double_conv_block(u8, 128)

    u9 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = double_conv_block(u9, 64)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(c9)

    return models.Model(inputs, outputs, name="U-Net")

model = build_unet((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)

#  Implement IoU metric calculation
class SparseMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

#  Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', SparseMeanIoU(num_classes=NUM_CLASSES, name='iou')])

#  Train model
print("Starting training...")
history = model.fit(train_batches, epochs=EPOCHS, validation_data=test_batches)

# Step 8: Visualize predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def display(display_list):
    plt.figure(figsize=(15, 5))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])

print("Visualizing predictions on test data...")
show_predictions(test_batches, 2)

#  Save model in .h5 format
h5_path = "unet_model.h5"
model.save(h5_path)
print(f"Model saved to {h5_path}")

#  Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = "unet_model.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved to {tflite_path}")

#  Test on new images (via TFLite interpreter)
print("Testing TFLite model on a new image...")
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

for image, mask in test_dataset.batch(1).take(1):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    tflite_pred = interpreter.get_tensor(output_details[0]['index'])
    
    print("TFLite Prediction successful. Visualizing...")
    display([image[0], mask[0], create_mask(tflite_pred)])