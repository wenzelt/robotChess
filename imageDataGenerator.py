import os
from pathlib import Path

import tensorflow as tf
from keras.layers import RandomRotation, RandomFlip
from keras.layers import Resizing
from keras.layers import Resizing

from keras.layers import Rescaling
from keras.preprocessing.image_dataset import image_dataset_from_directory
from tensorflow import keras
from tensorflow.python import data

from prepare_ds import prepare

batch_size = 32
img_height = 224
img_width = 224

original_dataset_dir = "./images"

base_dir = './base'
Path(base_dir).mkdir(parents=True, exist_ok=True)
# Directories for our training, validation and test splits
train_dir = os.path.join(base_dir, 'train')
Path(train_dir).mkdir(parents=True, exist_ok=True)

validation_dir = os.path.join(base_dir, 'validation')
Path(validation_dir).mkdir(parents=True, exist_ok=True)

test_dir = os.path.join(base_dir, 'test')
Path(test_dir).mkdir(parents=True, exist_ok=True)

train_dataset = image_dataset_from_directory(
    './images',
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

val_dataset = image_dataset_from_directory(
    './images',
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation',
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
)

test_dataset = image_dataset_from_directory(
    './base/test',
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
)

normalization_layer = Rescaling(1. / 255)

AUTOTUNE = data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 7

model = keras.Sequential([
    keras.layers.Rescaling(1. / 255),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

image_batch, label_batch = next(iter(train_dataset))

IMG_SIZE = 224

train_ds = prepare(train_dataset, shuffle=True, augment=True)
val_ds = prepare(val_dataset)
test_ds = prepare(test_dataset)

a = 1
#
# train_datagen = ImageDataGenerator(
#     preprocess_input=preprocess_input,
#     rotation_range=40,
#     shear_range=0.2,
#     zoom_range=0.2,
#     vertical_flip=True,
#     horizontal_flip=True)
#
# # the validation data should not be augmented!
# test_datagen = ImageDataGenerator(preprocess_input=preprocess_input)
# a = 1
