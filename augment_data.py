import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers

import tensorflow as tf
from glob import glob

from tensorflow.python.data import AUTOTUNE

"""
TAKEN FROM load: https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control
        augment: https://www.tensorflow.org/tutorials/images/data_augmentation#apply_augmentation_to_a_dataset
"""

data_dir = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\UERC Competition Package 2019\\Dataset\\"
train_dir = data_dir + "Train Dataset\\"
test_dir = data_dir + "Test Dataset\\"

# 0 are impostors
file_list = data_dir + "Info Files\\files.txt"
group_list = data_dir + "Info Files\\groups.txt"

train_files = list(glob(train_dir + '*\\*.png'))

f = open(file_list, "r")
test_files = [line.strip("\n").replace('/', '\\') for line in f]
f.close()
f = open(group_list, "r")
test_labels = [line.strip('\n') for line in f]
f.close()

print("full size: " + str(len(test_files)))
test_files = [test_dir + test_files[i] for i in range(len(test_labels)) if test_labels[i] == '1']
print("size after removal of imposters " + str(len(test_files)))

"""
Test to check that images are shown after resize
img = tf.io.read_file(test_files[0])
img = tf.image.decode_jpeg(img, channels=3)
img = tf.image.resize(img, [180,180])
img = tf.cast(img, tf.uint8)
plt.imshow(img)
plt.show()
"""

# define datasets
list_ds = tf.data.Dataset.list_files(train_files, shuffle=False)
list_ds = list_ds.shuffle(len(train_files), reshuffle_each_iteration=False)
test_list_ds = tf.data.Dataset.list_files(test_files, shuffle=False)
test_list_ds = list_ds.shuffle(len(test_files), reshuffle_each_iteration=False)

# split train and val, make test
val_size = int(len(train_files) * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
test_ds = test_list_ds.take(len(test_files))

# print sizes
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())
print(tf.data.experimental.cardinality(test_ds).numpy())

# define class labels
cn = []
for i in train_files:
    cn.append(i.split("\\")[-2])
class_names = np.array(sorted(cn))


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    one_hot = parts[-2] == class_names
    # Integer encode the label
    return tf.argmax(one_hot)


img_height = 180
img_width = 180


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    img = tf.image.resize(img, [img_height, img_width])  # resize if needed
    img = tf.cast(img, tf.uint8)
    return img


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# create image, label pairs
# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(32)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


image, label = next(iter(train_ds))

"""
# TEST TO SHOW IMAGE - WORKING
plt.imshow(image)
plt.show()
"""

# AUGMENTATION START
resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(img_height, img_width),
    layers.experimental.preprocessing.Rescaling(1. / 255)
])

"""
# test - WORKS BUT SET SIZE AND CHECK ALL
result = resize_and_rescale(image)
plt.imshow(result)
plt.show()
"""

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.3),
    layers.experimental.preprocessing.RandomContrast([0.5, 1.5])
])

# Add the image to a batch
image = tf.expand_dims(image, 0)

"""
# Test
plt.figure(figsize=(10, 10))
for i in range(12):
    augmented_image = data_augmentation(image)
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(augmented_image[0])
    plt.axis("off")
plt.show()
"""

# OTHER CHOICES AFTER TEST WORKS - the latter 2 aren't used as whole ears are needed
# layers.RandomContrast, layers.RandomCrop, layers.RandomZoom


# Add all this to model

model = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # Rest of your model
])


# modify datasets
def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets
    ds = ds.batch(32)

    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

# well shuffled data, batches - DO BEFORE TRAINING
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)

# USE model.save! to get the augmentation steps in place and load it into the next step.
# try to save the datasets as well
print("end")
