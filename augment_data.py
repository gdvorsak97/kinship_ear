import numpy as np
import os

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

list_ds = tf.data.Dataset.list_files(train_files, shuffle=False)
list_ds = list_ds.shuffle(len(train_files), reshuffle_each_iteration=False)

val_size = int(len(train_files) * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

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


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    # return tf.image.resize(img, [img_height, img_width]) # resize if needed
    return img


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(32)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
