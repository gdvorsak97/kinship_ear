from collections import defaultdict
from glob import glob

import keras
import pandas as pd
import cv2
import numpy as np
from random import choice, sample
from tqdm import tqdm
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, \
    Subtract
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 10)

train_file_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train_list.csv"
train_folders_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train/"
val_famillies = "family10"  # validation images, list families to separate train, test

all_images = glob(train_folders_path + '/*/*/*.jpg')
all_images = [x.replace("\\", "/") for x in all_images]
train_images = [x for x in all_images if val_famillies not in x]  # all except families in val_families
val_images = [x for x in all_images if val_famillies in x]  # all images not in validation set

# dictionary for a map of person:image
train_person_to_images_map = defaultdict(list)
ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]  # family/person
for x in train_images:
    # append train image to the person in the above format
    train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

# similar but validation
val_person_to_images_map = defaultdict(list)
for x in val_images:
    # same as above but val images
    val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

# get pairs from csv to a zipped list
relationships = pd.read_csv(train_file_path)
relationships = list(zip(relationships.p1.values, relationships.p2.values))
# validate for people
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

# get train and val relationship list
train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_loss(history, label, n):
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0, 1])  # originally from 0.8 to 1
        else:
            plt.ylim([0, 1])

        plt.legend()
    plt.show()


# read images
def read_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img, dtype="float64")
    return preprocess_input(img, version=1)  # 1 for VGG, 2 otherwise


# generator of labels for pairs
def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)  # random tuples
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            # choose 2 people
            p1 = choice(ppl)
            p2 = choice(ppl)

            # if not in random tuples, add it there and append a zero label
            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        # A check if all exist
        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        # images loaded from random choices, labels added
        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        labels = np.array(labels)

        yield [X1, X2], labels


# There are 17 layers with trainable parameters
layers_to_freeze_b2f = -6


# Straightforward, generate model as described in the post
def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = load_model("model_resnet_rec_ears.h5")
    base_model.load_weights("weights_recognition_resnet_val96_finish.h5")

    # for x in base_model.layers[:layers_to_freeze_b2f]:  # Freeze layers here - experiment with the num
    #    x.trainable = False

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    # x1_ = Reshape(target_shape=(7*7, 2048))(x1)
    # x2_ = Reshape(target_shape=(7*7, 2048))(x2)
    # x_dot = Dot(axes=[2, 2], normalize=True)([x1_, x2_])
    # x_dot = Flatten()(x_dot)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])
    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])
    x = Multiply()([x1, x2])
    x = Concatenate(axis=-1)([x, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.01)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)
    model.compile(loss="binary_crossentropy", metrics=METRICS, optimizer=Adam(0.00001))
    model.summary()

    return model


n_epochs = 25
n_steps_per_epoch = 30
n_val_steps = 10
key = "031109_" + str(n_epochs) + "_" + str(n_steps_per_epoch) + "_" + str(n_val_steps)
file_path = "D:/Files on Desktop/engine/fax/magistrska naloga/vgg_face_" + key + ".h5"

# callback to save weights
checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

# reduce rate when learning stagnates
reduce_on_plateau = ReduceLROnPlateau(factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]
# callbacks_list = [reduce_on_plateau]

model = baseline_model()

# model.load_weights(file_path)
baseline_history = model.fit(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=False,
                             validation_data=gen(val, val_person_to_images_map, batch_size=5), epochs=n_epochs,
                             verbose=2,
                             workers=1, callbacks=callbacks_list, steps_per_epoch=n_steps_per_epoch,
                             validation_steps=n_val_steps)

# plot_loss(baseline_history, "Baseline", 0)
plot_metrics(baseline_history)

test_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\test\\"


def chunker(seq, size=64):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# file which is later used for the confusion matrix
results = pd.read_csv('D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\test.csv')

predictions = []

# get predictions
for batch in tqdm(chunker(results.img_pair.values)):
    X1 = [x.split("g-")[0] + 'g' for x in batch]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = [x.split("g-")[1] for x in batch]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = model.predict([X1, X2]).ravel().tolist()
    predictions += pred

# output
results['is_related'] = predictions

results.to_csv("vgg_face_results" + key + ".csv", index=False)
