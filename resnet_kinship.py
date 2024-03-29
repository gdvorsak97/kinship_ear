import datetime
import pickle
from collections import defaultdict
from glob import glob
from random import choice, sample

import cv2
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_vggface.utils import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import RandomTranslation, RandomRotation, RandomZoom, Rescaling, Flatten
from tensorflow.python.ops.summary_ops_v2 import SummaryWriter
from tqdm import tqdm

mpl.rcParams['figure.figsize'] = (12, 10)
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(histogram_freq=1, log_dir=log_dir)

train_file_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train_list.csv"
train_folders_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train/"
val_famillies = ["family10", "family4"]  # validation images, list families to separate train, test

all_images = glob(train_folders_path + '/*/*/*.jpg')
all_images = [x.replace("\\", "/") for x in all_images]
all_files = [str(i).split("/")[-1][:-4] for i in all_images]

# Filter step fOr bounding boxes
delete_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\bounding boxes alligment" \
              "\\delete list.txt"
delete_file = pd.read_csv(delete_path, delimiter=";")
FILTERS = "major_out_of_bounds,minor_out_of_bounds,blurry,illuminated,dark,green,label"
filters = FILTERS.replace(" ", "")
filters = filters.split(",")
deleted = []
for i in all_files:
    check = delete_file["filename"] == i
    check = list(np.where(check)[0])
    if len(check) != 0:
        d_fs = delete_file["filter"].iloc[check]
        current_f = list(d_fs.values)[0].split(',')
        for f in current_f:
            if f in filters:
                to_delete = all_files.index(i)
                deleted.append(to_delete)
                break

# delete filtered from all images
all_images = [all_images[i] for i in range(len(all_images)) if i not in deleted]

train_images = []
val_images = []
for x in all_images:
    for i in range(len(val_famillies)):
        if val_famillies[i] not in x:
            train_images.append(x)
        elif val_famillies[i] in x:
            val_images.append(x)

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
train = []
val = []

for i in range(len(val_famillies)):
    for x in relationships:
        if val_famillies[i] not in x[0]:
            train.append(x)
        elif val_famillies[i] in x[0]:
            val.append(x)

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


rescale = Sequential([
    Rescaling(1. / 255)
])


def visualize_crop(in_img, crp_img):
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    crp_img = cv2.cvtColor(crp_img, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(in_img[:, :, :])
    axes[0].set_title('Original image')
    axes[1].imshow(crp_img[:, :, :])
    axes[1].set_title('Cropped input')
    fig.suptitle(f'Original and cropped input')
    fig.set_size_inches(9, 5, forward=True)
    plt.show()


def alignment(image, path, visualize=False, save=False):
    label_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\bounding boxes alligment\\"
    delete_ist_path = label_path + "delete list.txt"
    filename = ""
    if "/" in path:
        family = path.split("/")[-3]
        filename = path.split("/")[-1]
        label_path += "labels " + str(family) + ".csv"
    elif "\\" in path:
        filename = path.split("\\")[-1]
        label_path = label_path + "all_labels.csv"
    label_file = pd.read_csv(label_path)
    bbox_data = label_file[label_file['file'] == filename]
    bbox = image[bbox_data['y1'].values[0]:bbox_data['y1'].values[0] + bbox_data['dy'].values[0],
           bbox_data['x1'].values[0]:bbox_data['x1'].values[0] + bbox_data['dx'].values[0]]
    if visualize:
        cv2.imshow("Detected", bbox)
        cv2.waitKey()
    if save:
        cv2.imwrite("example.png", bbox)
    return bbox


def crop_ears(img, region):
    if region == "left":
        img = img[:, 0:int(np.round(224 / 3))]
    elif region == "right":
        img = img[:, -int(np.round(224 / 3)):]
    elif region == "mid_vertical":
        img = img[:, int(np.round(224 / 3)):int(np.round(224 / 3)) + int(np.round(224 / 3))]
    elif region == "top":
        img = img[0:int(np.round(224 / 3)), :]
    elif region == "mid_horizontal":
        img = img[int(np.round(224 / 3)):int(np.round(224 / 3)) + int(np.round(224 / 3)), :]
    elif region == "bottom":
        img = img[-int(np.round(224 / 3)):, :]
    return img


# read images
def read_img(path):
    in_img = cv2.imread(path)
    in_img = alignment(in_img, path)
    # in_img = cv2.resize(in_img, (224, 224))
    # in_img = crop_ears(in_img, "right")
    img = cv2.resize(in_img, (224, 224))
    # visualize_crop(in_img, img)
    img = np.array(img, dtype="float64")
    img = preprocess_input(img, version=2)  # 1 for VGG, 2 otherwise
    img = rescale(img)
    return img


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


# layers_to_freeze_f2b = 80
# layers_to_freeze_b2f = -80


# Straightforward, generate model as described in the post
def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = ResNet152(weights='imagenet', include_top=False)
    # base_model = load_model("model_resnet_rec_ears.h5")
    # base_model.load_weights("weights_recognition_resnet_val96_finish.h5")

    #  518 total layers
    # for x in base_model.layers[layers_to_freeze_f2b:]:
    # for x in base_model.layers[:layers_to_freeze_b2f]:  # Freeze layers here - experiment with the num
    #    x.trainable = False

    x1 = RandomTranslation(width_factor=0.10, height_factor=0.10, fill_mode='nearest')(input_1)
    x2 = RandomTranslation(width_factor=0.10, height_factor=0.10, fill_mode='nearest')(input_2)

    x1 = RandomRotation(factor=0.125, fill_mode='nearest')(x1)
    x2 = RandomRotation(factor=0.125, fill_mode='nearest')(x2)

    x1 = RandomZoom(width_factor=0.1, height_factor=0.1, fill_mode='nearest')(x1)
    x2 = RandomZoom(width_factor=0.1, height_factor=0.1, fill_mode='nearest')(x2)

    x1 = base_model(x1)
    x2 = base_model(x2)

    # use these when training from scratch
    x1 = Flatten()(x1)
    x2 = Flatten()(x2)

    x = Concatenate(axis=-1)([x1, x2])

    x = Dense(100, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)

    base_model.summary()

    model = Model([input_1, input_2], out)
    model.compile(loss="binary_crossentropy", metrics=METRICS, optimizer=Adam(0.00001))
    model.summary()

    return model


n_epochs = 100
n_steps_per_epoch = 32
n_val_steps = 32
file_path = "weights_resnet_kin_best.h5"

# callback to save weights
checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

# reduce rate when learning stagnates
reduce_on_plateau = ReduceLROnPlateau(factor=0.1, patience=20, verbose=1)

# callbacks_list = [checkpoint, reduce_on_plateau, tensorboard_callback]
callbacks_list = [reduce_on_plateau, tensorboard_callback]

model = baseline_model()

img_gen = gen(train, train_person_to_images_map, batch_size=4)

# model.load_weights(file_path)
baseline_history = model.fit(img_gen, use_multiprocessing=False,
                             validation_data=gen(val, val_person_to_images_map, batch_size=5), epochs=n_epochs,
                             workers=1, callbacks=callbacks_list, steps_per_epoch=n_steps_per_epoch,
                             validation_steps=n_val_steps)

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(baseline_history.history, file_pi)

# history = pickle.load(open('trainHistoryDict', "rb"))
plot_loss(baseline_history, "Baseline", 0)
# plot_metrics(baseline_history)

model.save_weights("weights_resnet_kin_finish.h5")

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

results.to_csv("resnet_kinship_results.csv", index=False)
