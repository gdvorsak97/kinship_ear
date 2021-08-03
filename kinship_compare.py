from collections import defaultdict
from glob import glob
from random import choice, sample

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from matplotlib import pyplot as plt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Rescaling, RandomTranslation, RandomRotation, RandomZoom
from tqdm import tqdm

val_famillies_list = ["family10", "family4"]


def get_train_val(familly_name):
    train_file_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train_list.csv"
    train_folders_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train/"
    val_famillies = familly_name

    all_images = glob(train_folders_path + '/*/*/*.jpg')
    all_images = [x.replace("\\", "/") for x in all_images]
    all_files = [str(i).split("/")[-1][:-4] for i in all_images]

    # Filter step fOr bounding boxes
    delete_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\bounding boxes alligment" \
                  "\\delete list.txt"
    delete_file = pd.read_csv(delete_path, delimiter=";")
    FILTERS = "maj_oob,mnr_oob,blr,ilu,drk,grn,lbl"
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

    train_images = [x for x in all_images if val_famillies not in x]
    val_images = [x for x in all_images if val_famillies in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
    relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]
    return train, val, train_person_to_images_map, val_person_to_images_map


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


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma)
            * K.log(1. - pt_0 + K.epsilon()))

    return focal_loss_fixed


def signed_sqrt(x):
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)


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
        img = img[:, int(np.round(224 / 6)):int(np.round(224 / 6)) + int(np.round(224 / 3))]
    elif region == "top":
        img = img[0:int(np.round(224 / 3)), :]
    elif region == "mid_horizontal":
        img = img[int(np.round(224 / 6)):int(np.round(224 / 6)) + int(np.round(224 / 3)), :]
    elif region == "bottom":
        img = img[-int(np.round(224 / 3)):, :]
    return img


def read_img(path):
    in_img = cv2.imread(path)
    in_img = alignment(in_img, path)
    in_img = cv2.resize(in_img, (224, 224))
    # img = crop_ears(in_img, "top")
    img = cv2.resize(in_img, (224, 224))
    # visualize_crop(in_img, img)
    img = np.array(img, dtype="float64")
    img = preprocess_input(img, version=2)  # 1 for VGG, 2 otherwise
    img = rescale(img)
    return img


def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        labels = np.array(labels)

        yield [X1, X2], labels


layers_to_freeze_f2b = 350


def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', weights=None, include_top=False)

    # 518 total layers
    # for x in base_model.layers[layers_to_freeze_f2b:]:  # Freeze layers here - experiment with the num
    #   x.trainable = False
    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = RandomTranslation(width_factor=0.10, height_factor=0.10, fill_mode='nearest')(input_1)
    x2 = RandomTranslation(width_factor=0.10, height_factor=0.10, fill_mode='nearest')(input_2)

    x1 = RandomRotation(factor=0.125, fill_mode='nearest')(x1)
    x2 = RandomRotation(factor=0.125, fill_mode='nearest')(x2)

    x1 = RandomZoom(width_factor=0.1, height_factor=0.1, fill_mode='nearest')(x1)
    x2 = RandomZoom(width_factor=0.1, height_factor=0.1, fill_mode='nearest')(x2)

    x1 = base_model(x1)
    x2 = base_model(x2)

    x1 = GlobalMaxPool2D()(x1)
    x2 = GlobalAvgPool2D()(x2)

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])

    x5 = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x3, x4, x5])
    #  x = Dense(512, activation="relu")(x)
    #  x = Dropout(0.03)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.02)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))
    # model.compile(loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc'], optimizer=Adam(0.00003))
    # model.compile(loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc'], optimizer=Adam(0.00001))
    model.summary()

    return model


model = baseline_model()
for i in tqdm(range(len(val_famillies_list))):
    train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_famillies_list[i])
    file_path = "compare_kin_{}.h5".format(i)
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.3, patience=30, verbose=1)
    # reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.2, patience=20, verbose=1)
    callbacks_list = [checkpoint, reduce_on_plateau]

    history = model.fit(gen(train, train_person_to_images_map, batch_size=16),
                        use_multiprocessing=False,
                        validation_data=gen(val, val_person_to_images_map, batch_size=16),
                        epochs=66, verbose=2,
                        workers=1, callbacks=callbacks_list,
                        steps_per_epoch=300, validation_steps=200)

test_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\test\\"

submission = pd.read_csv('D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\test.csv')


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


preds_for_sub = np.zeros(submission.shape[0])

for i in tqdm(range(len(val_famillies_list))):
    file_path = f"./compare_kin_{i}.h5"
    model.load_weights(file_path)
    # Get the predictions
    predictions = []

    for batch in tqdm(chunker(submission.image_pair.values)):
        X1 = [x.split("g-")[0] + 'g' for x in batch]
        X1 = np.array([read_img(test_path + x) for x in X1])

        X2 = [x.split("g-")[1] for x in batch]
        X2 = np.array([read_img(test_path + x) for x in X2])

        pred = model.predict([X1, X2]).ravel().tolist()
        predictions += pred

    preds_for_sub += np.array(predictions) / len(val_famillies_list)  # average

submission['is_related'] = preds_for_sub
submission.to_csv("compare_results.csv", index=False)
