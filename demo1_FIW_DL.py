from collections import defaultdict
from glob import glob
import pandas as pd
import cv2
import numpy as np
from random import choice, sample
from tqdm import tqdm

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

train_file_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/train_list.csv"
train_folders_path = "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/"
# this F09 is a family folder one of many.. check why only one
val_famillies = "F09"  # validation images, list families to separate train, test

all_images = glob(train_folders_path + "D:/Files on Desktop/engine/fax/magistrska naloga/Ankitas Ears/*/*/*.jpg")
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
# validate for people - check after if all accounted for
relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

# get train and val relationship list
train = [x for x in relationships if val_famillies not in x[0]]
val = [x for x in relationships if val_famillies in x[0]]


# read images
def read_img(path):
    img = cv2.imread(path)
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)


# generator of labels for pairs
def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2) # random tuples
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

        yield [X1, X2], labels


# Straightforward, generate model as described in the post
def baseline_model():
    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    # x1_ = Reshape(target_shape=(7*7, 2048))(x1)
    # x2_ = Reshape(target_shape=(7*7, 2048))(x2)
    #
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
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.00001))
    model.summary()

    return model


file_path = "vgg_face.h5"

# callback to save weights
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# reduce rate when learning stagnates
reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

callbacks_list = [checkpoint, reduce_on_plateau]

# can load weights below
model = baseline_model()
# model.load_weights(file_path)
model.fit_generator(gen(train, train_person_to_images_map, batch_size=16), use_multiprocessing=True,
                    validation_data=gen(val, val_person_to_images_map, batch_size=16), epochs=100, verbose=2,
                    workers=4, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)

test_path = "../input/test/"


def chunker(seq, size=32):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


# have to recreate this file
submission = pd.read_csv('../input/sample_submission.csv')

predictions = []

# get predictions
for batch in tqdm(chunker(submission.img_pair.values)):
    X1 = [x.split("-")[0] for x in batch]
    X1 = np.array([read_img(test_path + x) for x in X1])

    X2 = [x.split("-")[1] for x in batch]
    X2 = np.array([read_img(test_path + x) for x in X2])

    pred = model.predict([X1, X2]).ravel().tolist()
    predictions += pred

# output
submission['is_related'] = predictions

submission.to_csv("vgg_face.csv", index=False)
