from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.resnet import ResNet152
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

"""
TAKEN FROM augment: https://medium.com/mlait/image-data-augmentation-image-processing-in-tensorflow-part-2-b77237256df0
        apply to model: https://medium.com/analytics-vidhya/how-to-do-image-classification-on-custom-dataset-using-tensorflow-52309666498e
        resnet training: https://medium.com/swlh/resnet-with-tensorflow-transfer-learning-13ff0773cf0c
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
# include non-imposters from "test" into training
# print(len(train_files))
for i in test_files:
    train_files.append(i)
# print(len(train_files))
"""

img_height = 224
img_width = 224


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


image_generator = ImageDataGenerator(rescale=1. / 255, width_shift_range=.15, rotation_range=45, height_shift_range=.15,
                                     horizontal_flip=True, zoom_range=0.2)

valid_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=.20)

valid_generator = valid_datagen.flow_from_directory(train_dir, subset="validation", shuffle=True,
                                                    target_size=(img_height, img_width))

train_generator = image_generator.flow_from_directory(batch_size=8, directory=train_dir, subset="training",
                                                      shuffle=True, target_size=(img_height, img_width))

# print(train_generator.class_indices)

labels = '\n'.join(sorted(train_generator.class_indices.keys()))
with open('labels.txt', 'w') as f:
    f.write(labels)

# Add all this to model
base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# for layer in base_model.layers:
#    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(1000, activation='relu')(x)
predictions = layers.Dense(train_generator.num_classes, activation='softmax')(x)

head_model = Model(inputs=base_model.input, outputs=predictions)
head_model.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])
head_model.summary()

steps_per_epoch = np.ceil(train_generator.samples / train_generator.batch_size)
val_steps_per_epoch = np.ceil(valid_generator.samples / valid_generator.batch_size)

file_path = "weights.h5"

reduce_on_plateau = ReduceLROnPlateau(factor=0.1, patience=10)
checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True)

callbacks_list = [checkpoint,reduce_on_plateau]

head_model.load_weights("weights_at_finish.h5")
history = head_model.fit(train_generator, batch_size=8, epochs=100, steps_per_epoch=steps_per_epoch,
                         validation_data=valid_generator,
                         validation_steps=val_steps_per_epoch, callbacks=callbacks_list)
head_model.save_weights("weights_at_finish.h5")

fig, axs = plt.subplots(2, 1, figsize=(15, 15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train', 'Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train', 'Val'])

fig.show()

# USE model.save! to get the augmentation steps in place and load it into the next step.
head_model.save("model.h5")
