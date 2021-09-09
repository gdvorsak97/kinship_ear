import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt


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


image_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\all imgs copy\\"
im_path = image_path + "family10\\mother10\\IMG20200908183014.jpg"

im = cv2.imread(im_path)
im = alignment(im, im_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (224, 224))

im1 = crop_ears(im, "top")
im2 = crop_ears(im, "mid_horizontal")
im3 = crop_ears(im, "bottom")
im4 = crop_ears(im, "left")
im5 = crop_ears(im, "mid_vertical")
im6 = crop_ears(im, "right")

im1 = cv2.resize(im1, (224, 224))
im2 = cv2.resize(im2, (224, 224))
im3 = cv2.resize(im3, (224, 224))
im4 = cv2.resize(im4, (224, 224))
im5 = cv2.resize(im5, (224, 224))
im6 = cv2.resize(im6, (224, 224))

fig1, ((ax1_11, ax1_12, ax1_13), (ax1_21, ax1_22, ax1_23)) = plt.subplots(2, 3)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
ax1_13.imshow(im3)
ax1_21.imshow(im4)
ax1_22.imshow(im5)
ax1_23.imshow(im6)

ax1_11.axis('off')
ax1_12.axis('off')
ax1_13.axis('off')
ax1_21.axis('off')
ax1_22.axis('off')
ax1_23.axis('off')

ax1_11.text(0.5, -0.2, "zgoraj", transform=ax1_11.transAxes, ha="center")
ax1_12.text(0.5, -0.2, "sredina horizontalno", transform=ax1_12.transAxes, ha="center")
ax1_13.text(0.5, -0.2, "spodaj", transform=ax1_13.transAxes, ha="center")
ax1_21.text(0.5, -0.2, "levo", transform=ax1_21.transAxes, ha="center")
ax1_22.text(0.5, -0.2, "sredina vertikalno", transform=ax1_22.transAxes, ha="center")
ax1_23.text(0.5, -0.2, "desno", transform=ax1_23.transAxes, ha="center")
plt.show()
