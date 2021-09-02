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


image_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\all imgs copy\\"
im1_path = image_path + "family10\\mother10\\IMG20200908183014.jpg"
im2_path = image_path + "family12\\father12\\IMG20200913121120.jpg"
im3_path = image_path + "family15\\son15\\IMG20200913180503.jpg"

im1 = cv2.imread(im1_path)
im2 = cv2.imread(im2_path)
im3 = cv2.imread(im3_path)

im1 = alignment(im1, im1_path)
im2 = alignment(im2, im2_path)
im3 = alignment(im3, im3_path)

im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)

im1 = cv2.resize(im1, (224, 224))
im2 = cv2.resize(im2, (224, 224))
im3 = cv2.resize(im3, (224, 224))

fig1, (ax1_11, ax1_12, ax1_13) = plt.subplots(1, 3)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
ax1_13.imshow(im3)

ax1_11.axis('off')
ax1_12.axis('off')
ax1_13.axis('off')

plt.show()

