import cv2
import pandas as pd


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


# path = "C:\\Users\\Grega\\Desktop\\pairs\\"
# pair_type = "true positive"
# file = "IMG-20200825-WA0063"
# load_path = path + pair_type + "\\1\\" + file + ".jpg"

path = "C:\\Users\\Grega\\Desktop\\examples\\"
image = "20200915_115411"
load_path = path + image + ".jpg"

in_img = cv2.imread(load_path)
in_img = alignment(in_img, load_path)
in_img = cv2.resize(in_img, (224, 224))

cv2.imshow("img", in_img)
cv2.waitKey()
# save_path = path + pair_type + "\\1\\resized\\"
save_path = path + "\\out\\" + image + ".jpg"
print("save to " + save_path)
# cv2.imwrite(save_path + file + ".jpg", in_img)
cv2.imwrite(save_path,in_img)
