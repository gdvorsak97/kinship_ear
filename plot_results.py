import cv2
from matplotlib import pyplot as plt

path = "C:\\Users\\Grega\\Desktop\\graphs\\"
file1 = path + "resnet_main\\example_roc_52_loss.png"
file2 = path + "resnet_main\\example_roc_52_test.png"

im1 = cv2.imread(file1)
im2 = cv2.imread(file2)
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)

fig1, (ax1_11, ax1_12) = plt.subplots(1, 2)

ax1_11.axis('off')
ax1_12.axis('off')

ax1_11.imshow(im1)
ax1_12.imshow(im2)

plt.show()

# USE MERGING ONLINE, IT'S BETTER
