import cv2
from matplotlib import pyplot as plt

image_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\all imgs copy\\"
im1 = image_path + "family3\\son3\\IMG-20200826-WA0019.jpg"  # blr
im2 = image_path + "family12\\mother12\\IMG-20200914-WA0010.jpg"  # drk
im3 = image_path + "family14\\daughter14\\IMG20200913124014.jpg"  # grn
im4 = image_path + "family10\\son10\\IMG20200909084124.jpg"  # ilu
im5 = image_path + "family20\\son20\\IMG-20200903-WA0027.jpg"  # lbl
im6 = image_path + "family17\\mother17\\IMG_20200822_091612.jpg"  # maj_oob
im7 = image_path + "family1\\Father1\\IMG20200719144440.jpg"  # mnr_oob
im8 = image_path + "family12\\son12\\IMG-20200914-WA0014.jpg"  # combination - mnr_oob + drk (+ blr)

im1 = cv2.imread(im1)
im2 = cv2.imread(im2)
im3 = cv2.imread(im3)
im4 = cv2.imread(im4)
im5 = cv2.imread(im5)
im6 = cv2.imread(im6)
im7 = cv2.imread(im7)
im8 = cv2.imread(im8)

im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)
im5 = cv2.cvtColor(im5, cv2.COLOR_BGR2RGB)
im6 = cv2.cvtColor(im6, cv2.COLOR_BGR2RGB)
im7 = cv2.cvtColor(im7, cv2.COLOR_BGR2RGB)
im8 = cv2.cvtColor(im8, cv2.COLOR_BGR2RGB)

im1 = cv2.resize(im1, (224, 224))
im2 = cv2.resize(im2, (224, 224))
im3 = cv2.resize(im3, (224, 224))
im4 = cv2.resize(im4, (224, 224))
im5 = cv2.resize(im5, (224, 224))
im6 = cv2.resize(im6, (224, 224))
im7 = cv2.resize(im7, (224, 224))
im8 = cv2.resize(im8, (224, 224))

fig1, ((ax1_11, ax1_12, ax1_13, ax1_14), (ax1_21, ax1_22, ax1_23, ax1_24)) = plt.subplots(2, 4)
ax1_11.imshow(im1)
ax1_12.imshow(im2)
ax1_13.imshow(im3)
ax1_14.imshow(im4)
ax1_21.imshow(im5)
ax1_22.imshow(im6)
ax1_23.imshow(im7)
ax1_24.imshow(im8)

ax1_11.axis('off')
ax1_12.axis('off')
ax1_13.axis('off')
ax1_14.axis('off')
ax1_21.axis('off')
ax1_22.axis('off')
ax1_23.axis('off')
ax1_24.axis('off')

ax1_11.text(0.5, -0.2, "zamegljena slika", transform=ax1_11.transAxes, ha="center")
ax1_12.text(0.5, -0.2, "temna slika", transform=ax1_12.transAxes, ha="center")
ax1_13.text(0.5, -0.2, "zelena slika", transform=ax1_13.transAxes, ha="center")
ax1_14.text(0.5, -0.2, "močna osvetlitev", transform=ax1_14.transAxes, ha="center")
ax1_21.text(0.5, -0.35, "logotip prekriva\nuhelj", transform=ax1_21.transAxes, ha="center")
ax1_22.text(0.5, -0.35, "večji del\nuhlja manjka", transform=ax1_22.transAxes, ha="center")
ax1_23.text(0.5, -0.35, "manjši del\nuhlja manjka", transform=ax1_23.transAxes, ha="center")
ax1_24.text(0.5, -0.50, "kombinacija temne\nin manjka manjšega\ndela uhlja ", transform=ax1_24.transAxes, ha="center")

plt.show()


