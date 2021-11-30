from matplotlib import pyplot as plt


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

vgg_res = 74.1
cotnet_res = 64.0
aff_res = 63.4
resnet_glob_res = 63.1
resnet_loc_res = 61.5
ustc_res = 53.8

# col = [colors[0], colors[1], colors[2], colors[3], colors[4], colors[6]]
names =["VGG16", "CoTNet", "AFF", "ResNet\nglobalno", "ResNet\nlokalno", "USTC-NELSLIP"]
data = [vgg_res, cotnet_res, aff_res, resnet_glob_res, resnet_loc_res, ustc_res]
plt.figure()
plt.bar(names, data)
plt.grid(axis='y')
destination = "C:\\Users\\Grega\\Desktop\\output\\"
filename = destination + "chart.pdf"
plt.savefig(filename)
plt.show()
