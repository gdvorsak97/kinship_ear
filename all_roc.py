import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('Lažno pozitivna stopnja')
    plt.ylabel('Resnično pozitivna stopnja')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.figure()
path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\for paper - updated 3011\\result_files\\"

filename_aff = path + "aff results\\aff_best.csv"
filename_cotnet = path + "cotnet\\transformers_best.csv"
filename_fiw = path + "fiw20 compare\\compare_results.csv"
filename_resnet = path + "resnet_from_scratch\\global\\predictions_resnet_from_scratch_best.csv"
filename_res_local = path + "resnet_from_scratch\\local\\local_top_615.csv"
filename_vgg = path + "vgg_face_model\\vgg_face_results_best_roc.csv"

results_aff = pd.read_csv(filename_aff)
results_cotnet = pd.read_csv(filename_cotnet)
results_fiw = pd.read_csv(filename_fiw)
results_resnet = pd.read_csv(filename_resnet)
results_res_local = pd.read_csv(filename_res_local)
results_vgg = pd.read_csv(filename_vgg)

pred_aff = list(results_aff['is_related'])
pred_cotnet = list(results_cotnet['is_related'])
pred_fiw = list(results_fiw['is_related'])
pred_resnet = list(results_resnet['is_related'])
pred_res_local = list(results_res_local['is_related'])
pred_vgg = list(results_vgg['is_related'])

truth_aff = list(results_aff['ground_truth'])
truth_cotnet = list(results_cotnet['ground_truth'])
truth_fiw = list(results_fiw['ground_truth'])
truth_resnet = list(results_resnet['ground_truth'])
truth_res_local = list(results_res_local['ground_truth'])
truth_vgg = list(results_vgg['ground_truth'])


plot_roc("VGG Face", truth_vgg, pred_vgg, color=colors[0])
plot_roc("CoTNet", truth_cotnet, pred_cotnet, color=colors[1])
plot_roc("AFF", truth_aff, pred_aff, color=colors[2])
plot_roc("ResNet152 globalno", truth_resnet, pred_resnet, color=colors[3])
plot_roc("ResNet152 lokalno", truth_res_local, pred_res_local, color=colors[4])
plot_roc("USTC-NELSLIP", truth_fiw, pred_fiw, color=colors[6])

plt.legend(loc='lower right')
destination = "C:\\Users\\Grega\\Desktop\\output\\"
filename = destination + "all_roc_vect.pdf"
plt.savefig(filename)
plt.show()
