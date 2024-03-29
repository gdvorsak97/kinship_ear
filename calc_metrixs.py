import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, thresholds = roc_curve(labels, predictions)

    plt.plot(fp, tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

    eer = brentq(lambda x: 1. - x - interp1d(fp, tp)(x), 0., 1.)
    thresh = interp1d(fp, thresholds)(eer)
    print("EER:\t" + str(eer))
    print("Thresh\t" + str(thresh))
    return thresh


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
filename = "resnet_kinship_results.csv"
results = pd.read_csv(filename)

pred = list(results['is_related'])
truth = list(results['ground_truth'])

threshold = plot_roc("Test Baseline", truth, pred, color=colors[0])

c_pred = np.zeros(len(pred))

# first define what is binary
for i in range(len(pred)):
    if pred[i] >= threshold:
        c_pred[i] = 1
    else:
        c_pred[i] = 0

cm = confusion_matrix(truth, c_pred)
ca = accuracy_score(truth, c_pred)
auc = roc_auc_score(truth, pred)

print(cm)
print('CA:\t\t\t\t ' + str(ca))

# TNi = 00 , FNi = 10 , TPi = 11 and FPi = 01.
TN = cm[0, 0]
FN = cm[1, 0]
TP = cm[1, 1]
FP = cm[0, 1]

sens = TP / (TP + FN)  # pred relationships as true relationships
spec = TN / (TN + FP)  # pred non-related as non-related

print('Sensitivity:\t ' + str(sens))
print('Specificity:\t ' + str(spec))
print('ROC-AUC:\t\t ' + str(auc))


plt.show()
