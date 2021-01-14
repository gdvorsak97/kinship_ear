import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

filename = 'D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\vgg_face_results.csv'
results = pd.read_csv(filename)

r = list(results['is_related'])
gt = list(results['ground_truth'])

# print(r)
# print(gt)

# first define what is binary
for i in range(len(r)):
    if r[i] >= 0.5:
        r[i] = 1
    else:
        r[i] = 0

cm = confusion_matrix(gt, r)
ca = accuracy_score(gt, r)

print(cm)
print('CA:\t\t\t\t ' + str(ca))

# TNi = 00 , FNi = 10 , TPi = 11 and FPi = 01.
TN = cm[0, 0]
FN = cm[1, 0]
TP = cm[1, 1]
FP = cm[0, 1]

sens = TP / (TP + TN)   # pred relationships as true relationships
spec = TN / (TP + TN)   # pred non-related as non-related

print('Sensitivity:\t ' + str(sens))
print('Specificity:\t ' + str(spec))

