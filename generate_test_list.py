import itertools
import csv
from glob import glob


base_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\all imgs copy\\"
family = "20"
member2 = base_path + "family" + family + "\\mother20"
member1 = base_path + "family" + family + "\\son20"
# 1 means yes 0 means no
relation = 1

print('img_pair,ground_truth,is_related')
list1 = glob(member1 + '\\*.jpg')
list1 = [x.split('\\')[-1] for x in list1]
# print(list1)
list2 = glob(member2 + '\\*.jpg')
list2 = [x.split('\\')[-1] for x in list2]
# print(list2)

combs = list(itertools.product(list1, list2))

# filename = "pairs.txt"
# with open(filename, 'a') as f:
#    f.write(str(len(combs)) + ",")

# for main task
numTest = len(glob("test*.csv"))
filename = 'test.csv'

# for pair counting
# numTest = len(glob("test_fam*.csv"))
# filename = 'test_fam.csv'

# writing to csv file
with open(filename, 'a',  newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    if numTest == 0:
        csvwriter.writerow(['img_pair', 'ground_truth', 'is_related'])
    for c in combs:
        r =[c[0] + '-' + c[1], str(relation), '']
        print(r)
        csvwriter.writerow(r)

