import itertools
import csv
from glob import glob


base_path = "D:\\Files on Desktop\\engine\\fax\\magistrska naloga\\Ankitas Ears\\all imgs copy\\"
member1 = base_path + "family3\\daughter3"
member2 = base_path + "family3\\father3"
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

numTest = len(glob("test*.csv"))
filename = 'test.csv'

# writing to csv file
with open(filename, 'a',  newline='') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    if numTest == 0:
        csvwriter.writerow(['img_pair', 'ground_truth', 'is_related'])
    for c in combs:
        r =[c[0] + '-' + c[1], str(relation), ' ']
        print(r)
        csvwriter.writerow(r)

