#Kinship Analysis Using Ear Images

####NOTE: last Nvidia driver to support tensorflow at the moment was from 2.12.2020

###USAGE
To use this model, first obtain the data at https://unilj-my.sharepoint.com/:u:/g/personal/zemersic_fri1_uni-lj_si/EedkQGEDuMBOqMTJnIg3NLgBK0OYq6f_K1uQgcRr27G06g?e=LFhtjl

Next, change the paths in all 3 .py to files of your preferred locations.
It is important that the test folder contains all family directories used in the testing and validation sets.
The training directory should only contain images that are used in training.

Also included are: the train_list_backup.csv file which can used because a list of pairs of members that are in kin needs to be provided to assign the correct testing labels.
The test_qt_for_3_11_19_copy.csv can be used as the test set list when using families 3, 11 and 19 in the testing set. If used, make sure that the training directory does not contain these three family directories.
A custom test set list can be created using generate_test_list.py, where we add image pairs to a csv file by specifying members of families (directory names), and the label 1 if the members are in kin or 0 otherwise.

To train the model and make predictions, use demo1_FIW_DL.py. Parameters can be specified in the file.

To calculate CA, specificity and sensitivity, use calc_metrics.py.