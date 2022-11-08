# In[1]
'CSI data split'
import numpy as np
from os import listdir
from os.path import isfile, join
#In[1]
'Load data from BALANCED DATASET'
path_balanced_train = 'competition data/balanced data/train_data/'
path_balanced_val = 'competition data/balanced data/validation_data/'

# In[2]
balan_train_data_list = [f for f in listdir(path_balanced_train) if isfile(join(path_balanced_train, f))]
balan_val_data_list = [f for f in listdir(path_balanced_val) if isfile(join(path_balanced_val, f))]

'NOTE! VALIDATION is used for test only, because I know that my models are not overfitted/underfitted'

'get range of SNRs'
'use for loop not to waste memory'
train_snr = []
val_snr = []
for data in balan_train_data_list:
    train_snr.append(np.linalg.norm(np.load(path_balanced_train+data)))

for data in balan_val_data_list:
    val_snr.append(np.linalg.norm(np.load(path_balanced_val+data)))

print('TRAIN SNR:',np.min(train_snr),'-', np.max(train_snr))
print('VAL SNR:',np.min(val_snr),'-', np.max(val_snr))

# In[3]
'create your own unbalanced classes'
n_classes = 6

' In my case '
' CLASS 1 NORM in range 38 to 40 '
' CLASS 2 NORM in range 43 to 45 '
' CLASS 3 NORM in range 48 to 50 '
' CLASS 4 NORM in range 53 to 55 '
' CLASS 5 NORM in range 58 to 60 '
' CLASS 6 NORM in range 63 to 65 '

start = 38.0
ranges = []
for i in range(n_classes):
    ranges.append(np.linspace(start, start+2.0, 2))
    start += 5.0
ranges = np.array(ranges)

train_data_size = 1000 # for each class
val_data_size = 270 # for each class

# In[6]
'load data for each Class 1~6'
'This data will be used in order to create local datasets with different heter. level'

class_train_data = [[] for i in range(n_classes)]
for data in balan_train_data_list:
    x = np.load(path_balanced_train+data)
    norm = np.floor(np.linalg.norm(x))
    if norm in ranges:
        if len(class_train_data[np.where(ranges == norm)[0][0]]) < train_data_size:
            class_train_data[np.where(ranges == norm)[0][0]].append(x)

class_val_data = [[] for i in range(n_classes)]
for data in balan_val_data_list:
    x = np.load(path_balanced_val+data)
    norm = np.floor(np.linalg.norm(x))
    if norm in ranges:
        if len(class_val_data[np.where(ranges == norm)[0][0]]) < val_data_size:
            class_val_data[np.where(ranges == norm)[0][0]].append(x)

# In[5]
'load unbalanced data'
heter = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for het in heter:
    print(het)
    for i in range(n_classes):
        'define path to files'
        path_load_train = 'FED_HETERO_CSI/het'+str(het)+'%/CLASS_'+str(i+1)+'/train_data/'
        path_load_val = 'FED_HETERO_CSI/het'+str(het)+'%/CLASS_'+str(i+1)+'/val_data/'
        
        'save train data'
        biased_data_size = int(train_data_size * het / 100)
        biased_data = class_train_data[i][0:biased_data_size]
        
        rest_data_size = int((train_data_size - biased_data_size)/5)
        rest_data = []
        for j in range(n_classes):
            if i != j:
                rest_data += class_train_data[j][-rest_data_size:]
        
        total_data = biased_data + rest_data
        
        'check if data is distributed properly'
        'uncomment lines 96-110'
        # check = []
        # for data in biased_data:
        #     check.append(np.linalg.norm(data))
        # print(np.min(check), np.max(check))
        # print()
        # check = []
        # for data in rest_data:
        #     check.append(np.linalg.norm(data))
        # print(np.min(check), np.max(check))
        # print()
        # check = []
        # for data in total_data:
        #     check.append(np.linalg.norm(data))
        # print(np.min(check), np.max(check))
        # print()
        
        for k in range(len(total_data)):
            if k < train_data_size:
                np.save(path_load_train + str(k) + '.npy', total_data[i])

        'save val data'
        biased_data_size = int(val_data_size * het / 100)
        biased_data = class_val_data[i][0:biased_data_size]
        
        rest_data_size = int((val_data_size - biased_data_size)/5)
        rest_data = []
        for j in range(n_classes):
            if i != j:
                rest_data += class_val_data[j][-rest_data_size:]
        
        total_data = biased_data + rest_data
        
        save_val = 250
        for k in range(len(total_data)):
            if k < save_val:
                np.save(path_load_val + str(k) + '.npy', total_data[i])
