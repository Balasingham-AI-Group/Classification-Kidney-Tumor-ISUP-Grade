#make the indices for image classification
#the purpose was splitting the dataset in a way that put 70 percent of dead people 
#in training and 30 percent of dead people in test dataset

import numpy as np
import pandas as pd
import random
import os


#get the whole file and detect the number of classes
#__________________________________________________________________________________
data_dir = '/home/mary/Documents/kidney_ds/ISUP_C'
class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
num_class = len(class_names)
image_files = [[os.path.join(data_dir, class_name,'image', x) 
                for x in os.listdir(os.path.join(data_dir, class_name, 'image'))] 
               for class_name in class_names]

tumor_files = [[os.path.join(data_dir, class_name,'label', x) 
                for x in os.listdir(os.path.join(data_dir, class_name, 'label'))] 
               for class_name in class_names]

image_file_list = []
tumor_file_list = []
image_label_list = []

for i, class_name in enumerate(class_names):
    
    image_file_list.extend(sorted(image_files[i]))
    tumor_file_list.extend(sorted(tumor_files[i]))
    image_label_list.extend([i] * len(image_files[i]))    
num_total = len(image_label_list)
print('Total image count:', num_total)
print('Total label count:', len(tumor_file_list))
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])
print("Percent of every class:", [int(((len(image_files[i])/num_total)*100)) for i in range(num_class)])

order_of_cases = []
classes = []
for i in image_file_list:
    order_of_cases.append(int(i[58:63]))
    classes.append(int(i[43:44]))

#Split dataset to train,validation and test
#__________________________________________________________________________________
# We use kits_label_244.csv. this is the clinical dataset from Kits21 just the order 
# of cases are based on the ISUP1,2,3 and 4. The orders are based on the files order
# in class folders.
# We have to split dataset to 2 different parts. Alive and dead
#_________________________________________________________________________________________

label_d = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/kits_label_244.csv')
dead_index = []
censored_index = []
for i in range(0,244):
    if label_d.iloc[i,2] == 1:
        dead_index.append(i)
    else:
        censored_index.append(i)

dead_index = np.array(dead_index)
censored_index = np.array(censored_index)

classes=[]
for i in range(0,244):
    classes.append(label_d.iloc[i,])

# Then make 3 pots from alive(71,71,70). and make 3 pots from dead(11,11,10)
# _______________________________________________________________________________

pot1_alive = random.sample(sorted(censored_index),71)
for i in pot1_alive:
    censored_index = np.delete(censored_index, np.where( censored_index == i))

pot2_alive = random.sample(sorted(censored_index),71)
for i in pot2_alive:
    censored_index = np.delete(censored_index, np.where( censored_index == i))
    
pot3_alive = censored_index

pot1_dead = random.sample(sorted(dead_index),11)
for i in pot1_dead:
    dead_index = np.delete(dead_index, np.where( dead_index == i))

pot2_dead = random.sample(sorted(dead_index),11)
for i in pot2_dead:
    dead_index = np.delete(dead_index, np.where( dead_index == i))
    
pot3_dead = dead_index

# for fold 0 use pot1 and pot2 from alive and dead for train and pot3 for test
# for fold 1 use pot1 and pot3 from alive and dead for train and pot2 for test
# for fold 2 use pot2 and pot3 from alive and dead for train and pot1 for test
# _______________________________________________________________________________

train_index_fold0 = sorted(np.concatenate((pot1_alive, pot2_alive,pot1_dead, pot2_dead)))
train_index_fold1 = sorted(np.concatenate((pot1_alive, pot3_alive,pot1_dead, pot3_dead)))
train_index_fold2 = sorted(np.concatenate((pot2_alive, pot3_alive,pot2_dead, pot3_dead)))

test_index_fold0 = sorted(np.concatenate((pot3_alive, pot3_dead)))
test_index_fold1 = sorted(np.concatenate((pot2_alive, pot2_dead)))
test_index_fold2 = sorted(np.concatenate((pot1_alive, pot1_dead)))

#choose 4 from every class for validation dataset
#_________________________________________________________________________________
list_val_fold0 = [[],[],[],[]]
for i in train_index_fold0:
    if classes[i] == 0:
      list_val_fold0[0].append(i)
    if classes[i] == 1:
      list_val_fold0[1].append(i)
    if classes[i] == 2:
      list_val_fold0[2].append(i)
    if classes[i] == 3:
      list_val_fold0[3].append(i)

val_index_fold0 = sorted(random.sample(list_val_fold0[0],k=4))
x1 = sorted(random.sample(list_val_fold0[1],k=4))
x2 = sorted(random.sample(list_val_fold0[2],k=4))
x3 = sorted(random.sample(list_val_fold0[3],k=4))

val_index_fold0.extend(x1)
val_index_fold0.extend(x2)
val_index_fold0.extend(x3)

#_________________________________________________________________________________
list_val_fold1 = [[],[],[],[]]
for i in train_index_fold1:
    if classes[i] == 0:
      list_val_fold1[0].append(i)
    if classes[i] == 1:
      list_val_fold1[1].append(i)
    if classes[i] == 2:
      list_val_fold1[2].append(i)
    if classes[i] == 3:
      list_val_fold1[3].append(i)

val_index_fold1 = sorted(random.sample(list_val_fold1[0],k=4))
x1 = sorted(random.sample(list_val_fold1[1],k=4))
x2 = sorted(random.sample(list_val_fold1[2],k=4))
x3 = sorted(random.sample(list_val_fold1[3],k=4))

val_index_fold1.extend(x1)
val_index_fold1.extend(x2)
val_index_fold1.extend(x3)

#_________________________________________________________________________________
list_val_fold2 = [[],[],[],[]]
for i in train_index_fold2:
    if classes[i] == 0:
      list_val_fold2[0].append(i)
    if classes[i] == 1:
      list_val_fold2[1].append(i)
    if classes[i] == 2:
      list_val_fold2[2].append(i)
    if classes[i] == 3:
      list_val_fold2[3].append(i)

val_index_fold2 = sorted(random.sample(list_val_fold2[0],k=4))
x1 = sorted(random.sample(list_val_fold2[1],k=4))
x2 = sorted(random.sample(list_val_fold2[2],k=4))
x3 = sorted(random.sample(list_val_fold2[3],k=4))

val_index_fold2.extend(x1)
val_index_fold2.extend(x2)
val_index_fold2.extend(x3)

#new train indexes after dropping validation indexes from them
#_________________________________________________________________________________

index_list = []
for i,item in enumerate(train_index_fold0):
    if item in val_index_fold0:
        index_list.append(i)
new_train_index_fold0 = np.delete(train_index_fold0,index_list)

index_list = []
for i,item in enumerate(train_index_fold1):
    if item in val_index_fold1:
        index_list.append(i)
new_train_index_fold1 = np.delete(train_index_fold1,index_list)

index_list = []
for i,item in enumerate(train_index_fold2):
    if item in val_index_fold2:
        index_list.append(i)
new_train_index_fold2 = np.delete(train_index_fold2,index_list)

#save the train,validation and test indexes
#_________________________________________________________________________________

pp0 = pd.DataFrame(data = new_train_index_fold0, columns = ['train'])
pp1 = pd.DataFrame(data = val_index_fold0, columns = ['validation'])
pp2 = pd.DataFrame(data = test_index_fold0, columns = ['test'])
pp3 = pd.concat([pp0, pp1,pp2],axis=1)
pp3.to_csv('fold0_indexes.csv',index=False)

pp0 = pd.DataFrame(data = new_train_index_fold1, columns = ['train'])
pp1 = pd.DataFrame(data = val_index_fold1, columns = ['validation'])
pp2 = pd.DataFrame(data = test_index_fold1, columns = ['test'])
pp3 = pd.concat([pp0, pp1,pp2],axis=1)
pp3.to_csv('fold1_indexes.csv',index=False)

pp0 = pd.DataFrame(data = new_train_index_fold2, columns = ['train'])
pp1 = pd.DataFrame(data = val_index_fold2, columns = ['validation'])
pp2 = pd.DataFrame(data = test_index_fold2, columns = ['test'])
pp3 = pd.concat([pp0, pp1,pp2],axis=1)
pp3.to_csv('fold2_indexes.csv',index=False)

