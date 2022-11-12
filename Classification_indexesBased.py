
#Image classification based on ISUP grades
#libraries

# In[]
import time
import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np

from numpy import nan
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import KFold
import random

from torch.utils.data import Subset
from monai.apps import download_and_extract
from monai.config import print_config
from monai.metrics import ROCAUCMetric
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    Activations,
    AsChannelFirstd,
    EnsureChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Spacingd,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityRanged,
    Resized,
    Orientationd,
    ToTensord,
    RandAffined,
    RandGaussianNoised
)

from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from efficientnet_pytorch_3d import EfficientNet3D

#total files before augmentation

# In[]:


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


#See the order of patients and classes

# In[]:


order_of_cases = []
classes = []
for i in image_file_list:
    order_of_cases.append(int(i[58:63]))
    classes.append(int(i[43:44]))


#total files after augmentation

# In[]:


data_dir = '/home/mary/Documents/kidney_ds/synthetic_data'
class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
num_class = len(class_names)
image_files = [[os.path.join(data_dir, class_name,'image', x) 
                for x in os.listdir(os.path.join(data_dir, class_name, 'image'))] 
               for class_name in class_names]

tumor_files = [[os.path.join(data_dir, class_name,'label', x) 
                for x in os.listdir(os.path.join(data_dir, class_name, 'label'))] 
               for class_name in class_names]

image_file_list_aug = []
tumor_file_list_aug = []
image_label_list_aug = []

for i, class_name in enumerate(class_names):
    
    image_file_list_aug.extend(sorted(image_files[i]))
    tumor_file_list_aug.extend(sorted(tumor_files[i]))
    image_label_list_aug.extend([i] * len(image_files[i]))

    
num_total_aug = len(image_label_list_aug)


print('Total image count:', num_total_aug)
print('Total label count:', len(tumor_file_list_aug))
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])
print("Percent of every class:", [int(((len(image_files[i])/num_total_aug)*100)) for i in range(num_class)])


#Read the train,validation and test indexes that obtained before for fold 0,1,2

# In[]:

fold0_indexes = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/fold0_indexes.csv')
fold1_indexes = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/fold1_indexes.csv')
fold2_indexes = pd.read_csv ('/home/mary/Documents/kidney_ds/survival/fold2_indexes.csv')

train_index_fold0 = fold0_indexes['train'].tolist()

val_index_fold0 = fold0_indexes['validation'].tolist()
val_index_fold0 = [x for x in val_index_fold0 if np.isnan(x) == False]
val_index_fold0 = [int(x) for x in val_index_fold0]

test_index_fold0 = fold0_indexes['test'].tolist()
test_index_fold0 = [x for x in test_index_fold0 if np.isnan(x) == False]
test_index_fold0 = [int(x) for x in test_index_fold0]

train_index_fold1 = fold1_indexes['train'].tolist()

val_index_fold1 = fold1_indexes['validation'].tolist()
val_index_fold1 = [x for x in val_index_fold1 if np.isnan(x) == False]
val_index_fold1 = [int(x) for x in val_index_fold1]

test_index_fold1 = fold1_indexes['test'].tolist()
test_index_fold1 = [x for x in test_index_fold1 if np.isnan(x) == False]
test_index_fold1 = [int(x) for x in test_index_fold1]

train_index_fold2 = fold2_indexes['train'].tolist()

val_index_fold2 = fold2_indexes['validation'].tolist()
val_index_fold2 = [x for x in val_index_fold2 if np.isnan(x) == False]
val_index_fold2 = [int(x) for x in val_index_fold2]

test_index_fold2 = fold2_indexes['test'].tolist()
test_index_fold2 = [x for x in test_index_fold2 if np.isnan(x) == False]
test_index_fold2 = [int(x) for x in test_index_fold2]


#Fold0
#Choosing cases from total augmented dataset based on the indexes that we obtained before(train,validation,test)

# In[]:
cases_train_fold0 = []
for i in train_index_fold0:
    cases_train_fold0.append(order_of_cases[i])
cases_val_fold0 = []
for j in val_index_fold0:
    cases_val_fold0.append(order_of_cases[j])
cases_test_fold0 = []
for m in test_index_fold0:
    cases_test_fold0.append(order_of_cases[m])


#Getting image file, tumor file and class label for train,validation and test data

# In[]:

trainX, trainY, tumTr = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_train_fold0:
        if int(item[66:71]) == j:
            trainX.append(item)
            trainY.append(int(item[51:52]))
            tumTr.append(tumor_file_list_aug[i])
valX, valY, tumV = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_val_fold0:
        if int(item[66:71]) == j:
            valX.append(item)
            valY.append(int(item[51:52]))
            tumV.append(tumor_file_list_aug[i])
testX, testY, tumTs = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_test_fold0:
        if int(item[66:71]) == j:
            testX.append(item)
            testY.append(int(item[51:52]))
            tumTs.append(tumor_file_list_aug[i])

print("Training count =",len(trainX),"Validation count =", len(valX), "Test count =",len(testX))

#Monai transformers for image preprocessing

# In[]:

train_transforms = Compose([
    LoadImaged(keys=['image']),
    AddChanneld(keys=['image']),
    Spacingd(keys=['image'], pixdim=(1.5, 1.5, 2)),
    Orientationd(keys=['image'], axcodes="RAS"),
    ScaleIntensityRanged(keys='image', a_min=-200, a_max=500, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=['image'], spatial_size=[128, 128, 128]),
    ToTensord(keys=['image'])
])
     
val_transforms = Compose([
    LoadImaged(keys=['image']),
    AddChanneld(keys=['image']),
    Orientationd(keys=['image'],axcodes="RAS"),
    ScaleIntensityRanged(keys='image', a_min=-200, a_max=500, b_min=0.0, b_max=1.0, clip=True),
    Resized(keys=['image'],spatial_size=[128, 128, 128]),
    ToTensord(keys=['image'])
])

lb_transforms = Compose([
    LoadImaged(keys=['label']),
    AddChanneld(keys=['label']),
    Orientationd(keys=['label'],axcodes="RAS"),
    Resized(keys=['label'],spatial_size=[128, 128, 128]),
    ToTensord(keys=['label'])
])

#Making main image datasets and tumor image datasets seperately

# In[]:

train_files = [{"image": image_name} for image_name in trainX]
val_files = [{"image": image_name} for image_name in valX]
test_files = [{"image": image_name} for image_name in testX]

train_files_l = [{"label": label_name} for label_name in tumTr]
val_files_l = [{"label": label_name} for label_name in tumV]
test_files_l = [{"label": label_name} for label_name in tumTs]

train_im= Dataset(data=train_files, transform=train_transforms)
val_im= Dataset(data=val_files, transform=val_transforms)
test_im= Dataset(data=test_files, transform=val_transforms)

train_lb= Dataset(data=train_files_l, transform=lb_transforms)
val_lb= Dataset(data=val_files_l, transform=lb_transforms)
test_lb= Dataset(data=test_files_l, transform=lb_transforms)

#Make custom dataset for image classification
#It gets images , tumors and labels of images. Concatenate images and tumors on the first dimension, return them as the input to the training model and the labels are 0,1,2,3 corresponding to our classes.

# In[]:

class KDataset(Dataset):

    def __init__(self, image_files, tumor_files, labels):
        self.image_files = image_files
        self.tumor_files = tumor_files
        self.labels = labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        im =  self.image_files[index]['image']
        lb = self.tumor_files[index]['label']
        imlb = torch.cat((im, lb),0)
#         print(index)
        
        return imlb, self.labels[index]

#Make train, validation and test dataloaders

# In[]:

train_ds = KDataset(train_im, train_lb, trainY)
train_loader = DataLoader(train_ds, batch_size = 10, shuffle = True, pin_memory=True)

val_ds = KDataset(val_im, val_lb, valY)
val_loader = DataLoader(val_ds, batch_size = 2, pin_memory=True)

test_ds = KDataset(test_im, test_lb, testY)
test_loader = DataLoader(test_ds, batch_size = 2, pin_memory=True)

#check dataloader

# In[]:

dataiter = iter(train_loader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)


# Define network and optimizer
# 1. Set learning rate for how much the model is updated per batch.
# 2. Set total epoch number, as we have shuffle and random transforms, so the training data of every epoch is different.
# 3. Use Efficientnet b7 from Efficientnet3d pytorch on github and move to GPU device.
# 4. Use Adam optimizer.
# 5. Use onehot for classification

# In[]:

device = torch.device("cuda:0")
model = EfficientNet3D.from_name("efficientnet-b7", override_params={'num_classes': num_class}, in_channels=2)
model = model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
epoch_num = 50
val_interval = 1
act = Activations(softmax=True)
to_onehot = AsDiscrete(to_onehot=num_class, n_classes=num_class)

#Model training
#Execute a typical PyTorch training that run epoch loop and step loop, and do validation after every epoch. Will save the model weights to file if got best validation accuracy.

# In[]:

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
auc_metric = ROCAUCMetric()
metric_values = list()
start_time = time.time()
for epoch in range(epoch_num):
    print('-' * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds) // train_loader.batch_size
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [to_onehot(i) for i in y]
            y_pred_act = [act(i) for i in y_pred]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values.append(auc_result)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            if acc_metric > best_metric:
                best_metric = acc_metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), 'best_model_ISUP_fold0.pth')
                print('saved new best metric model')
            print(f"current epoch: {epoch + 1} current AUC: {auc_result:.4f}"
                  f" current accuracy: {acc_metric:.4f} best accuracy: {best_metric:.4f}"
                  f" at epoch: {best_metric_epoch}")
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")

#Fold1

#Choosing cases from total augmented dataset based on the indexes that we obtained before(train,validation,test)

# In[]:

cases_train_fold1 = []
for i in train_index_fold1:
    cases_train_fold1.append(order_of_cases[i])
cases_val_fold1 = []
for j in val_index_fold1:
    cases_val_fold1.append(order_of_cases[j])
cases_test_fold1 = []
for m in test_index_fold1:
    cases_test_fold1.append(order_of_cases[m])

#Getting image file, tumor file and class label for train,validation and test data

# In[]:

trainX_1, trainY_1, tumTr_1 = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_train_fold1:
        if int(item[66:71]) == j:
            trainX_1.append(item)
            trainY_1.append(int(item[51:52]))
            tumTr_1.append(tumor_file_list_aug[i])
valX_1, valY_1, tumV_1 = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_val_fold1:
        if int(item[66:71]) == j:
            valX_1.append(item)
            valY_1.append(int(item[51:52]))
            tumV_1.append(tumor_file_list_aug[i])
testX_1, testY_1, tumTs_1 = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_test_fold1:
        if int(item[66:71]) == j:
            testX_1.append(item)
            testY_1.append(int(item[51:52]))
            tumTs_1.append(tumor_file_list_aug[i])

print("Training count =",len(trainX_1),"Validation count =", len(valX_1), "Test count =",len(testX_1))

#Making main image datasets and tumor image datasets seperately

# In[]:

train_files_1 = [{"image": image_name} for image_name in trainX_1]
val_files_1 = [{"image": image_name} for image_name in valX_1]
test_files_1 = [{"image": image_name} for image_name in testX_1]

train_files_l_1 = [{"label": label_name} for label_name in tumTr_1]
val_files_l_1 = [{"label": label_name} for label_name in tumV_1]
test_files_l_1 = [{"label": label_name} for label_name in tumTs_1]

train_im_1= Dataset(data=train_files_1, transform=train_transforms)
val_im_1= Dataset(data=val_files_1, transform=val_transforms)
test_im_1= Dataset(data=test_files_1, transform=val_transforms)

train_lb_1= Dataset(data=train_files_l_1, transform=lb_transforms)
val_lb_1= Dataset(data=val_files_l_1, transform=lb_transforms)
test_lb_1= Dataset(data=test_files_l_1, transform=lb_transforms)

#Make train, validation and test dataloaders

# In[]:

train_ds_1 = KDataset(train_im_1, train_lb_1, trainY_1)
train_loader_1 = DataLoader(train_ds_1, batch_size = 10, shuffle = True, pin_memory=True)

val_ds_1 = KDataset(val_im_1, val_lb_1, valY_1)
val_loader_1 = DataLoader(val_ds_1, batch_size = 2, pin_memory=True)

test_ds_1 = KDataset(test_im_1, test_lb_1, testY_1)
test_loader_1 = DataLoader(test_ds_1, batch_size = 2, pin_memory=True)

#Model training

# In[]:

best_metric_1 = -1
best_metric_epoch_1 = -1
epoch_loss_values_1 = list()
auc_metric = ROCAUCMetric()
metric_values_1 = list()
for epoch in range(epoch_num):
    print('-' * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader_1:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds_1) // train_loader_1.batch_size}, train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds_1) // train_loader_1.batch_size
    epoch_loss /= step
    epoch_loss_values_1.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader_1:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [to_onehot(i) for i in y]
            y_pred_act = [act(i) for i in y_pred]
            auc_metric(y_pred_act, y_onehot)
            auc_result_1 = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values_1.append(auc_result_1)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric_1 = acc_value.sum().item() / len(acc_value)
            if acc_metric_1 > best_metric_1:
                best_metric_1 = acc_metric_1
                best_metric_epoch_1 = epoch + 1
                torch.save(model.state_dict(), 'best_model_ISUP_fold1.pth')
                print('saved new best metric model')
            print(f"current epoch: {epoch + 1} current AUC: {auc_result_1:.4f}"
                  f" current accuracy: {acc_metric_1:.4f} best accuracy: {best_metric_1:.4f}"
                  f" at epoch: {best_metric_epoch_1}")
print(f"train completed, best_metric: {best_metric_1:.4f} at epoch: {best_metric_epoch_1}")


#Fold2

#Choosing cases from total augmented dataset based on the indexes that we obtained before(train,validation,test)

# In[]:

cases_train_fold2 = []
for i in train_index_fold2:
    cases_train_fold2.append(order_of_cases[i])
cases_val_fold2 = []
for j in val_index_fold2:
    cases_val_fold2.append(order_of_cases[j])
cases_test_fold2 = []
for m in test_index_fold2:
    cases_test_fold2.append(order_of_cases[m])

#Getting image file, tumor file and class label for train,validation and test data

# In[]:

trainX_2, trainY_2, tumTr_2 = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_train_fold2:
        if int(item[66:71]) == j:
            trainX_2.append(item)
            trainY_2.append(int(item[51:52]))
            tumTr_2.append(tumor_file_list_aug[i])
valX_2, valY_2, tumV_2 = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_val_fold2:
        if int(item[66:71]) == j:
            valX_2.append(item)
            valY_2.append(int(item[51:52]))
            tumV_2.append(tumor_file_list_aug[i])
testX_2, testY_2, tumTs_2 = [], [], []
for i,item in enumerate(image_file_list_aug):
    for j in cases_test_fold2:
        if int(item[66:71]) == j:
            testX_2.append(item)
            testY_2.append(int(item[51:52]))
            tumTs_2.append(tumor_file_list_aug[i])

print("Training count =",len(trainX_2),"Validation count =", len(valX_2), "Test count =",len(testX_2))


#Making main image datasets and tumor image datasets seperately

# In[]:

train_files_2 = [{"image": image_name} for image_name in trainX_2]
val_files_2 = [{"image": image_name} for image_name in valX_2]
test_files_2 = [{"image": image_name} for image_name in testX_2]

train_files_l_2 = [{"label": label_name} for label_name in tumTr_2]
val_files_l_2 = [{"label": label_name} for label_name in tumV_2]
test_files_l_2 = [{"label": label_name} for label_name in tumTs_2]

train_im_2= Dataset(data=train_files_2, transform=train_transforms)
val_im_2= Dataset(data=val_files_2, transform=val_transforms)
test_im_2= Dataset(data=test_files_2, transform=val_transforms)

train_lb_2= Dataset(data=train_files_l_2, transform=lb_transforms)
val_lb_2= Dataset(data=val_files_l_2, transform=lb_transforms)
test_lb_2= Dataset(data=test_files_l_2, transform=lb_transforms)

#Make train, validation and test dataloaders

# In[]:

train_ds_2 = KDataset(train_im_2, train_lb_2, trainY_2)
train_loader_2 = DataLoader(train_ds_2, batch_size = 10, shuffle = True, pin_memory=True)

val_ds_2 = KDataset(val_im_2, val_lb_2, valY_2)
val_loader_2 = DataLoader(val_ds_2, batch_size = 2, pin_memory=True)

test_ds_2 = KDataset(test_im_2, test_lb_2, testY_2)
test_loader_2 = DataLoader(test_ds_2, batch_size = 2, pin_memory=True)

#Model training

# In[]:

best_metric_2 = -1
best_metric_epoch_2 = -1
epoch_loss_values_2 = list()
auc_metric = ROCAUCMetric()
metric_values_2 = list()
for epoch in range(epoch_num):
    print('-' * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader_2:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds_2) // train_loader_2.batch_size}, train_loss: {loss.item():.4f}")
        epoch_len = len(train_ds_2) // train_loader_2.batch_size
    epoch_loss /= step
    epoch_loss_values_2.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            for val_data in val_loader_2:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                y = torch.cat([y, val_labels], dim=0)
            y_onehot = [to_onehot(i) for i in y]
            y_pred_act = [act(i) for i in y_pred]
            auc_metric(y_pred_act, y_onehot)
            auc_result_2 = auc_metric.aggregate()
            auc_metric.reset()
            del y_pred_act, y_onehot
            metric_values_2.append(auc_result_2)
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric_2 = acc_value.sum().item() / len(acc_value)
            if acc_metric_2 > best_metric_2:
                best_metric_2 = acc_metric_2
                best_metric_epoch_2 = epoch + 1
                torch.save(model.state_dict(), 'best_model_ISUP_fold2.pth')
                print('saved new best metric model')
            print(f"current epoch: {epoch + 1} current AUC: {auc_result_2:.4f}"
                  f" current accuracy: {acc_metric_2:.4f} best accuracy: {best_metric_2:.4f}"
                  f" at epoch: {best_metric_epoch_2}")
print(f"train completed, best_metric: {best_metric_2:.4f} at epoch: {best_metric_epoch_2}")

# Plot the loss and metric

# Fold0

# In[]:


plt.figure('train', (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel('epoch')
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Validation: Area under the ROC curve")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel('epoch')
plt.plot(x, y)
plt.show()


# Fold1

# In[]:


plt.figure('train', (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values_1))]
y = epoch_loss_values_1
plt.xlabel('epoch')
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Validation: Area under the ROC curve")
x = [val_interval * (i + 1) for i in range(len(metric_values_1))]
y = metric_values_1
plt.xlabel('epoch')
plt.plot(x, y)
plt.show()


# Fold2

# In[]:


plt.figure('train', (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values_2))]
y = epoch_loss_values_2
plt.xlabel('epoch')
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Validation: Area under the ROC curve")
x = [val_interval * (i + 1) for i in range(len(metric_values_2))]
y = metric_values_2
plt.xlabel('epoch')
plt.plot(x, y)
plt.show()


#Evaluate the model on datasets
#After training and validation, we already got the best model on validation test. We need to evaluate the model on test dataset to check whether it's robust and not over-fitting. We'll use these predictions to generate a classification report.

#Fold0

# train

# In[]:

model.load_state_dict(torch.load('best_model_ISUP_fold0.pth'))
model.eval()
y_t_true = list()
y_t_pred = list()
with torch.no_grad():
    for train_data in train_loader:
        train_images, train_labels = train_data[0].to(device), train_data[1].to(device)
        pred = model(train_images).argmax(dim=1)
        for i in range(len(pred)):
            y_t_true.append(train_labels[i].item())
            y_t_pred.append(pred[i].item())


# test

# In[]:


model.load_state_dict(torch.load('best_model_ISUP_fold0.pth'))
model.eval()
y_true_0 = list()
y_pred_0 = list()
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true_0.append(test_labels[i].item())
            y_pred_0.append(pred[i].item())


# In[]:


pp0 = pd.DataFrame(data = y_true_0, columns = ['true'])
pp1 = pd.DataFrame(data = y_pred_0, columns = ['predicted'])
pp2 = pd.DataFrame(data = testX, columns = ['test'])
pp3 = pd.concat([pp0, pp1,pp2],axis=1)
pp3.to_csv('label_test_0.csv',index=False)


# validation

# In[]:


model.load_state_dict(torch.load('best_model_ISUP_fold0.pth'))
model.eval()
y_v_true = list()
y_v_pred = list()
with torch.no_grad():
    for val_data in val_loader:
        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
        pred = model(val_images).argmax(dim=1)
        for i in range(len(pred)):
            y_v_true.append(val_labels[i].item())
            y_v_pred.append(pred[i].item())


#Fold1

# train

# In[]:


model.load_state_dict(torch.load('best_metric_ISUP_fold1.pth'))
model.eval()
y_t_true_1 = list()
y_t_pred_1 = list()
with torch.no_grad():
    for train_data in train_loader_1:
        train_images, train_labels = train_data[0].to(device), train_data[1].to(device)
        pred = model(train_images).argmax(dim=1)
        for i in range(len(pred)):
            y_t_true_1.append(train_labels[i].item())
            y_t_pred_1.append(pred[i].item())


# test

# In[]:


model.load_state_dict(torch.load('best_metric_ISUP_fold1.pth'))
model.eval()
y_true_1 = list()
y_pred_1 = list()
with torch.no_grad():
    for test_data in test_loader_1:
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true_1.append(test_labels[i].item())
            y_pred_1.append(pred[i].item())


# In[]:

pp0 = pd.DataFrame(data = y_true_1, columns = ['true'])
pp1 = pd.DataFrame(data = y_pred_1, columns = ['predicted'])
pp2 = pd.DataFrame(data = testX_1, columns = ['test'])
pp3 = pd.concat([pp0, pp1,pp2],axis=1)
pp3.to_csv('label_test_1.csv',index=False)

# validation

# In[]:

model.load_state_dict(torch.load('best_metric_ISUP_fold1.pth'))
model.eval()
y_v_true_1 = list()
y_v_pred_1 = list()
with torch.no_grad():
    for val_data in val_loader_1:
        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
        pred = model(val_images).argmax(dim=1)
        for i in range(len(pred)):
            y_v_true_1.append(val_labels[i].item())
            y_v_pred_1.append(pred[i].item())


#Fold2

# train

# In[]:


model.load_state_dict(torch.load('best_metric_ISUP_fold2.pth'))
model.eval()
y_t_true_2 = list()
y_t_pred_2 = list()
with torch.no_grad():
    for train_data in train_loader_2:
        train_images, train_labels = train_data[0].to(device), train_data[1].to(device)
        pred = model(train_images).argmax(dim=1)
        for i in range(len(pred)):
            y_t_true_2.append(train_labels[i].item())
            y_t_pred_2.append(pred[i].item())


# test

# In[]:


model.load_state_dict(torch.load('best_metric_ISUP_fold2.pth'))
model.eval()
y_true_2 = list()
y_pred_2 = list()
with torch.no_grad():
    for test_data in test_loader_2:
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true_2.append(test_labels[i].item())
            y_pred_2.append(pred[i].item())


# validation

# In[]:


model.load_state_dict(torch.load('best_metric_ISUP_fold2.pth'))
model.eval()
y_v_true_2 = list()
y_v_pred_2 = list()
with torch.no_grad():
    for val_data in val_loader_2:
        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
        pred = model(val_images).argmax(dim=1)
        for i in range(len(pred)):
            y_v_true_2.append(val_labels[i].item())
            y_v_pred_2.append(pred[i].item())


#Accuracy metrics

#fold0

# In[]:


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
report_test_0 = classification_report(y_true_0, y_pred_0, target_names=class_names, digits=6)
report_train_0 = classification_report(y_t_true, y_t_pred, target_names=class_names, digits=6)
report_val_0 = classification_report(y_v_true, y_v_pred, target_names=class_names, digits=6)

precision_fold0_test = precision_score(y_true_0, y_pred_0, average='weighted')
precision_fold0_train = precision_score(y_t_true, y_t_pred, average='weighted')
precision_fold0_val = precision_score(y_v_true, y_v_pred, average='weighted')

recall_fold0_test = recall_score(y_true_0, y_pred_0, average='weighted')
recall_fold0_train = recall_score(y_t_true, y_t_pred, average='weighted')
recall_fold0_val = recall_score(y_v_true, y_v_pred, average='weighted')

f1_fold0_test = f1_score(y_true_0, y_pred_0, average='weighted')
f1_fold0_train = f1_score(y_t_true, y_t_pred, average='weighted')
f1_fold0_val = f1_score(y_v_true, y_v_pred, average='weighted')


# In[]:


print(report_test_0)
print(report_train_0)
print(report_val_0)


#fold1

# In[]:


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
report_test_1 = classification_report(y_true_1, y_pred_1, target_names=class_names, digits=6)
report_train_1 = classification_report(y_t_true_1, y_t_pred_1, target_names=class_names, digits=6)
report_val_1 = classification_report(y_v_true_1, y_v_pred_1, target_names=class_names, digits=6)

precision_fold1_test = precision_score(y_true_1, y_pred_1, average='weighted')
precision_fold1_train = precision_score(y_t_true_1, y_t_pred_1, average='weighted')
precision_fold1_val = precision_score(y_v_true_1, y_v_pred_1, average='weighted')

recall_fold1_test = recall_score(y_true_1, y_pred_1, average='weighted')
recall_fold1_train = recall_score(y_t_true_1, y_t_pred_1, average='weighted')
recall_fold1_val = recall_score(y_v_true_1, y_v_pred_1, average='weighted')

f1_fold1_test = f1_score(y_true_1, y_pred_1, average='weighted')
f1_fold1_train = f1_score(y_t_true_1, y_t_pred_1, average='weighted')
f1_fold1_val = f1_score(y_v_true_1, y_v_pred_1, average='weighted')


# In[]:


print(report_test_1)
print(report_train_1)
print(report_val_1)


#fold2

# In[]:


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
report_test_2 = classification_report(y_true_2, y_pred_2, target_names=class_names, digits=6)
report_train_2 = classification_report(y_t_true_2, y_t_pred_2, target_names=class_names, digits=6)
report_val_2 = classification_report(y_v_true_2, y_v_pred_2, target_names=class_names, digits=6)

precision_fold2_test = precision_score(y_true_2, y_pred_2, average='weighted')
precision_fold2_train = precision_score(y_t_true_2, y_t_pred_2, average='weighted')
precision_fold2_val = precision_score(y_v_true_2, y_v_pred_2, average='weighted')

recall_fold2_test = recall_score(y_true_2, y_pred_2, average='weighted')
recall_fold2_train = recall_score(y_t_true_2, y_t_pred_2, average='weighted')
recall_fold2_val = recall_score(y_v_true_2, y_v_pred_2, average='weighted')

f1_fold2_test = f1_score(y_true_2, y_pred_2, average='weighted')
f1_fold2_train = f1_score(y_t_true_2, y_t_pred_2, average='weighted')
f1_fold2_val = f1_score(y_v_true_2, y_v_pred_2, average='weighted')


# In[]:


print(report_test_2)
print(report_train_2)
print(report_val_2)


#Finding the average precision,recall,f-1

# In[]:


import statistics as st

mean3fold_precision_test = st.mean([precision_fold0_test, precision_fold1_test, precision_fold2_test])
mean3fold_precision_test = round(mean3fold_precision_test, 2)

mean3fold_precision_train = st.mean([precision_fold0_train, precision_fold1_train, precision_fold2_train])
mean3fold_precision_train = round(mean3fold_precision_train, 2)

mean3fold_precision_val = st.mean([precision_fold0_val, precision_fold1_val, precision_fold2_val])
mean3fold_precision_val = round(mean3fold_precision_val, 2)

mean3fold_recall_test = st.mean([recall_fold0_test, recall_fold1_test, recall_fold2_test])
mean3fold_recall_test = round(mean3fold_recall_test, 2)

mean3fold_recall_train = st.mean([recall_fold0_train, recall_fold1_train, recall_fold2_train])
mean3fold_recall_train = round(mean3fold_recall_train, 2)

mean3fold_recall_val = st.mean([recall_fold0_val, recall_fold1_val, recall_fold2_val])
mean3fold_recall_val = round(mean3fold_recall_val, 2)

mean3fold_fscore_test = st.mean([f1_fold0_test, f1_fold1_test, f1_fold2_test])
mean3fold_fscore_test = round(mean3fold_fscore_test, 2)

mean3fold_fscore_train = st.mean([f1_fold0_train, f1_fold1_train, f1_fold2_train])
mean3fold_fscore_train = round(mean3fold_fscore_train, 2)

mean3fold_fscore_val = st.mean([f1_fold0_val, f1_fold1_val, f1_fold2_val])
mean3fold_fscore_val = round(mean3fold_fscore_val, 2)


# In[]:


metrics_dataframe = pd.DataFrame({'col1' : [mean3fold_precision_train, mean3fold_precision_val,mean3fold_precision_test],
                                 'col2' : [mean3fold_recall_train, mean3fold_recall_val, mean3fold_recall_test],
                                 'col3' : [mean3fold_fscore_train, mean3fold_fscore_val, mean3fold_fscore_test]})

metrics_dataframe.columns = ['precision','recall','f_score']
metrics_dataframe.index = ['train','validation', 'test']
metrics_dataframe
