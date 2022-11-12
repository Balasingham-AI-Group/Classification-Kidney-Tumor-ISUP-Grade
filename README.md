# Classification-Kidney-Tumor-ISUP-Grade
 In this project, we propose a novel DL-based framework to automate the classification of kidney tumors based on the International Society of Urological Pathology (ISUP) renal tumor grading system in CT volumes. 
<br>We use some CT images(3D) from kidney and try to classify them in four classes.
<br>Our classification is based on ISUP grades that is related to the size of tumor.
<br>The first dataset had 300 images and labels based on Monai transformers we made synthetic data and finally we had 2000 images for training model.
<br>For preprocessing our dataset we used Monai library too.
<br>We used **efficientnet-b7**, the state of the art architecture for image classification.
<br>Before we do the image classification we split the dataset to train, validation and test based on the number of dead people. You can find it in Make_T&V&T_Dataset_Indexes.py. 70 percent of dead people belong to train and 30 percent of dead people belong to test.
<br>We saved the indices in a csv file to use it for image classification. And for image classification you can find it in Classification_indexesBased.py.
<br>The trained model is saved in this link : https://drive.google.com/file/d/1FgR79XtAVXQpDR4Q7vbxy08yEX-_1KtL/view?usp=sharing. If you want to load the model you should make a model like this:
```
device = torch.device("cuda:0")
model = EfficientNet3D.from_name("efficientnet-b7", override_params={'num_classes': num_class}, in_channels=2)
model = model.to(device)
```
