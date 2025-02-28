# PRODIGY_ML_03

This notebook demonstrates how to perform image classification using a Support Vector Machine (SVM) classifier on the Dogs vs. Cats dataset.

## Dataset
The dataset used in this task is the Dogs vs. Cats dataset from Kaggle. It is downloaded and extracted using the opendatasets library.

## Setup
Install necessary libraries and download the dataset:
Python
```
!pip install opendatasets --quiet
import opendatasets as od
od.download("https://www.kaggle.com/c/dogs-vs-cats/data")
```

## Extract the dataset:
Python
```
import zipfile
with zipfile.ZipFile('dogs-vs-cats/train.zip', 'r') as zip_ref:
    zip_ref.extractall('working/train')
with zipfile.ZipFile('dogs-vs-cats/test1.zip', 'r') as zip_ref:
    zip_ref.extractall('working/test')
```

## Define the directories for training and testing data:
Python
```
train_dir = 'working/train/train'
test_dir = 'working/test/test1'
```

## Image Transformations
Python
```
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CustomImageDataset(image_dir=train_dir, transform=transform)
test_dataset = CustomImageDataset(image_dir=test_dir, transform=transform)
```

## Training and Evaluation
SVM Acuuracy 68%
